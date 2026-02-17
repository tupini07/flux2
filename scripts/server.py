"""
HTTP server wrapping the Flux.2 generation pipeline.

Multi-GPU layout (auto-detected):
  - 4+ GPUs: flow model sharded across GPU 0+1, text encoder across GPU 2+3, AE on GPU 0
  - 2-3 GPUs: flow model on GPU 0, text encoder on GPU 1 (+ GPU 2 if available), AE on GPU 0
  - 1 GPU:   all models on GPU 0 with CPU offloading (flow model swaps CPU↔GPU)

Requests are serialised through an asyncio queue (one generation at a time).

Usage:
    python scripts/server.py [--model_name flux.2-dev] [--host 0.0.0.0] [--port 8192]
                             [--debug_mode]

Endpoint:
    POST /generate  (multipart/form-data)
        Fields:
          - prompt          (str, required)
          - input_images    (file[], optional) — reference/conditioning images
          - width           (int, optional, default 1360)
          - height          (int, optional, default 768)
          - num_steps       (int, optional)
          - guidance        (float, optional)
          - seed            (int, optional)
          - match_image_size (int, optional) — index into input_images to auto-match dims
        Response: image/png
"""

import asyncio
import io
import random
import sys
from typing import List, Optional

import torch
import uvicorn
from einops import rearrange
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import ExifTags, Image

from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model

app = FastAPI(title="Flux.2 Generation Server")

# Global model references, populated at startup
_model = None
_ae = None
_text_encoder = None
_model_info = None
_model_name = None
_cpu_offloading = False
# Device where denoise inputs should be created (first device of the flow model)
_flow_device: torch.device = torch.device("cuda:0")
_ae_device: torch.device = torch.device("cuda:0")

# Serialisation queue — only one generation runs at a time
_queue: asyncio.Queue = asyncio.Queue()
_queue_lock: asyncio.Lock = asyncio.Lock()


def _shard_flow_model_manual(model, gpu_ids: list[int]):
    """
    Manually shard Flux2 across GPUs by moving submodules.

    Flux2.forward precomputes modulation vectors (vec, mods) on the first GPU,
    then passes them to every block. accelerate.dispatch_model can't handle this
    broadcast pattern, so we wrap forward to move tensors between devices.

    Layout: embeddings + double_blocks on gpu_ids[0],
            single_blocks + final_layer on gpu_ids[1].
    """
    dev0 = torch.device(f"cuda:{gpu_ids[0]}")
    dev1 = torch.device(f"cuda:{gpu_ids[1]}")

    # Embeddings + modulations + double blocks stay on first GPU (already there)
    model.pe_embedder.to(dev0)
    model.img_in.to(dev0)
    model.txt_in.to(dev0)
    model.time_in.to(dev0)
    if hasattr(model, "guidance_in"):
        model.guidance_in.to(dev0)
    model.double_stream_modulation_img.to(dev0)
    model.double_stream_modulation_txt.to(dev0)
    model.single_stream_modulation.to(dev0)
    model.double_blocks.to(dev0)

    # Single blocks + final layer on second GPU
    model.single_blocks.to(dev1)
    model.final_layer.to(dev1)

    # Patch forward to handle device transitions
    _orig_forward = model.forward

    def _sharded_forward(x, x_ids, timesteps, ctx, ctx_ids, guidance):
        num_txt_tokens = ctx.shape[1]

        from flux2.model import timestep_embedding

        timestep_emb = timestep_embedding(timesteps, 256)
        vec = model.time_in(timestep_emb)
        if model.use_guidance_embed:
            guidance_emb = timestep_embedding(guidance, 256)
            vec = vec + model.guidance_in(guidance_emb)

        double_block_mod_img = model.double_stream_modulation_img(vec)
        double_block_mod_txt = model.double_stream_modulation_txt(vec)
        single_block_mod, _ = model.single_stream_modulation(vec)

        img = model.img_in(x)
        txt = model.txt_in(ctx)
        pe_x = model.pe_embedder(x_ids)
        pe_ctx = model.pe_embedder(ctx_ids)

        for block in model.double_blocks:
            img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt)

        # Move to second GPU for single blocks
        img = torch.cat((txt, img), dim=1).to(dev1)
        pe = torch.cat((pe_ctx, pe_x), dim=2).to(dev1)
        single_block_mod = tuple(t.to(dev1) for t in single_block_mod)
        vec_dev1 = vec.to(dev1)

        for block in model.single_blocks:
            img = block(img, pe, single_block_mod)

        img = img[:, num_txt_tokens:, ...]
        img = model.final_layer(img, vec_dev1)
        # Move output back to first GPU
        return img.to(dev0)

    model.forward = _sharded_forward
    print(
        f"  Flow model manually sharded: double_blocks on cuda:{gpu_ids[0]}, single_blocks on cuda:{gpu_ids[1]}"
    )
    return model


def _load_text_encoder_sharded(model_name: str, gpu_ids: list[int]):
    """Load the Mistral text encoder sharded across multiple GPUs via device_map."""
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration, pipeline

    from flux2.text_encoder import (
        MAX_LENGTH,
        Mistral3SmallEmbedder,
        UPSAMPLING_MAX_IMAGE_SIZE,
    )

    # Build max_memory dict restricting to only our target GPUs
    max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3)}GiB" for i in gpu_ids}

    embedder = Mistral3SmallEmbedder.__new__(Mistral3SmallEmbedder)
    torch.nn.Module.__init__(embedder)

    print(f"  Loading Mistral model with device_map across GPUs {gpu_ids}...")
    embedder.model = Mistral3ForConditionalGeneration.from_pretrained(
        "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    embedder.processor = AutoProcessor.from_pretrained(
        "mistralai/Mistral-Small-3.1-24B-Instruct-2503", use_fast=False
    )
    embedder.yes_token, embedder.no_token = embedder.processor.tokenizer.encode(
        ["yes", "no"], add_special_tokens=False
    )
    embedder.max_length = MAX_LENGTH
    embedder.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE
    # NSFW classifier is small — put on the first text encoder GPU
    embedder.nsfw_classifier = pipeline(
        "image-classification", model="Falconsai/nsfw_image_detection", device=f"cuda:{gpu_ids[0]}"
    )
    return embedder


def load_models(
    model_name: str = "flux.2-dev",
    debug_mode: bool = False,
):
    global _model, _ae, _text_encoder, _model_info, _model_name
    global _cpu_offloading, _flow_device, _ae_device

    _model_name = model_name
    _model_info = FLUX2_MODEL_INFO[model_name]

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s)")

    if num_gpus >= 4:
        flow_gpus = [0, 1]
        te_gpus = [2, 3]
        ae_dev = "cuda:0"
        _cpu_offloading = False
    elif num_gpus >= 2:
        flow_gpus = [0]
        te_gpus = list(range(1, num_gpus))
        ae_dev = "cuda:0"
        _cpu_offloading = False
    else:
        flow_gpus = []
        te_gpus = [0]
        ae_dev = "cuda:0"
        _cpu_offloading = True

    # --- Text encoder ---
    if len(te_gpus) >= 2:
        print(f"Loading text encoder for {model_name} across GPUs {te_gpus}...")
        _text_encoder = _load_text_encoder_sharded(model_name, te_gpus)
    else:
        from flux2.util import load_text_encoder

        print(f"Loading text encoder for {model_name} on cuda:{te_gpus[0]}...")
        _text_encoder = load_text_encoder(model_name, device=f"cuda:{te_gpus[0]}")
    _text_encoder.eval()

    # --- Flow model ---
    if _cpu_offloading:
        print(f"Loading flow model for {model_name} on CPU (will offload to GPU for inference)...")
        _model = load_flow_model(model_name, debug_mode=debug_mode, device="cpu")
        _flow_device = torch.device("cuda:0")
    elif len(flow_gpus) >= 2:
        print(f"Loading flow model for {model_name}, will shard across GPUs {flow_gpus}...")
        _model = load_flow_model(model_name, debug_mode=debug_mode, device=f"cuda:{flow_gpus[0]}")
        _model = _shard_flow_model_manual(_model, flow_gpus)
        _flow_device = torch.device(f"cuda:{flow_gpus[0]}")
    else:
        dev = f"cuda:{flow_gpus[0]}"
        print(f"Loading flow model for {model_name} on {dev}...")
        _model = load_flow_model(model_name, debug_mode=debug_mode, device=dev)
        _flow_device = torch.device(dev)

    # --- Autoencoder ---
    print(f"Loading autoencoder for {model_name} on {ae_dev}...")
    _ae = load_ae(model_name, device=ae_dev)
    _ae.eval()
    _ae_device = torch.device(ae_dev)

    print("All models loaded and ready.")


def _run_generation(
    prompt: str,
    img_ctx: list[Image.Image],
    w: int,
    h: int,
    steps: int,
    guid: float,
    s: int,
) -> bytes:
    """Synchronous generation — must be called from the worker loop."""
    with torch.no_grad():
        ref_tokens, ref_ids = encode_image_refs(_ae, img_ctx)

        if _model_info["guidance_distilled"]:
            ctx = _text_encoder([prompt]).to(torch.bfloat16)
        else:
            ctx_empty = _text_encoder([""]).to(torch.bfloat16)
            ctx_prompt = _text_encoder([prompt]).to(torch.bfloat16)
            ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Move text embeddings to the flow model's device
        ctx = ctx.to(_flow_device)
        ctx_ids = ctx_ids.to(_flow_device)
        if ref_tokens is not None:
            ref_tokens = ref_tokens.to(_flow_device)
            ref_ids = ref_ids.to(_flow_device)

        if _cpu_offloading:
            _text_encoder.cpu()
            torch.cuda.empty_cache()
            _model.to(_flow_device)

        shape = (1, 128, h // 16, w // 16)
        generator = torch.Generator(device=_flow_device).manual_seed(s)
        randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=_flow_device)
        x, x_ids = batched_prc_img(randn)

        timesteps = get_schedule(steps, x.shape[1])
        if _model_info["guidance_distilled"]:
            x = denoise(
                _model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guid,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
        else:
            x = denoise_cfg(
                _model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guid,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
        x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)

        # Decode on the AE device
        x = _ae.decode(x.to(_ae_device)).float()

        if _cpu_offloading:
            _model.cpu()
            torch.cuda.empty_cache()
            _text_encoder.to(_flow_device)

    x = x.clamp(-1, 1)
    x = rearrange(x[0], "c h w -> h w c")
    img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

    # Content moderation on output
    if _text_encoder.test_image(img):
        raise ValueError("Generated image flagged as unsuitable.")

    exif_data = Image.Exif()
    exif_data[ExifTags.Base.Software] = "AI generated;flux2"
    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
    buf = io.BytesIO()
    img.save(buf, format="PNG", exif=exif_data)
    return buf.getvalue()


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    input_images: List[UploadFile] = File(default=[]),
    width: Optional[int] = Form(default=None),
    height: Optional[int] = Form(default=None),
    num_steps: Optional[int] = Form(default=None),
    guidance: Optional[float] = Form(default=None),
    seed: Optional[int] = Form(default=None),
    match_image_size: Optional[int] = Form(default=None),
):
    defaults = _model_info.get("defaults", {})
    w = width if width is not None else 1360
    h = height if height is not None else 768
    steps = num_steps if num_steps is not None else defaults.get("num_steps", 50)
    guid = guidance if guidance is not None else defaults.get("guidance", 4.0)
    s = seed if seed is not None else random.randrange(2**31)

    # Validate against fixed params
    fixed_params = _model_info.get("fixed_params", set())
    if "num_steps" in fixed_params and steps != defaults["num_steps"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{_model_name}' requires num_steps={defaults['num_steps']}",
        )
    if "guidance" in fixed_params and guid != defaults["guidance"]:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{_model_name}' requires guidance={defaults['guidance']}",
        )

    # Read uploaded images (IO-bound, fine to do before queuing)
    img_ctx: list[Image.Image] = []
    for upload in input_images:
        data = await upload.read()
        img_ctx.append(Image.open(io.BytesIO(data)).convert("RGB"))

    # Content moderation on prompt
    if _text_encoder.test_txt(prompt):
        raise HTTPException(
            status_code=400,
            detail="Prompt flagged for potential copyright or public persona concerns.",
        )

    # Content moderation on input images
    for i, img in enumerate(img_ctx):
        if _text_encoder.test_image(img):
            raise HTTPException(
                status_code=400,
                detail=f"Input image {i} flagged as unsuitable.",
            )

    # Match dimensions from a reference image
    if match_image_size is not None:
        if match_image_size < 0 or match_image_size >= len(img_ctx):
            raise HTTPException(
                status_code=400,
                detail=f"match_image_size={match_image_size} out of range (0-{len(img_ctx)-1})",
            )
        w, h = img_ctx[match_image_size].size

    # Serialise GPU work through the queue lock
    loop = asyncio.get_event_loop()
    async with _queue_lock:
        try:
            png_bytes = await loop.run_in_executor(
                None, _run_generation, prompt, img_ctx, w, h, steps, guid, s
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            print(f"Generation error: {type(e).__name__}: {e}", file=sys.stderr)
            raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    return Response(content=png_bytes, media_type="image/png")


def main(
    model_name: str = "flux.2-dev",
    host: str = "0.0.0.0",
    port: int = 8192,
    debug_mode: bool = False,
):
    load_models(model_name=model_name, debug_mode=debug_mode)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
