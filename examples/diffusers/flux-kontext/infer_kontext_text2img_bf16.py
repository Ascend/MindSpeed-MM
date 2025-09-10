# Copyright 2025 Huawei Technologies Co., Ltd

import os

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from transformer_patches import apply_patches

apply_patches()

DEVICE = "npu"
MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"  # Model path for Flux Kontext
OUTPUT_PATH = "./infer_result"  # Output path

IMAGE = "./flux_cat.png"  # input image
OUTPUT_IMAGE_1 = "flux_kontext_1.png"  # output image 1
OUTPUT_IMAGE_2 = "flux_kontext_2.png"  # output image 2

GENERATOR = torch.Generator(device="cpu").manual_seed(42)  # Generator for inference
STEPS = 50  # Number of steps for inference
GUIDANCE = 2.5  # Guidance scale for inference
RESOLUTION = 1024  # Resolution for inference

os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create the output folder

pipe = FluxKontextPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

pipe = pipe.to(DEVICE)
pipe.transformer.set_attention_backend("_native_npu")

image = load_image(IMAGE).convert("RGB")
prompt = "Relight the scene with a soft, diffused foggy glow emanating from the top left side"
image = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=STEPS,
    height=RESOLUTION,
    width=RESOLUTION,
    guidance_scale=GUIDANCE,
    generator=GENERATOR,
).images[0]
saved_image = OUTPUT_PATH + "/" + OUTPUT_IMAGE_1
image.save(saved_image)

image = load_image(saved_image).convert("RGB")
prompt = "Change the sign from 'Hello World' to 'Mindspeed MM'"
image = pipe(
    image=image,
    prompt=prompt,
    num_inference_steps=STEPS,
    height=RESOLUTION,
    width=RESOLUTION,
    guidance_scale=GUIDANCE,
    generator=GENERATOR,
).images[0]
image.save(OUTPUT_PATH + "/" + OUTPUT_IMAGE_2)
