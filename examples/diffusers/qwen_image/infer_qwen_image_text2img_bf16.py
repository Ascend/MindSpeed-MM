# Copyright 2025 Huawei Technologies Co., Ltd

import os

import torch
from diffusers import QwenImagePipeline

model_name = "Qwen/Qwen-Image" # Qwen Image pretrained model
output_path = "./infer_result"  # Inference result output folder
device = "npu"

os.makedirs(output_path, exist_ok=True)  # Create the output folder

# Generate image
prompt = '''A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition'''
negative_prompt = " " # using an empty string if you do not have specific concept to remove
torch_dtype = torch.bfloat16
num_inference_step = 50
seed = 42

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.", # for english prompt
    "zh": ", 超清，4K，电影级构图." # for chinese prompt
}

pipe = QwenImagePipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe.enable_model_cpu_offload(device=device)

# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=num_inference_step,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(seed)
).images[0]

image.save(os.path.join(output_path, "example.png"))