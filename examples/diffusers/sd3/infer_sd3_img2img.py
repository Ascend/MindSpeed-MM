# Copyright 2024 Huawei Technologies Co., Ltd
# Copyright 2024 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
# Copyright 2024 Stability AI and The HuggingFace Team

import torch
from diffusers import StableDiffusion3Img2ImgPipeline
from diffusers.utils import load_image

DTYPE = torch.float16
MODEL_PATH = (
    "stabilityai/stable-diffusion-3.5-large"  # 模型权重路径 或 微调结果路径
)
DEVICE = "npu"

pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    local_files_only=True,
).to(DEVICE)
init_image = load_image("cat.png")
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipe(prompt, image=init_image).images[0]
image.save("cat_wizard.png")
