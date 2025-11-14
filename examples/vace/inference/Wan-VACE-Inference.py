import os
import json
import sys
import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig

# read inference config file
config = "../inference/inference_wan2.1_1.3b.json"
if len(sys.argv) >= 2:
    config = sys.argv[1]
try:
    with open(config, 'r') as f:
        inference_config = json.load(f)
except OSError:
    print(f"FAILED: {config} can't open")

# model path
transformer_paths = inference_config['model']['transformer']
vae_path = inference_config['model']['vae']
text_encoder_path = inference_config['model']['text_encoder']
tokenizer_path = inference_config['model']['tokenizer']

# data path
video_path = inference_config['video_path']
image_path = inference_config['image_path']
output_path = inference_config['output_path']

if not os.path.exists(output_path):
    os.mkdir(output_path)

model_config = []
for model_path in model_config:
    model_config.append(ModelConfig(path=model_path))
model_config.append(ModelConfig(path=vae_path))
model_config.append(ModelConfig(path=text_encoder_path))

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="npu",
    model_configs=model_config,
    tokenizer_config=ModelConfig(path=tokenizer_path)
)

pipe.enable_vram_management()

# Depth video -> Video
control_video = VideoData(video_path, height=480, width=832)
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    seed=1, tiled=True
)
save_video(video, os.path.join(output_path, "video1.mp4"), fps=15, quality=5)

# Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_reference_image=Image.open(image_path).resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, os.path.join(output_path, "video2.mp4"), fps=15, quality=5)

# Depth video + Reference image -> Video
video = pipe(
    prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=Image.open(image_path).resize((832, 480)),
    seed=1, tiled=True
)
save_video(video, os.path.join(output_path, "video3.mp4"), fps=15, quality=5)
