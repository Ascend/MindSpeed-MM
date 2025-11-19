from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
import torch

transformer_path = "Qwen/Qwen-Image-Edit/transformer"
transformer_files = "${transformer_path}/diffusion_pytorch_model*.safetensors"

text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
text_encoder_files = "${text_encoder_path}/model*.safetensors"

vae_file = "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"

tokenizer_file = "Qwen/Qwen-Image/tokenizer"

processor_file = "Qwen/Qwen-Image/processor"

lora_path = "Qwen-Image-LoRA/model.safetensors"

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="npu",
    model_configs=[
        ModelConfig(path=transformer_files),
        ModelConfig(path=text_encoder_files),
        ModelConfig(path=vae_file),
    ],
    tokenizer_config=ModelConfig(
        path=tokenizer_file,
    ),
    processor_config=ModelConfig(
        path=processor_file,
    ),
)
pipe.load_lora(pipe.dit, lora_path)

prompt = "精致肖像，水下少女，蓝裙飘逸，发丝轻扬，光影透澈，气泡环绕，面容恬静，细节精致，梦幻唯美。"
input_image = pipe(prompt=prompt, seed=1234, num_inference_steps=40, height=1024, width=1024)
input_image.save("./inference/image1.jpg")

edit_prompt = "少女改为身穿粉红色的裙子"
# edit_image_auto_resize=True: auto resize input image to match the area of 1024*1024 with the original aspect ratio
image_edit1 = pipe(edit_prompt, edit_image=input_image, seed=1234, num_inference_steps=60, height=1024, width=1024, edit_image_auto_resize=True)
image_edit1.save(f"./inference/image2.jpg")

# edit_image_auto_resize=False: do not resize input image
image_edit2 = pipe(edit_prompt, edit_image=input_image, seed=1234, num_inference_steps=60, height=1024, width=1024, edit_image_auto_resize=False)
image_edit2.save(f"./inference/image3.jpg")