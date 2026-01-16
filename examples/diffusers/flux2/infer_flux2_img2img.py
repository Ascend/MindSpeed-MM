import functools
import os
import time
from functools import partial

import torch
import torch.distributed as dist
from accelerate import PartialState
from diffusers import ContextParallelConfig, Flux2Pipeline
from diffusers.models.transformers.transformer_flux2 import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)
from diffusers.utils import load_image
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformer_patches import apply_patches, patched_prepare_latents
from transformers import AutoConfig, Mistral3ForConditionalGeneration, PixtralProcessor

apply_patches()
# --- Setup Environments ---
OUTPUT_PATH = "./infer_result"  # Output path
MODEL_PATH = "/data/datasets/FLUX.2-dev"  # Model path
LORA_WEIGHTS = "./logs_t2i_5000/pytorch_lora_weights.safetensors"  # Path for saved LoRA
IMAGE = "./infer_result/flux2.fsdp_ulysses4.png"  # Image path that you want to edit
PROMPT = "Change the crab to a dog"  # Prompt for editing task
SEED = 0  # Seed for the generator
STEPS = 20  # Number of steps for inference
GUIDANCE = 1  # Guidance scale for inference
RESOLUTION = 1024  # Resolution for inference

# --- Setup Distributed Environment ---
dist.init_process_group(backend="hccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
distributed_state = PartialState()
device = distributed_state.device

os.makedirs(OUTPUT_PATH, exist_ok=True)  # Create the output folder

img = load_image(IMAGE).convert("RGB")
config = AutoConfig.from_pretrained(MODEL_PATH, subfolder="text_encoder", revision=None)
config.text_config._attn_implementation = "eager"


def compute_text_embeddings(prompt, pipeline):
    with torch.no_grad():
        embeds, ids = pipeline.encode_prompt(prompt=prompt, max_sequence_length=512)
    return embeds, ids


tokenizer = PixtralProcessor.from_pretrained(
    MODEL_PATH,
    subfolder="tokenizer",
    revision=None,
)
text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    subfolder="text_encoder",
    revision=None,
    variant=None,
    config=config,
).to(dtype=torch.bfloat16, device="cpu")
text_encoder.requires_grad_(False)

text_encoding_pipeline = Flux2Pipeline.from_pretrained(
    MODEL_PATH,
    vae=None,
    transformer=None,
    tokenizer=tokenizer,
    text_encoder=text_encoder,
    scheduler=None,
    revision=None,
)

# ---  Wrap Text Ecnoder  ---
transformer_layer_cls = type(text_encoder.model.language_model.layers[0])
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={transformer_layer_cls},
)

text_encoder_fsdp = FSDP(
    text_encoding_pipeline.text_encoder,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    device_id=device,
    use_orig_params=False,
)
text_encoding_pipeline.text_encoder = text_encoder_fsdp
dist.barrier()

prompt_embeds, _ = compute_text_embeddings(PROMPT, text_encoding_pipeline)
text_encoding_pipeline = text_encoding_pipeline.to("cpu")
del text_encoder, tokenizer

pipe: Flux2Pipeline = Flux2Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)

if os.path.exists(LORA_WEIGHTS):  # Load Lora weights
    print(f"Loading LoRA weights from {LORA_WEIGHTS}")
    pipe.load_lora_weights(LORA_WEIGHTS)
else:
    print("LoRA weights not found. Using the base model")

# ---  Wrap Transformer  ---
transformer = pipe.transformer
transformer.set_attention_backend("native")

transformer.requires_grad_(False)

if world_size > 1:
    transformer.enable_parallelism(
        config=ContextParallelConfig(ulysses_degree=world_size)
    )

transformer = FSDP(
    transformer,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Flux2TransformerBlock, Flux2SingleTransformerBlock},
    ),
    device_id=device,
    use_orig_params=False,
)
pipe.transformer = transformer
pipe.vae.to(device)


class NPUPipeline(type(pipe)):
    @property
    def _execution_device(self):
        return device


pipe.__class__ = NPUPipeline

torch.npu.synchronize()
dist.barrier()

pipe.prepare_latents = patched_prepare_latents.__get__(pipe, type(pipe))
pipe.set_progress_bar_config(disable=rank != 0)


# --- Run Inference ---
def run_pipe():
    generator = torch.Generator("cpu").manual_seed(SEED)
    image = pipe(
        prompt_embeds=prompt_embeds,
        image=[img],
        num_inference_steps=STEPS,
        height=RESOLUTION,
        width=RESOLUTION,
        guidance_scale=GUIDANCE,
        generator=generator,
    ).images[0]
    return image


start = time.time()

output_image = run_pipe()

end = time.time()

if rank == 0:
    time_cost = end - start
    save_path = f"{OUTPUT_PATH}/flux2.fsdp_ulysses{world_size}_i2i.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    output_image.save(save_path)

if dist.is_initialized():
    dist.destroy_process_group()
