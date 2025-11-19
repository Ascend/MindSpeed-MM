import gc
import sys
import math
import pickle
from pathlib import Path
from typing import Tuple, Optional, Union, List

import torch
import torch_npu
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from diffsynth.models.qwen_image_dit import (
    qwen_image_flash_attention,
    apply_rotary_emb_qwen,
    QwenEmbedRope,
    QwenDoubleStreamAttention,
    QwenImageTransformerBlock,
    QwenImageDiT,
    QwenImageDiTStateDictConverter
)
from diffsynth.trainers.utils import (
    DiffusionTrainingModule,
    ModelLogger,
    launch_training_task
)
from .sd3_dit import TimestepEmbeddings, RMSNorm
from .flux_dit import AdaLayerNorm


class RMSNorm_npu(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


def patched_qwen_image_flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, attention_mask=None, enable_fp8_attention: bool = False):
    if q.dtype == torch.bfloat16:
        scale = 1.0 / math.sqrt(q.shape[-1])
        x = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            pse=None,
            scale=scale,
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1.0,
            sync=False,
            inner_precise=0,
            atten_mask=attention_mask
        )[0]
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def patched_apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
):
    x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_out = torch.view_as_real(x_rotated * freqs_cis.to(torch.complex64)).flatten(3)
    return x_out.type_as(x)


class PatchedQwenDoubleStreamAttention(QwenDoubleStreamAttention):
    def __init__(
        self,
        dim_a,
        dim_b,
        num_heads,
        head_dim,
    ):
        super().__init__(
            dim_a,
            dim_b,
            num_heads,
            head_dim
        )
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim_a, dim_a)
        self.to_k = nn.Linear(dim_a, dim_a)
        self.to_v = nn.Linear(dim_a, dim_a)
        self.norm_q = RMSNorm_npu(head_dim, eps=1e-6)
        self.norm_k = RMSNorm_npu(head_dim, eps=1e-6)

        self.add_q_proj = nn.Linear(dim_b, dim_b)
        self.add_k_proj = nn.Linear(dim_b, dim_b)
        self.add_v_proj = nn.Linear(dim_b, dim_b)
        self.norm_added_q = RMSNorm_npu(head_dim, eps=1e-6)
        self.norm_added_k = RMSNorm_npu(head_dim, eps=1e-6)

        self.to_out = torch.nn.Sequential(nn.Linear(dim_a, dim_a))
        self.to_add_out = nn.Linear(dim_b, dim_b)

    def forward(
        self,
        image: torch.FloatTensor,
        text: torch.FloatTensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        enable_fp8_attention: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        img_q, img_k, img_v = self.to_q(image), self.to_k(image), self.to_v(image)
        txt_q, txt_k, txt_v = self.add_q_proj(text), self.add_k_proj(text), self.add_v_proj(text)
        seq_txt = txt_q.shape[1]

        img_q = rearrange(img_q, 'b s (h d) -> b h s d', h=self.num_heads)
        img_k = rearrange(img_k, 'b s (h d) -> b h s d', h=self.num_heads)
        img_v = rearrange(img_v, 'b s (h d) -> b h s d', h=self.num_heads)

        txt_q = rearrange(txt_q, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_k = rearrange(txt_k, 'b s (h d) -> b h s d', h=self.num_heads)
        txt_v = rearrange(txt_v, 'b s (h d) -> b h s d', h=self.num_heads)

        img_q, img_k = self.norm_q(img_q), self.norm_k(img_k)
        txt_q, txt_k = self.norm_added_q(txt_q), self.norm_added_k(txt_k)
        
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_q = patched_apply_rotary_emb_qwen(img_q, img_freqs)
            img_k = patched_apply_rotary_emb_qwen(img_k, img_freqs)
            txt_q = patched_apply_rotary_emb_qwen(txt_q, txt_freqs)
            txt_k = patched_apply_rotary_emb_qwen(txt_k, txt_freqs)

        joint_q = torch.cat([txt_q, img_q], dim=2)
        joint_k = torch.cat([txt_k, img_k], dim=2)
        joint_v = torch.cat([txt_v, img_v], dim=2)

        joint_attn_out = patched_qwen_image_flash_attention(joint_q, joint_k, joint_v, num_heads=joint_q.shape[1], attention_mask=attention_mask, enable_fp8_attention=enable_fp8_attention).to(joint_q.dtype)

        txt_attn_output = joint_attn_out[:, :seq_txt, :]
        img_attn_output = joint_attn_out[:, seq_txt:, :]

        img_attn_output = self.to_out(img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


class PatchedQwenImageDiT(QwenImageDiT):
    def __init__(
        self,
        num_layers: int = 60,
    ):
        super().__init__()

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        self.time_text_embed = TimestepEmbeddings(256, 3072, diffusers_compatible_format=True, scale=1000, align_dtype_to_timestep=True)
        self.txt_norm = RMSNorm_npu(3584, eps=1e-6)

        self.img_in = nn.Linear(64, 3072)
        self.txt_in = nn.Linear(3584, 3072)

        self.transformer_blocks = nn.ModuleList(
            [
                PatchedQwenImageTransformerBlock(
                    dim=3072,
                    num_attention_heads=24,
                    attention_head_dim=128,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNorm(3072, single=True)
        self.proj_out = nn.Linear(3072, 64)


    def process_entity_masks(self, latents, prompt_emb, prompt_emb_mask, entity_prompt_emb, entity_prompt_emb_mask, entity_masks, height, width, image, img_shapes):
        # prompt_emb
        all_prompt_emb = entity_prompt_emb + [prompt_emb]
        all_prompt_emb = [self.txt_in(self.txt_norm(local_prompt_emb)) for local_prompt_emb in all_prompt_emb]
        all_prompt_emb = torch.cat(all_prompt_emb, dim=1)

        # image_rotary_emb
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)
        entity_seq_lens = [emb_mask.sum(dim=1).tolist() for emb_mask in entity_prompt_emb_mask]
        entity_rotary_emb = [self.pos_embed(img_shapes, entity_seq_len, device=latents.device)[1] for entity_seq_len in entity_seq_lens]
        txt_rotary_emb = torch.cat(entity_rotary_emb + [image_rotary_emb[1]], dim=0)
        image_rotary_emb = (image_rotary_emb[0], txt_rotary_emb)

        # attention_mask
        repeat_dim = latents.shape[1]
        max_masks = entity_masks.shape[1]
        entity_masks = entity_masks.repeat(1, 1, repeat_dim, 1, 1)
        entity_masks = [entity_masks[:, i, None].squeeze(1) for i in range(max_masks)]
        global_mask = torch.ones_like(entity_masks[0]).to(device=latents.device, dtype=latents.dtype)
        entity_masks = entity_masks + [global_mask]

        N = len(entity_masks)
        batch_size = entity_masks[0].shape[0]
        seq_lens = [mask_.sum(dim=1).item() for mask_ in entity_prompt_emb_mask] + [prompt_emb_mask.sum(dim=1).item()]
        total_seq_len = sum(seq_lens) + image.shape[1]
        patched_masks = []
        for i in range(N):
            patched_mask = rearrange(entity_masks[i], "B C (H P) (W Q) -> B (H W) (C P Q)", H=height // 16, W=width // 16, P=2, Q=2)
            patched_masks.append(patched_mask)
        attention_mask = torch.ones((batch_size, total_seq_len, total_seq_len), dtype=torch.bool).to(device=entity_masks[0].device)

        # prompt-image attention mask
        image_start = sum(seq_lens)
        image_end = total_seq_len
        cumsum = [0]
        single_image_seq = image_end - image_start
        for length in seq_lens:
            cumsum.append(cumsum[-1] + length)
        for i in range(N):
            prompt_start = cumsum[i]
            prompt_end = cumsum[i + 1]
            image_mask = torch.sum(patched_masks[i], dim=-1) > 0
            image_mask = image_mask.unsqueeze(1).repeat(1, seq_lens[i], 1)
            # repeat image mask to match the single image sequence length
            repeat_time = single_image_seq // image_mask.shape[-1]
            image_mask = image_mask.repeat(1, 1, repeat_time)
            # prompt update with image
            attention_mask[:, prompt_start:prompt_end, image_start:image_end] = image_mask
            # image update with prompt
            attention_mask[:, image_start:image_end, prompt_start:prompt_end] = image_mask.transpose(1, 2)
        # prompt-prompt attention mask, let the prompt tokens not attend to each other
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                start_i, end_i = cumsum[i], cumsum[i + 1]
                start_j, end_j = cumsum[j], cumsum[j + 1]
                attention_mask[:, start_i:end_i, start_j:end_j] = False

        attention_mask = attention_mask.float()
        attention_mask[attention_mask == 0] = float('-inf')
        attention_mask[attention_mask == 1] = 0
        attention_mask = attention_mask.to(device=latents.device, dtype=latents.dtype).unsqueeze(1)

        return all_prompt_emb, image_rotary_emb, attention_mask


    def forward(
        self,
        latents=None,
        timestep=None,
        prompt_emb=None,
        prompt_emb_mask=None,
        height=None,
        width=None,
    ):
        img_shapes = [(latents.shape[0], latents.shape[2] // 2, latents.shape[3] // 2)]
        txt_seq_lens = prompt_emb_mask.sum(dim=1).tolist()
        
        image = rearrange(latents, "B C (H P) (W Q) -> B (H W) (C P Q)", H=height // 16, W=width // 16, P=2, Q=2)
        image = self.img_in(image)
        text = self.txt_in(self.txt_norm(prompt_emb))

        conditioning = self.time_text_embed(timestep, image.dtype)

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=latents.device)

        for block in self.transformer_blocks:
            text, image = block(
                image=image,
                text=text,
                temb=conditioning,
                image_rotary_emb=image_rotary_emb,
            )
        
        image = self.norm_out(image, conditioning)
        image = self.proj_out(image)
        
        latents = rearrange(image, "B (H W) (C P Q) -> B C (H P) (W Q)", H=height // 16, W=width // 16, P=2, Q=2)
        return image
    
    @staticmethod
    def state_dict_converter():
        return QwenImageDiTStateDictConverter()


def Patched_launch_training_task(
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 8,
    save_steps: int = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
    find_unused_parameters: bool = False,
    args=None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
    
    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
    )
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    preprocess_cache_path = Path(f"./preprocessed_cache_{torch.distributed.get_rank()}.pkl")

    preprocessed_inputs = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc="preprocess prompts"):
            inputs = model.forward_preprocess(data)
            preprocessed_inputs.append(inputs)

    if hasattr(model, 'module'):
        original_model = model.module
    else:   
        original_model = model
    if hasattr(original_model.pipe, 'text_encoder') and original_model.pipe.text_encoder is not None:
        if accelerator.is_main_process:
            print("开始释放 text_encoder 内存...")

        original_model.pipe.text_encoder = original_model.pipe.text_encoder.to("cpu")
        del original_model.pipe.text_encoder
        original_model.pipe.text_encoder = None
        if hasattr(model.pipe, 'text_encoder'):
            del model.pipe.text_encoder
            model.pipe.text_encoder = None
    
    gc.collect()
    torch.npu.empty_cache()
    if accelerator.is_main_process:
        print("Prompt preprocess complated, release memory allready...")

    model.train()
    batches_per_epoch = len(preprocessed_inputs)
    total_steps = num_epochs * batches_per_epoch
    global_step = 0

    progress_bar = tqdm(
        range(0, total_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch_id in range(num_epochs):
        for inputs in preprocessed_inputs:
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model({}, inputs=inputs)
                accelerator.backward(loss)
                optimizer.step()
                global_step += 1
                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(refresh=False, **logs)
                model_logger.on_step_end(accelerator, model, save_steps)
                scheduler.step()
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)


def apply_patches():
    dit_module = sys.modules["diffsynth.models.qwen_image_dit"]
    dit_module.qwen_image_flash_attention = patched_qwen_image_flash_attention
    dit_module.apply_rotary_emb_qwen = patched_apply_rotary_emb_qwen
    dit_module.QwenDoubleStreamAttention = PatchedQwenDoubleStreamAttention
    dit_module.QwenImageDiT = PatchedQwenImageDiT

    trainer_module = sys.modules["diffsynth.trainer.utils"]
    trainer_module.launch_training_task = Patched_launch_training_task