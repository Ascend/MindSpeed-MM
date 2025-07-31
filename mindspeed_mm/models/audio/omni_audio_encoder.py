# Copyright 2025 The Qwen team; Alibaba Group and the HuggingFace Inc. team. All rights reserved.

import numpy as np
import torch
from torch import nn as nn

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb

from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import Qwen2vlVitSelfAttention


class AudioLinear(torch.nn.Linear):
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.matmul(input_, self.weight.T) + self.bias
        else:
            return torch.matmul(input_, self.weight.T)


class SinusoidsPositionEmbedding(nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding needs even channels input")
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp((-log_timescale_increment * torch.arange(channels // 2).to(torch.bfloat16))).float()
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


class QwenOmniAudioSelfAttention(Qwen2vlVitSelfAttention):
    """Omni Audio模块的q_bias/v_bias为True,k_bias为False，Megatron的SelfAttention是一个统一的linear_qkv.bias
    这里为了迁移到Megatron的SelfAttention适配tp，将linear_qkv.bias中的k_bias初始权重置0并在反向更新时将k_bias部分拆出来对应的梯度置0
    """

    def __init__(self, config: TransformerConfig, submodules: SelfAttentionSubmodules, layer_number: int,
                 attn_mask_type=AttnMaskType.padding):
        super().__init__(config, submodules, layer_number, attn_mask_type)

        def freeze_k_bias_grad_hook(grad):
            grad_clone = grad.clone()
            head_size = self.hidden_size_per_attention_head
            num_heads = self.num_attention_heads_per_partition
            # 遍历每个注意力头，冻结其对应的 K 部分
            for i in range(num_heads):
                start = i * QKV_SIZE * head_size + head_size
                end = start + head_size
                grad_clone[start:end, ...] = 0  # 置零梯度
            return grad_clone

        self.linear_qkv.bias.register_hook(freeze_k_bias_grad_hook)

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        inference_params=None,
    ):
        # hidden_states: [sq, b, h]
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
        
        if self.config.context_parallel_size > key.shape[2]:
            key = key.repeat_interleave(
                query.shape[2] // key.shape[2], dim=2
            )
            value = value.repeat_interleave(
                query.shape[2] // value.shape[2], dim=2
            )
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_context,
            query,
            key,
            value,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            sequence_len_offset,
        )

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            query = apply_rotary_pos_emb(
                query, q_pos_emb, config=self.config,
            )
            key = apply_rotary_pos_emb(
                key, k_pos_emb, config=self.config,
            )

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        # =================
        # Output. [sq, b, h]
        # =================
        output, bias = self.linear_proj(core_attn_out)
        return output, bias


QKV_SIZE = 3
