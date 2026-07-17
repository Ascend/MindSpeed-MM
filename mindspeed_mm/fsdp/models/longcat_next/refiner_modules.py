# -*- coding: utf-8 -*-
# Copyright (c) 2026 Meituan
# This code is licensed under the MIT License, for details, see the ./LICENSE file.

import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from flash_attn import flash_attn_varlen_func  # type: ignore
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from einops import rearrange, repeat

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


def swiglu(x, y):
    return F.silu(x.float(), inplace=False).to(x.dtype) * y


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.linear_1.weight, std=0.02)
        nn.init.zeros_(self.linear_1.bias)
        nn.init.normal_(self.linear_2.weight, std=0.02)
        nn.init.zeros_(self.linear_2.bias)

    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


@dataclass
class TeaCacheParams:
    """
    TeaCache parameters for Transformer2DModel.
    See https://github.com/ali-vilab/TeaCache/ for a more comprehensive understanding.
    """
    previous_residual: Optional[torch.Tensor] = None
    previous_modulated_inp: Optional[torch.Tensor] = None
    accumulated_rel_l1_distance: float = 0
    is_first_or_last_step: bool = False


def derivative_approximation(*args, **kwargs):
    pass


def taylor_formula(*args, **kwargs):
    pass


def taylor_cache_init(*args, **kwargs):
    pass


def cache_init(*args, **kwargs):
    pass


def cal_type(*args, **kwargs):
    pass


class LuminaRMSNormZero(nn.Module):
    """
    Norm layer adaptive RMS normalization zero.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])
        return x, gate_msa, scale_mlp, gate_mlp


class LuminaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear_1 = nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = nn.Linear(embedding_dim, out_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]
        if self.linear_2 is not None:
            x = self.linear_2(x)
        return x


class LuminaFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        inner_dim: int,
        multiple_of: Optional[int] = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()

        if ffn_dim_multiplier is not None:
            inner_dim = int(ffn_dim_multiplier * inner_dim)
        inner_dim = multiple_of * ((inner_dim + multiple_of - 1) // multiple_of)

        self.linear_1 = nn.Linear(dim, inner_dim, bias=False)
        self.linear_2 = nn.Linear(inner_dim, dim, bias=False)
        self.linear_3 = nn.Linear(dim, inner_dim, bias=False)

    def forward(self, x):
        h1, h2 = self.linear_1(x), self.linear_3(x)
        return self.linear_2(swiglu(h1, h2))


class Lumina2CombinedTimestepCaptionEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        text_feat_dim: int = 2048,
        frequency_embedding_size: int = 256,
        norm_eps: float = 1e-5,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.time_proj = Timesteps(
            num_channels=frequency_embedding_size,
            flip_sin_to_cos=True,
            downscale_freq_shift=0.0,
            scale=timestep_scale,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=frequency_embedding_size,
            time_embed_dim=min(hidden_size, 1024),
        )
        self.caption_embedder = nn.Sequential(
            RMSNorm(text_feat_dim, eps=norm_eps),
            nn.Linear(text_feat_dim, hidden_size, bias=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.caption_embedder[1].weight, std=0.02)
        nn.init.zeros_(self.caption_embedder[1].bias)

    def forward(
        self, timestep: torch.Tensor, text_hidden_states: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        timestep_proj = self.time_proj(timestep).to(dtype=dtype)
        time_embed = self.timestep_embedder(timestep_proj)
        caption_embed = self.caption_embedder(text_hidden_states)
        return time_embed, caption_embed


class AttnProcessorFlash2Varlen:
    """
    Processor for implementing scaled dot-product attention with flash attention
    and variable length sequences.
    """

    def __init__(self) -> None:
        pass
    #     if not is_flash_attn_available():
    #         raise ImportError(
    #             "AttnProcessorFlash2Varlen requires flash_attn. "
    #             "Please install flash_attn."
    #         )

    def _upad_input(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
        query_length: int,
        num_heads: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[int, int]]:
        def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
            seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
            indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
            max_seqlen_in_batch = seqlens_in_batch.max().item()
            cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
            return indices, cu_seqlens, max_seqlen_in_batch

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k,
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer, key_layer, value_layer, indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        (
            query_states, key_states, value_states, indices_q,
            cu_seq_lens, max_seq_lens,
        ) = self._upad_input(query, key, value, attention_mask, sequence_length, attn.heads)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        if kv_heads < attn.heads:
            key_states = repeat(key_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)
            value_states = repeat(value_states, "l h c -> l (h k) c", k=attn.heads // kv_heads)

        attn_output_unpad = flash_attn_varlen_func(
            query_states, key_states, value_states,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q, max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=0.0, causal=False, softmax_scale=softmax_scale,
        )

        hidden_states = pad_input(attn_output_unpad, indices_q, batch_size, sequence_length)
        hidden_states = hidden_states.flatten(-2)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttnProcessor:
    """
    Processor for implementing scaled dot-product attention (PyTorch 2.0+).
    """

    def __init__(self) -> None:
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor requires PyTorch 2.0. "
                "Please upgrade PyTorch to version 2.0 or later."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype
        kv_heads = inner_dim // head_dim

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, use_real=False)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        if base_sequence_length is not None:
            softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
        else:
            softmax_scale = attn.scale

        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states



class RotaryPosEmbed(nn.Module):
    def __init__(
        self,
        theta: int,
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        patch_size: int = 2,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.axes_lens = axes_lens
        self.patch_size = patch_size

    @staticmethod
    def get_freqs_cis(
        axes_dim: Tuple[int, int, int],
        axes_lens: Tuple[int, int, int],
        theta: int,
    ) -> List[torch.Tensor]:
        freqs_cis = []
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        for i, (d, e) in enumerate(zip(axes_dim, axes_lens)):
            emb = get_1d_rotary_pos_embed(d, e, theta=theta, freqs_dtype=freqs_dtype)
            freqs_cis.append(emb)
        return freqs_cis

    def _get_freqs_cis(self, freqs_cis, ids: torch.Tensor) -> torch.Tensor:
        device = ids.device
        if ids.device.type == "mps":
            ids = ids.to("cpu")

        result = []
        for i in range(len(self.axes_dim)):
            freqs = freqs_cis[i].to(ids.device)
            index = ids[:, :, i : i + 1].repeat(1, 1, freqs.shape[-1]).to(torch.int64)
            result.append(
                torch.gather(freqs.unsqueeze(0).repeat(index.shape[0], 1, 1), dim=1, index=index)
            )
        return torch.cat(result, dim=-1).to(device)

    def forward(
        self,
        freqs_cis,
        attention_mask,
        l_effective_ref_img_len,
        l_effective_img_len,
        ref_img_sizes,
        img_sizes,
        device,
    ):
        batch_size = len(attention_mask)
        p = self.patch_size

        encoder_seq_len = attention_mask.shape[1]
        l_effective_cap_len = attention_mask.sum(dim=1).tolist()

        seq_lengths = [
            cap_len + sum(ref_img_len) + img_len
            for cap_len, ref_img_len, img_len in zip(
                l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len
            )
        ]

        max_seq_len = max(seq_lengths)
        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        position_ids = torch.zeros(batch_size, max_seq_len, 3, dtype=torch.int32, device=device)

        for i, (cap_seq_len, seq_len) in enumerate(zip(l_effective_cap_len, seq_lengths)):
            position_ids[i, :cap_seq_len] = repeat(
                torch.arange(cap_seq_len, dtype=torch.int32, device=device), "l -> l 3"
            )

            pe_shift = cap_seq_len
            pe_shift_len = cap_seq_len

            if ref_img_sizes[i] is not None:
                for ref_img_size, ref_img_len in zip(ref_img_sizes[i], l_effective_ref_img_len[i]):
                    H, W = ref_img_size
                    ref_H_tokens, ref_W_tokens = H // p, W // p
                    if ref_H_tokens * ref_W_tokens != ref_img_len:
                        raise AssertionError("ref_H_tokens * ref_W_tokens should equal ref_img_len")

                    row_ids = repeat(
                        torch.arange(ref_H_tokens, dtype=torch.int32, device=device),
                        "h -> h w", w=ref_W_tokens,
                    ).flatten()
                    col_ids = repeat(
                        torch.arange(ref_W_tokens, dtype=torch.int32, device=device),
                        "w -> h w", h=ref_H_tokens,
                    ).flatten()
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 0] = pe_shift
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 1] = row_ids
                    position_ids[i, pe_shift_len:pe_shift_len + ref_img_len, 2] = col_ids

                    pe_shift += max(ref_H_tokens, ref_W_tokens)
                    pe_shift_len += ref_img_len

            H, W = img_sizes[i]
            H_tokens, W_tokens = H // p, W // p
            if H_tokens * W_tokens != l_effective_img_len[i]:
                raise AssertionError("H_tokens * W_tokens should equal l_effective_img_len[i]")

            row_ids = repeat(
                torch.arange(H_tokens, dtype=torch.int32, device=device), "h -> h w", w=W_tokens
            ).flatten()
            col_ids = repeat(
                torch.arange(W_tokens, dtype=torch.int32, device=device), "w -> h w", h=H_tokens
            ).flatten()

            if pe_shift_len + l_effective_img_len[i] != seq_len:
                raise AssertionError("pe_shift_len + l_effective_img_len[i] should equal seq_len")

            position_ids[i, pe_shift_len: seq_len, 0] = pe_shift
            position_ids[i, pe_shift_len: seq_len, 1] = row_ids
            position_ids[i, pe_shift_len: seq_len, 2] = col_ids

        freqs_cis = self._get_freqs_cis(freqs_cis, position_ids)

        cap_freqs_cis = torch.zeros(
            batch_size, encoder_seq_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        ref_img_freqs_cis = torch.zeros(
            batch_size, max_ref_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )
        img_freqs_cis = torch.zeros(
            batch_size, max_img_len, freqs_cis.shape[-1], device=device, dtype=freqs_cis.dtype
        )

        for i, (cap_seq_len, ref_img_len, img_len, seq_len) in enumerate(
            zip(l_effective_cap_len, l_effective_ref_img_len, l_effective_img_len, seq_lengths)
        ):
            cap_freqs_cis[i, :cap_seq_len] = freqs_cis[i, :cap_seq_len]
            ref_img_freqs_cis[i, :sum(ref_img_len)] = freqs_cis[
                i, cap_seq_len:cap_seq_len + sum(ref_img_len)
            ]
            img_freqs_cis[i, :img_len] = freqs_cis[
                i,
                cap_seq_len + sum(ref_img_len):cap_seq_len + sum(ref_img_len) + img_len,
            ]

        return (
            cap_freqs_cis,
            ref_img_freqs_cis,
            img_freqs_cis,
            freqs_cis,
            l_effective_cap_len,
            seq_lengths,
        )


class TransformerBlock(nn.Module):
    """
    Transformer block for refiner model.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        multiple_of: int,
        ffn_dim_multiplier: float,
        norm_eps: float,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.head_dim = dim // num_attention_heads
        self.modulation = modulation

        try:
            processor = AttnProcessorFlash2Varlen()
        except ImportError:
            processor = AttnProcessor()

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            qk_norm="rms_norm",
            heads=num_attention_heads,
            kv_heads=num_kv_heads,
            eps=1e-5,
            bias=False,
            out_bias=False,
            processor=processor,
        )

        self.feed_forward = LuminaFeedForward(
            dim=dim,
            inner_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        if modulation:
            self.norm1 = LuminaRMSNormZero(
                embedding_dim=dim,
                norm_eps=norm_eps,
                norm_elementwise_affine=True,
            )
        else:
            self.norm1 = RMSNorm(dim, eps=norm_eps)

        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.initialize_weights()

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.attn.to_q.weight)
        nn.init.xavier_uniform_(self.attn.to_k.weight)
        nn.init.xavier_uniform_(self.attn.to_v.weight)
        nn.init.xavier_uniform_(self.attn.to_out[0].weight)

        nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
        nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)

        if self.modulation:
            nn.init.zeros_(self.norm1.linear.weight)
            nn.init.zeros_(self.norm1.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")

                if self.current['type'] == 'full':
                    self.current['module'] = 'total'
                    taylor_cache_init(cache_dic=self.cache_dic, current=self.current)

                    norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
                    attn_output = self.attn(
                        hidden_states=norm_hidden_states,
                        encoder_hidden_states=norm_hidden_states,
                        attention_mask=attention_mask,
                        image_rotary_emb=image_rotary_emb,
                    )
                    hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
                    mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
                    hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)

                    derivative_approximation(cache_dic=self.cache_dic, current=self.current, feature=hidden_states)

                elif self.current['type'] == 'Taylor':
                    self.current['module'] = 'total'
                    hidden_states = taylor_formula(cache_dic=self.cache_dic, current=self.current)
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)
        else:
            if self.modulation:
                if temb is None:
                    raise ValueError("temb must be provided when modulation is enabled")

                norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + gate_msa.unsqueeze(1).tanh() * self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states) * (1 + scale_mlp.unsqueeze(1)))
                hidden_states = hidden_states + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(mlp_output)
            else:
                norm_hidden_states = self.norm1(hidden_states)
                attn_output = self.attn(
                    hidden_states=norm_hidden_states,
                    encoder_hidden_states=norm_hidden_states,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + self.norm2(attn_output)
                mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
                hidden_states = hidden_states + self.ffn_norm2(mlp_output)

        return hidden_states


class Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Transformer 2D Model.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]
    _skip_layerwise_casting_patterns = ["x_embedder", "norm"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 16,
        out_channels: Optional[int] = None,
        hidden_size: int = 2304,
        num_layers: int = 26,
        num_refiner_layers: int = 2,
        num_attention_heads: int = 24,
        num_kv_heads: int = 8,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        axes_dim_rope: Tuple[int, int, int] = (32, 32, 32),
        axes_lens: Tuple[int, int, int] = (300, 512, 512),
        text_feat_dim: int = 1024,
        timestep_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
            raise ValueError(
                f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})"
            )

        self.out_channels = out_channels or in_channels

        self.rope_embedder = RotaryPosEmbed(
            theta=10000,
            axes_dim=axes_dim_rope,
            axes_lens=axes_lens,
            patch_size=patch_size,
        )

        self.x_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.ref_image_patch_embedder = nn.Linear(
            in_features=patch_size * patch_size * in_channels,
            out_features=hidden_size,
        )

        self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
            hidden_size=hidden_size,
            text_feat_dim=text_feat_dim,
            norm_eps=norm_eps,
            timestep_scale=timestep_scale,
        )

        self.noise_refiner = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True,
            )
            for _ in range(num_refiner_layers)
        ])

        self.ref_image_refiner = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True,
            )
            for _ in range(num_refiner_layers)
        ])

        self.context_refiner = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=False,
            )
            for _ in range(num_refiner_layers)
        ])

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size, num_attention_heads, num_kv_heads,
                multiple_of, ffn_dim_multiplier, norm_eps, modulation=True,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = LuminaLayerNormContinuous(
            embedding_dim=hidden_size,
            conditioning_embedding_dim=min(hidden_size, 1024),
            elementwise_affine=False,
            eps=1e-6,
            bias=True,
            out_dim=patch_size * patch_size * self.out_channels,
        )

        self.image_index_embedding = nn.Parameter(torch.randn(5, hidden_size))

        self.gradient_checkpointing = False

        self.initialize_weights()

        self.enable_teacache = False
        self.teacache_rel_l1_thresh = 0.05
        self.teacache_params = TeaCacheParams()

        coefficients = [-5.48259225, 11.48772289, -4.47407401, 2.47730926, -0.03316487]
        self.rescale_func = np.poly1d(coefficients)

    def initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.x_embedder.weight)
        nn.init.constant_(self.x_embedder.bias, 0.0)

        nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
        nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

        nn.init.zeros_(self.norm_out.linear_1.weight)
        nn.init.zeros_(self.norm_out.linear_1.bias)
        nn.init.zeros_(self.norm_out.linear_2.weight)
        nn.init.zeros_(self.norm_out.linear_2.bias)

        nn.init.normal_(self.image_index_embedding, std=0.02)

    def img_patch_embed_and_refine(
        self,
        hidden_states,
        ref_image_hidden_states,
        padded_img_mask,
        padded_ref_img_mask,
        noise_rotary_emb,
        ref_img_rotary_emb,
        l_effective_ref_img_len,
        l_effective_img_len,
        temb,
    ):
        batch_size = len(hidden_states)
        max_combined_img_len = max([
            img_len + sum(ref_img_len)
            for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)
        ])

        hidden_states = self.x_embedder(hidden_states)
        ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

        for i in range(batch_size):
            shift = 0
            for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
                ref_image_hidden_states[i, shift:shift + ref_img_len, :] = (
                    ref_image_hidden_states[i, shift:shift + ref_img_len, :]
                    + self.image_index_embedding[j]
                )
                shift += ref_img_len

        for layer in self.noise_refiner:
            hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

        flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
        num_ref_images = len(flat_l_effective_ref_img_len)
        max_ref_img_len = max(flat_l_effective_ref_img_len)

        batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images, max_ref_img_len, dtype=torch.bool)
        batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, self.config.hidden_size
        )
        batch_ref_img_rotary_emb = hidden_states.new_zeros(
            num_ref_images, max_ref_img_len, ref_img_rotary_emb.shape[-1], dtype=ref_img_rotary_emb.dtype
        )
        batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                batch_ref_img_mask[idx, :ref_img_len] = True
                batch_ref_image_hidden_states[idx, :ref_img_len] = ref_image_hidden_states[i, shift:shift + ref_img_len]
                batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift:shift + ref_img_len]
                batch_temb[idx] = temb[i]
                shift += ref_img_len
                idx += 1

        for layer in self.ref_image_refiner:
            batch_ref_image_hidden_states = layer(
                batch_ref_image_hidden_states, batch_ref_img_mask, batch_ref_img_rotary_emb, batch_temb
            )

        idx = 0
        for i in range(batch_size):
            shift = 0
            for ref_img_len in l_effective_ref_img_len[i]:
                ref_image_hidden_states[i, shift:shift + ref_img_len] = batch_ref_image_hidden_states[idx, :ref_img_len]
                shift += ref_img_len
                idx += 1

        combined_img_hidden_states = hidden_states.new_zeros(
            batch_size, max_combined_img_len, self.config.hidden_size
        )
        for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
            combined_img_hidden_states[i, :sum(ref_img_len)] = ref_image_hidden_states[i, :sum(ref_img_len)]
            combined_img_hidden_states[i, sum(ref_img_len):sum(ref_img_len) + img_len] = hidden_states[i, :img_len]

        return combined_img_hidden_states

    def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
        batch_size = len(hidden_states)
        p = self.config.patch_size
        device = hidden_states[0].device

        img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
        l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

        if ref_image_hidden_states is not None and len(ref_image_hidden_states) > 0:
            ref_img_sizes = [
                [(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                for imgs in ref_image_hidden_states
            ]
            l_effective_ref_img_len = [
                [(ref_img_size[0] // p) * (ref_img_size[1] // p) for ref_img_size in _ref_img_sizes]
                if _ref_img_sizes is not None else [0]
                for _ref_img_sizes in ref_img_sizes
            ]
        else:
            ref_img_sizes = [None for _ in range(batch_size)]
            l_effective_ref_img_len = [[0] for _ in range(batch_size)]

        max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
        max_img_len = max(l_effective_img_len)

        flat_ref_img_hidden_states = []
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                imgs = []
                for ref_img in ref_image_hidden_states[i]:
                    C, H, W = ref_img.size()
                    ref_img = rearrange(ref_img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=p, p2=p)
                    imgs.append(ref_img)
                flat_ref_img_hidden_states.append(torch.cat(imgs, dim=0))
            else:
                flat_ref_img_hidden_states.append(None)

        flat_hidden_states = []
        for i in range(batch_size):
            img = hidden_states[i]
            C, H, W = img.size()
            img = rearrange(img, 'c (h p1) (w p2) -> (h w) (p1 p2 c)', p1=p, p2=p)
            flat_hidden_states.append(img)

        padded_ref_img_hidden_states = torch.zeros(
            batch_size, max_ref_img_len, flat_hidden_states[0].shape[-1],
            device=device, dtype=flat_hidden_states[0].dtype,
        )
        padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            if ref_img_sizes[i] is not None:
                padded_ref_img_hidden_states[i, :sum(l_effective_ref_img_len[i])] = flat_ref_img_hidden_states[i]
                padded_ref_img_mask[i, :sum(l_effective_ref_img_len[i])] = True

        padded_hidden_states = torch.zeros(
            batch_size, max_img_len, flat_hidden_states[0].shape[-1],
            device=device, dtype=flat_hidden_states[0].dtype,
        )
        padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
        for i in range(batch_size):
            padded_hidden_states[i, :l_effective_img_len[i]] = flat_hidden_states[i]
            padded_img_mask[i, :l_effective_img_len[i]] = True

        return (
            padded_hidden_states,
            padded_ref_img_hidden_states,
            padded_img_mask,
            padded_ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        )

    def forward(
        self,
        hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        timestep: torch.Tensor,
        text_hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        text_attention_mask: torch.Tensor,
        ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        enable_taylorseer = getattr(self, 'enable_taylorseer', False)
        if enable_taylorseer:
            cal_type(self.cache_dic, self.current)

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        batch_size = len(hidden_states)
        is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

        if is_hidden_states_tensor:
            if hidden_states.ndim != 4:
                raise AssertionError("hidden_states.ndim should equal 4")
            hidden_states = [_hidden_states for _hidden_states in hidden_states]

        device = hidden_states[0].device

        if not isinstance(text_hidden_states, torch.Tensor):
            raise AssertionError(
                f"text_hidden_states must be Tensor, got {type(text_hidden_states)}. "
                f"Check if freqs_cis and text_hidden_states are swapped in the caller."
            )

        temb, text_hidden_states = self.time_caption_embed(timestep, text_hidden_states, hidden_states[0].dtype)

        (
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
        ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

        (
            context_rotary_emb,
            ref_img_rotary_emb,
            noise_rotary_emb,
            rotary_emb,
            encoder_seq_lengths,
            seq_lengths,
        ) = self.rope_embedder(
            freqs_cis,
            text_attention_mask,
            l_effective_ref_img_len,
            l_effective_img_len,
            ref_img_sizes,
            img_sizes,
            device,
        )

        # 2. Context refinement
        for layer in self.context_refiner:
            text_hidden_states = layer(text_hidden_states, text_attention_mask, context_rotary_emb)

        combined_img_hidden_states = self.img_patch_embed_and_refine(
            hidden_states,
            ref_image_hidden_states,
            img_mask,
            ref_img_mask,
            noise_rotary_emb,
            ref_img_rotary_emb,
            l_effective_ref_img_len,
            l_effective_img_len,
            temb,
        )

        # 3. Joint Transformer blocks
        max_seq_len = max(seq_lengths)

        attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
        joint_hidden_states = hidden_states.new_zeros(batch_size, max_seq_len, self.config.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
            attention_mask[i, :seq_len] = True
            joint_hidden_states[i, :encoder_seq_len] = text_hidden_states[i, :encoder_seq_len]
            joint_hidden_states[i, encoder_seq_len:seq_len] = combined_img_hidden_states[i, :seq_len - encoder_seq_len]

        hidden_states = joint_hidden_states

        if self.enable_teacache:
            teacache_hidden_states = hidden_states.clone()
            teacache_temb = temb.clone()
            modulated_inp, _, _, _ = self.layers[0].norm1(teacache_hidden_states, teacache_temb)
            if self.teacache_params.is_first_or_last_step:
                should_calc = True
                self.teacache_params.accumulated_rel_l1_distance = 0
            else:
                self.teacache_params.accumulated_rel_l1_distance += self.rescale_func(
                    ((modulated_inp - self.teacache_params.previous_modulated_inp).abs().mean()
                     / self.teacache_params.previous_modulated_inp.abs().mean()).cpu().item()
                )
                if self.teacache_params.accumulated_rel_l1_distance < self.teacache_rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.teacache_params.accumulated_rel_l1_distance = 0
            self.teacache_params.previous_modulated_inp = modulated_inp

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.teacache_params.previous_residual
            else:
                ori_hidden_states = hidden_states.clone()
                for layer_idx, layer in enumerate(self.layers):
                    if torch.is_grad_enabled() and self.gradient_checkpointing:
                        hidden_states = self._gradient_checkpointing_func(
                            layer, hidden_states, attention_mask, rotary_emb, temb
                        )
                    else:
                        hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)
                self.teacache_params.previous_residual = hidden_states - ori_hidden_states
        else:
            if enable_taylorseer:
                self.current['stream'] = 'layers_stream'

            for layer_idx, layer in enumerate(self.layers):
                if enable_taylorseer:
                    layer.current = self.current
                    layer.cache_dic = self.cache_dic
                    layer.enable_taylorseer = True
                    self.current['layer'] = layer_idx

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states = self._gradient_checkpointing_func(
                        layer, hidden_states, attention_mask, rotary_emb, temb
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask, rotary_emb, temb)

        hidden_states = self.norm_out(hidden_states, temb)

        p = self.config.patch_size
        output = []
        for i, (img_size, img_len, seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
            height, width = img_size
            output.append(rearrange(
                hidden_states[i][seq_len - img_len:seq_len],
                '(h w) (p1 p2 c) -> c (h p1) (w p2)',
                h=height // p, w=width // p, p1=p, p2=p,
            ))
        if is_hidden_states_tensor:
            output = torch.stack(output, dim=0)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if enable_taylorseer:
            self.current['step'] += 1

        if not return_dict:
            return output
        return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# FlowMatch Euler Discrete Scheduler (merged from scheduling_flow_match_euler_discrete.py)
# ---------------------------------------------------------------------------

@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    _compatibles = []
    order = 1

    @register_to_config
    def __init__(self, num_train_timesteps: int = 1000, dynamic_time_shift: bool = False):
        timesteps = torch.linspace(0, 1, num_train_timesteps + 1, dtype=torch.float32)[:-1]
        self.timesteps = timesteps
        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_begin_index(self, begin_index: int = 0):
        self._begin_index = begin_index

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self._timesteps
        indices = (schedule_timesteps == timestep).nonzero()
        pos = 1 if len(indices) > 1 else 0
        return indices[pos].item()

    def set_timesteps(self, num_inference_steps=None, device=None, timesteps=None, num_tokens=None):
        if timesteps is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(0, 1, num_inference_steps + 1, dtype=np.float32)[:-1]
            if self.config.dynamic_time_shift and num_tokens is not None:
                m = np.sqrt(num_tokens) / 40
                timesteps = timesteps / (m - m * timesteps + timesteps)
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=device)
        _timesteps = torch.cat([timesteps, torch.ones(1, device=timesteps.device)])
        self.timesteps = timesteps
        self._timesteps = _timesteps
        self._step_index = None
        self._begin_index = None

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        if isinstance(timestep, (int, torch.IntTensor, torch.LongTensor)):
            raise ValueError("Pass scheduler.timesteps values, not integer indices.")
        if self.step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)
        t = self._timesteps[self.step_index]
        t_next = self._timesteps[self.step_index + 1]
        prev_sample = sample + (t_next - t) * model_output
        prev_sample = prev_sample.to(model_output.dtype)
        self._step_index += 1
        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps
