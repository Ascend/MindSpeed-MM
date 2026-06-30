# Copyright 2025 The Wan-Video Team and The HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import all_to_all
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.utils.utils import get_dtype


# Inline utilities to avoid pulling in heavy external modules at import time
class _LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        try:
            return torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        except TypeError:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)


class _FP32LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True, **kwargs):
        super().__init__()
        self.dim = (dim,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.dim)) if elementwise_affine else None
        self.bias = nn.Parameter(torch.zeros(self.dim)) if elementwise_affine else None

    def forward(self, inputs):
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.dim,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


_WanVideoVAE = None
_WanFlowMatchScheduler = None


def get_wan_video_vae():
    global _WanVideoVAE
    if _WanVideoVAE is None:
        from mindspeed_mm.models.ae.wan_video_vae import WanVideoVAE as _WanVideoVAE
    return _WanVideoVAE


def get_wan_flow_match_scheduler():
    global _WanFlowMatchScheduler
    if _WanFlowMatchScheduler is None:
        from mindspeed_mm.models.diffusion.wan_flow_match_scheduler import WanFlowMatchScheduler as _WanFlowMatchScheduler
    return _WanFlowMatchScheduler


class _SimpleTextEncoder(nn.Module):
    """Loads directly from transformers."""
    TRANSFORMERS_MAPPING = {
        "T5": "T5EncoderModel",
        "MT5": "MT5EncoderModel",
        "UMT5": "UMT5EncoderModel",
    }

    def __init__(
        self,
        model,
        use_attention_mask=True,
        output_key="last_hidden_state",
        hidden_state_skip_layer=None,
        ucg_rate=None,
    ):
        super().__init__()
        self.model = model
        self.use_attention_mask = use_attention_mask
        self.output_key = output_key
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.ucg_rate = ucg_rate

    def encode(self, input_ids, attention_mask, **kwargs):
        device = get_device_type()
        *BN, L = input_ids.shape
        input_ids = input_ids.to(device).view(-1, L)
        attention_mask = attention_mask.to(device).view(-1, L)
        model_attention_mask = attention_mask if self.use_attention_mask else None

        output = self.model(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )

        emb = output[self.output_key]
        if self.hidden_state_skip_layer:
            emb = emb[-(self.hidden_state_skip_layer + 1)]

        if self.ucg_rate is not None and self.ucg_rate > 0.0:
            def expand_dims_like(x, y):
                while x.dim() != y.dim():
                    x = x.unsqueeze(-1)
                return x
            emb = (
                expand_dims_like(
                    torch.bernoulli(
                        (1.0 - self.ucg_rate) * torch.ones(emb.shape[0], device=emb.device, dtype=emb.dtype)
                    ),
                    emb,
                )
                * emb
            )

        if self.output_key in ["last_hidden_state", "hidden_states"]:
            emb = emb.view(*BN, emb.shape[-2], -1)
        elif self.output_key in ["pooler_output", "text_embeds"]:
            emb = emb.view(*BN, -1)
        else:
            raise NotImplementedError(f"Text encoder output_key: {self.output_key} is not implemented!")

        attention_mask = attention_mask.view(*BN, -1)
        return emb, attention_mask


@dataclass
class Wan2_2ModelOutput:
    loss: torch.Tensor


def build_text_encoder(config: dict):
    """Build a _SimpleTextEncoder directly from transformers."""
    import transformers

    config = dict(config)  # shallow copy
    backend = config.pop("hub_backend", "hf")
    model_id = config.pop("model_id", "UMT5")
    use_attention_mask = config.pop("use_attention_mask", True)
    ucg_rate = config.pop("ucg_rate", None)
    output_key = config.pop("output_key", "last_hidden_state")
    hidden_state_skip_layer = config.pop("hidden_state_skip_layer", None)
    using_kwargs = config.pop("using_kwargs", None)

    pretrained_path = config.pop("from_pretrained")
    torch_dtype = get_dtype(config.pop("dtype", "bf16"))

    # Drop keys that transformers does not recognize
    config.pop("load_in_8bit", None)

    if model_id in _SimpleTextEncoder.TRANSFORMERS_MAPPING:
        automodel_name = _SimpleTextEncoder.TRANSFORMERS_MAPPING[model_id]
        automodel = getattr(transformers, automodel_name)
    else:
        raise ValueError(f"Model ID {model_id} is not supported for text encoder in pure FSDP2 mode")

    text_encoder = automodel.from_pretrained(
        pretrained_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        trust_remote_code=False,
    )

    return _SimpleTextEncoder(
        model=text_encoder,
        use_attention_mask=use_attention_mask,
        output_key=output_key,
        hidden_state_skip_layer=hidden_state_skip_layer,
        ucg_rate=ucg_rate,
    )


def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # Keep float64 to match the original repo's precision
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float64),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2, dtype=torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        if x.device.type == 'npu':
            freqs_parts = [
                torch.view_as_real(
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1)).float(),
                torch.view_as_real(
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)).float(),
                torch.view_as_real(
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)).float(),
            ]
            freqs_i = torch.view_as_complex(
                torch.cat(freqs_parts, dim=-2)).reshape(seq_len, 1, -1)
        else:
            freqs_i = torch.cat([
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x)


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    attn_implementation="eager",
    **kwargs,
):
    """
    Compatible flash_attention for both CUDA and NPU.

    Args:
        attn_implementation: "eager" for SDPA fallback, "flash_attention_2" for
            native flash attention (npu_fusion_attention on NPU, flash_attn on CUDA).
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    out_dtype = q.dtype
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # NPU path
    if q.device.type == 'npu' and attn_implementation == "flash_attention_2":
        import torch_npu
        q_npu = half(q).transpose(1, 2).contiguous()
        k_npu = half(k).transpose(1, 2).contiguous()
        v_npu = half(v).transpose(1, 2).contiguous()
        n = q.size(2)
        if q_scale is not None:
            q_npu = q_npu * q_scale
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.size(-1))
        out = torch_npu.npu_fusion_attention(
            q_npu, k_npu, v_npu, n, "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=None,
            keep_prob=1.0 - dropout_p,
            scale=softmax_scale,
        )[0]
        out = out.transpose(1, 2).contiguous()
        out = out.type(out_dtype)
        return out

    # Fallback for CPU / NPU eager / when flash_attn is unavailable
    if q.device.type != 'cuda' or q.size(-1) > 256 or attn_implementation == "eager":
        if q_lens is not None or k_lens is not None:
            # For training with equal-length batches, standard sdpa works.
            # Warn if lengths actually vary.
            if q_lens is not None and not torch.all(q_lens == q_lens[0]):
                import warnings
                warnings.warn("Variable-length attention with sdpa fallback may be incorrect. "
                            "Consider ensuring equal sequence lengths per batch.")
        q = half(q).transpose(1, 2)
        k = half(k).transpose(1, 2)
        v = half(v).transpose(1, 2)
        if q_scale is not None:
            q = q * q_scale
        attn_mask = None
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=causal)
        out = out.transpose(1, 2).contiguous()
        out = out.type(out_dtype)
        return out

    # CUDA flash attention path
    try:
        import flash_attn_interface
        FLASH_ATTN_3_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_3_AVAILABLE = False

    try:
        import flash_attn
        FLASH_ATTN_2_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_2_AVAILABLE = False

    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        import warnings
        warnings.warn('Flash attention 3 is not available, use flash attention 2 instead.')

    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True),
            seqused_q=None, seqused_k=None,
            max_seqlen_q=lq, max_seqlen_k=lk,
            softmax_scale=softmax_scale, causal=causal, deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device, non_blocking=True),
            max_seqlen_q=lq, max_seqlen_k=lk,
            dropout_p=dropout_p, softmax_scale=softmax_scale,
            causal=causal, window_size=window_size, deterministic=deterministic).unflatten(0, (b, lq))

    return x.type(out_dtype)


class WanSelfAttention(nn.Module):
    def __init__(self,
                dim,
                num_heads,
                window_size=(-1, -1),
                qk_norm=True,
                eps=1e-6,
                layer_idx=0,
                num_layers=1,
                attn_implementation="eager"):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_implementation = attn_implementation

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        # Apply RoPE before CP split since rope_apply uses full grid_sizes
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        ps = get_parallel_state()
        # Ulysses CP: split seq before all_to_all so gather yields correct global seq
        if ps.is_ulysses_enable():
            ulysses_size = ps.get_ulysses_group_size()
            if n % ulysses_size != 0:
                raise ValueError(f"num_heads ({n}) must be divisible by ulysses_size ({ulysses_size})")
            from mindspeed_mm.fsdp.distributed.context_parallel.communication import split_forward_gather_backward
            from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes
            split_sizes = cal_split_sizes(s, ulysses_size)
            q = split_forward_gather_backward(q, ps.get_ulysses_group(), dim=1, split_sizes=split_sizes)
            k = split_forward_gather_backward(k, ps.get_ulysses_group(), dim=1, split_sizes=split_sizes)
            v = split_forward_gather_backward(v, ps.get_ulysses_group(), dim=1, split_sizes=split_sizes)
            q = all_to_all(q, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1)
            k = all_to_all(k, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1)
            v = all_to_all(v, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1)

        # Ring CP: split seq for ring attention (requires seq_len % (2 * ring_size) == 0)
        if ps.is_ring_enable():
            ring_size = ps.get_ring_group_size()
            if s % (2 * ring_size) != 0:
                raise ValueError(
                    f"Ring Attention CP requires seq_len ({s}) to be divisible by 2 * ring_size "
                    f"({2 * ring_size}). Consider adjusting video resolution or padding tokens."
                )
            from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
                load_balanced_split_forward_gather_backward
            )
            q = load_balanced_split_forward_gather_backward(q, ps.get_ring_group(), dim=1)
            k = load_balanced_split_forward_gather_backward(k, ps.get_ring_group(), dim=1)
            v = load_balanced_split_forward_gather_backward(v, ps.get_ring_group(), dim=1)

        # Ring CP path: direct ring attention (q/k/v already split in seq dim)
        if ps.is_ring_enable():
            from einops import rearrange
            from mindspeed_mm.fsdp.distributed.context_parallel.ring_context_parallel.ring_context_parallel import (
                ringattn_context_parallel, AttentionWithCp
            )
            target_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
            q = q.to(target_dtype)
            k = k.to(target_dtype)
            v = v.to(target_dtype)
            b_, s_, n_, d_ = q.shape
            q_sbh = rearrange(q, 'b s n d -> s b (n d)')
            k_sbh = rearrange(k, 'b s n d -> s b (n d)')
            v_sbh = rearrange(v, 'b s n d -> s b (n d)')
            cp_para = dict(
                causal=False,
                cp_group=ps.get_ring_group(),
                cp_size=ps.get_ring_group_size(),
                rank=ps.get_ring_rank(),
                cp_global_ranks=ps.get_ring_device_mesh().mesh.tolist(),
                cp_group_for_send_recv_overlap=None,
                megatron_cp_in_bnsd=False,
            )
            AttentionWithCp.block_size = q_sbh.shape[0]
            AttentionWithCp.batch_size = q_sbh.shape[1]
            out_sbh = ringattn_context_parallel(
                q_sbh, k_sbh, v_sbh, n_, cp_para,
                softmax_scale=None, attn_mask=None, dropout_p=0.
            )
            x = rearrange(out_sbh, 's b (n d) -> b s n d', n=n_)
        else:
            _attn_kwargs = dict(
                q=q, k=k, v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                attn_implementation=self.attn_implementation,
                layer_idx=getattr(self, 'layer_idx', None),
            )
            x = flash_attention(**_attn_kwargs)

        if ps.is_ulysses_enable():
            x = all_to_all(x, ps.get_ulysses_group(), scatter_dim=1, gather_dim=2)
            # Gather attention output back to full seq so residual/addnorm operate on full x.
            # Pass gather_sizes to support unaligned sequence lengths across Ulysses ranks.
            from mindspeed_mm.fsdp.distributed.context_parallel.communication import gather_forward_split_backward
            from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes
            _ulysses_gather_sizes = cal_split_sizes(s, ps.get_ulysses_group_size())
            x = gather_forward_split_backward(x, ps.get_ulysses_group(), dim=1, gather_sizes=_ulysses_gather_sizes)

        if ps.is_ring_enable():
            # Gather ring attention output back to full seq.
            # Requires seq_len % ring_size == 0 for aligned gather.
            from mindspeed_mm.fsdp.distributed.context_parallel.communication import gather_forward_split_backward
            x = gather_forward_split_backward(x, ps.get_ring_group(), dim=1)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        x = flash_attention(q, k, v, k_lens=context_lens,
                            attn_implementation=self.attn_implementation,
                            layer_idx=getattr(self, 'layer_idx', None))

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanMoERouter(nn.Linear):
    def __init__(self, dim, num_experts, top_k=1):
        super().__init__(dim, num_experts, bias=False)
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.dim)
        router_logits = super().forward(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        return router_weights, router_logits, router_indices


class WanMoEExperts(nn.Module):
    def __init__(self, dim, ffn_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.hidden_size = dim
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, dim, 2 * ffn_dim))
        self.down_proj = nn.Parameter(torch.empty(num_experts, ffn_dim, dim))
        self.act_fn = nn.GELU(approximate='tanh')

    def forward(self, hidden_states, routing_weights, router_indices):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.dim)
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit[:]:
            expert_id = int(expert_idx)
            with torch.no_grad():
                _, token_idx = torch.where(expert_mask[expert_id])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_id]
            gate, up = gate_up.chunk(2, dim=-1)
            gated_output = up * self.act_fn(gate)
            out = gated_output @ self.down_proj[expert_id]
            weighted_output = out[0] * routing_weights[token_idx, expert_id, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))
        next_states = next_states.view(batch_size, -1, self.dim)
        return next_states

    def ep_forward(self, hidden_states, routing_weights, router_indices, ep_group, ep_plan):
        """Expert Parallel forward for Wan2.2 MoE.

        Called by expert_parallelize_modules when expert_parallel_size > 1.
        Uses all-to-all token dispatch with grouped matmul, compatible with
        the EP infrastructure in mindspeed_mm.fsdp.distributed.expert_parallel.
        """
        from mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher import ep_forward

        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)

        # Handle DTensor weights after EP shard (dim 0)
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj

        def _activation_fn(intermediate):
            gate, up = intermediate.chunk(2, dim=-1)
            return up * torch.nn.functional.gelu(gate, approximate='tanh')

        hidden_states = ep_forward(
            num_experts=self.num_experts,
            routing_weights=routing_weights,
            selected_experts=router_indices,
            hidden_states=hidden_states,
            fc1_weight=gate_up_proj,
            fc2_weight=down_proj,
            ep_group=ep_group,
            fused=getattr(ep_plan, 'use_npu_fused_ops', True),
            activation_fn=_activation_fn,
        )

        hidden_states = hidden_states.view(batch_size, -1, self.hidden_size)
        return hidden_states


class WanSparseMoEBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_experts, top_k=1):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.gate = WanMoERouter(dim, num_experts, top_k)
        self.experts = WanMoEExperts(dim, ffn_dim, num_experts)

    def forward(self, hidden_states):
        router_weights, router_logits, router_indices = self.gate(hidden_states)
        self.last_router_logits = router_logits
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out


class WanAttentionBlock(nn.Module):
    def __init__(self,
                dim,
                ffn_dim,
                num_heads,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                num_experts=1,
                top_k=1,
                layer_idx=0,
                num_layers=1,
                attn_implementation="eager"):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(
            dim, num_heads, window_size, qk_norm, eps,
            layer_idx=layer_idx,
            num_layers=num_layers,
            attn_implementation=attn_implementation,
        )
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(
            dim, num_heads, (-1, -1), qk_norm, eps,
            attn_implementation=attn_implementation)
        self.norm2 = WanLayerNorm(dim, eps)
        if num_experts > 1:
            self.ffn = WanSparseMoEBlock(dim, ffn_dim, num_experts, top_k)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
                nn.Linear(ffn_dim, dim))

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        e = [i.squeeze(2).to(x.dtype) for i in e]

        y = self.self_attn(
            self.norm1(x.to(torch.float32)).to(x.dtype) * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, freqs)
        x = x + y * e[2]

        y = self.cross_attn(self.norm3(x), context, context_lens)
        x = x + y

        y = self.ffn(
            self.norm2(x.to(torch.float32)).to(x.dtype) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        # Align with wan_dit default (fp32_calculate=False).
        e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
        e = [i.squeeze(2).to(x.dtype) for i in e]

        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanModel(nn.Module):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    Adapted from wan/modules/model.py for FSDP2 training.
    """

    def __init__(self,
                model_type='t2v',
                patch_size=(1, 2, 2),
                text_len=512,
                in_dim=16,
                dim=2048,
                ffn_dim=8192,
                freq_dim=256,
                text_dim=4096,
                out_dim=16,
                num_heads=16,
                num_layers=32,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=True,
                eps=1e-6,
                num_experts=1,
                top_k=1,
                **kwargs):
        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_experts = num_experts
        self.top_k = top_k

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                dim, ffn_dim, num_heads, window_size, qk_norm,
                cross_attn_norm, eps, num_experts, top_k,
                layer_idx=i,
                num_layers=num_layers,
                attn_implementation=kwargs.get('attn_implementation', 'eager'),
            )
            for i in range(num_layers)
        ])

        self.head = Head(dim, out_dim, patch_size, eps)

        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self._freqs = None

        # Defer init_weights when parameters are on meta device
        if not any(getattr(p, 'is_meta', False) for p in self.parameters()):
            self.init_weights()

    def forward(self, x, t, context, seq_len, y=None, context_lens=None):
        if self.model_type == 'i2v':
            assert y is not None
        device = self.patch_embedding.weight.device
        if self._freqs is None or self._freqs.device != device:
            assert (self.dim % self.num_heads) == 0 and (self.dim // self.num_heads) % 2 == 0
            d = self.dim // self.num_heads
            self._freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6))
            ], dim=1).to(device)
        freqs = self._freqs

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                    dim=1) for u in x
        ])

        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim))

        if context_lens is None:
            context_lens = torch.tensor([u.size(0) for u in context], dtype=torch.long, device=context[0].device)
        elif not isinstance(context_lens, torch.Tensor):
            context_lens = torch.tensor(context_lens, dtype=torch.long, device=context[0].device)
        context_stacked = torch.stack([
            torch.cat(
                [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ])
        context = self.text_embedding(context_stacked)

        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=freqs,
            context=context,
            context_lens=context_lens)

        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)

        x = self.head(x, e)

        x = self.unpatchify(x, grid_sizes)
        outputs = [u.float() for u in x]
        return outputs

    def unpatchify(self, x, grid_sizes):
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (WanRMSNorm, _LlamaRMSNorm)):
                nn.init.ones_(m.weight)
            elif isinstance(m, (_FP32LayerNorm, WanLayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, WanMoEExperts):
                # gate_up_proj and down_proj are created with torch.empty;
                # under init_empty_weights they end up as uninitialized garbage.
                nn.init.normal_(m.gate_up_proj, std=m.dim ** -0.5)
                nn.init.normal_(m.down_proj, std=m.ffn_dim ** -0.5)
            elif hasattr(m, 'modulation') and isinstance(m.modulation, nn.Parameter):
                # WanAttentionBlock and Head use modulation for time conditioning.
                # Under init_empty_weights the original torch.randn values are lost,
                # so we re-initialize with the same std used in __init__.
                nn.init.normal_(m.modulation, std=m.modulation.shape[-1] ** -0.5)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.time_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.zeros_(self.head.head.weight)
        if self.head.head.bias is not None:
            nn.init.zeros_(self.head.head.bias)


class MLPProj(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, flf_pos_emb=False, clip_token_len=257, fp32_calculate=False):
        super().__init__()
        self.proj = nn.Sequential(
            _FP32LayerNorm(in_dim) if fp32_calculate else nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            _FP32LayerNorm(out_dim) if fp32_calculate else nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:
            self.emb_pos = nn.Parameter(torch.zeros(1, clip_token_len * 2, in_dim))

    def forward(self, image_emb):
        if hasattr(self, 'emb_pos'):
            bs, n, d = image_emb.shape
            image_emb = image_emb.view(-1, 2 * n, d)
            image_emb = image_emb + self.emb_pos
        return self.proj(image_emb)


class WanDiTFSDP2(WanModel):
    """WanModel wrapper adapting the native implementation to FSDP2 batch-tensor interface."""

    def __init__(
        self,
        model_type: str = "t2v",
        patch_size: Tuple[int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        hidden_size: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        img_dim: int = 1280,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        qk_norm: bool = True,
        qk_norm_type: str = "rmsnorm",
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        max_seq_len: int = 1024,
        clip_token_len: int = 257,
        fp32_calculate: bool = False,
        seperated_timestep: bool = False,
        **kwargs,
    ):
        # Map model_type aliases used in MindSpeed-MM config
        native_model_type = model_type
        if model_type in ("wan2_2-t2v",):
            native_model_type = "t2v"
        elif model_type in ("wan2_2-i2v",):
            native_model_type = "i2v"

        num_experts = kwargs.pop('num_experts', 1)
        top_k = kwargs.pop('top_k', 1)
        attn_implementation = kwargs.pop('attn_implementation', 'eager')
        super().__init__(
            model_type=native_model_type,
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=hidden_size,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=(-1, -1),
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            num_experts=num_experts,
            top_k=top_k,
            attn_implementation=attn_implementation,
        )
        self.img_dim = img_dim
        self.max_seq_len = max_seq_len
        self.clip_token_len = clip_token_len
        self.fp32_calculate = fp32_calculate
        self.seperated_timestep = seperated_timestep

        if native_model_type in ["i2v", "flf2v"]:
            self.img_emb = MLPProj(img_dim, hidden_size, native_model_type == 'flf2v', clip_token_len, fp32_calculate)

    @property
    def dtype(self) -> torch.dtype:
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        buffers = tuple(self.buffers())
        return buffers[0].dtype

    @property
    def device(self) -> torch.device:
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].device
        buffers = tuple(self.buffers())
        return buffers[0].device

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        prompt: torch.Tensor,
        prompt_mask: torch.Tensor = None,
        i2v_clip_feature: torch.Tensor = None,
        i2v_vae_feature: torch.Tensor = None,
        **kwargs,
    ):
        target_dtype = self.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if isinstance(timestep, torch.Tensor) and timestep.dtype != target_dtype:
            timestep = timestep.to(target_dtype)
        if isinstance(prompt, torch.Tensor) and prompt.dtype != target_dtype:
            prompt = prompt.to(target_dtype)
        if isinstance(prompt_mask, torch.Tensor) and prompt_mask.dtype != target_dtype:
            prompt_mask = prompt_mask.to(target_dtype)
        if i2v_clip_feature is not None and i2v_clip_feature.dtype != target_dtype:
            i2v_clip_feature = i2v_clip_feature.to(target_dtype)
        if i2v_vae_feature is not None and i2v_vae_feature.dtype != target_dtype:
            i2v_vae_feature = i2v_vae_feature.to(target_dtype)

        # i2v condition concatenation
        if self.model_type in ["i2v", "flf2v"]:
            if i2v_clip_feature is not None:
                i2v_clip_feature = i2v_clip_feature.to(x)
            if i2v_vae_feature is not None:
                i2v_vae_feature = i2v_vae_feature.to(x)
            x = torch.cat([x, i2v_vae_feature], dim=1)
        elif self.model_type in ["wan2_2-i2v"]:
            if i2v_vae_feature is not None:
                i2v_vae_feature = i2v_vae_feature.to(x)
            x = torch.cat([x, i2v_vae_feature], dim=1)

        # Convert batch tensor to List[Tensor] for native WanModel.forward
        if isinstance(x, torch.Tensor):
            x_list = [x[i] for i in range(x.size(0))]
        else:
            x_list = x

        if isinstance(prompt, torch.Tensor):
            context_list = [prompt[i] for i in range(prompt.size(0))]
        else:
            context_list = prompt

        # Handle timestep
        if isinstance(timestep, torch.Tensor):
            if timestep.ndim == 0:
                timestep = timestep.unsqueeze(0)
            t = timestep
        else:
            t = torch.tensor([timestep], device=x_list[0].device, dtype=x_list[0].dtype)

        sample = x_list[0]
        seq_len = (sample.shape[1] // self.patch_size[0]) * (
            sample.shape[2] // self.patch_size[1]) * (sample.shape[3] // self.patch_size[2])

        ps = get_parallel_state()
        if ps.is_ulysses_enable():
            ulysses_size = ps.get_ulysses_group_size()
            seq_len = ((seq_len + ulysses_size - 1) // ulysses_size) * ulysses_size

        # y for i2v (already concatenated into x above, so y=None)
        y = None

        if self.model_type in ["i2v", "flf2v"] and i2v_clip_feature is not None:
            clip_embedding = self.img_emb(
                i2v_clip_feature.float() if self.fp32_calculate else i2v_clip_feature.to(x_list[0].dtype)
            )
            context_list = [
                torch.cat([clip_embedding[i], context_list[i]], dim=0)
                for i in range(len(context_list))
            ]

        # Derive actual text lengths from prompt_mask so cross-attention can mask
        # padding positions.  If prompt_mask is unavailable we fall back to full
        # length for every sample.
        if isinstance(prompt_mask, torch.Tensor) and prompt_mask.numel() > 0:
            _context_lens = prompt_mask.sum(dim=-1).long().tolist()
        else:
            _context_lens = None

        out_list = super().forward(
            x=x_list, t=t, context=context_list, seq_len=seq_len, y=y,
            context_lens=_context_lens,
        )
        out = torch.stack(out_list, dim=0)
        return out
