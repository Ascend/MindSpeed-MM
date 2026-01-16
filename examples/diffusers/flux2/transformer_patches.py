#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 Huawei Technologies Co., Ltd
# Copyright 2025 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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

import inspect
import sys
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch_npu
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2AttnProcessor,
    Flux2ParallelSelfAttention,
    Flux2ParallelSelfAttnProcessor,
    Flux2SwiGLU,
)
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)


def patched_prepare_latents(
    self,
    batch_size,
    num_latents_channels,
    height,
    width,
    dtype,
    device,
    generator: torch.Generator,
    latents: Optional[torch.Tensor] = None,
):
    """
    The original prepare latents will use randn_tensor in dtype based on the input.
    The CPU generator in image to image function is not working as the latents in bf16 is too week.

    1. Generation: generate randomness in float32 on CPU, this ensures the noise is mathematically correct (not zero)
    2. Casting: convert the noise to bfloat16 and move it to the device
    """
    # VAE applies 8x compression on images but we must also account for packing which requires
    # latent height and width to be divisible by 2.
    height = 2 * (int(height) // (self.vae_scale_factor * 2))
    width = 2 * (int(width) // (self.vae_scale_factor * 2))

    shape = (batch_size, num_latents_channels * 4, height // 2, width // 2)
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )
    if latents is None:
        latents = randn_tensor(
            shape, generator=generator, device=device, dtype=torch.float32
        ).to(dtype)
    else:
        latents = latents.to(device=device, dtype=dtype)

    latent_ids = self._prepare_latent_ids(latents)
    latent_ids = latent_ids.to(device)

    latents = self._pack_latents(latents)  # [B, C, H, W] -> [B, H*W, C]
    return latents, latent_ids


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]

    """
    if dim % 2 != 0:
        raise ValueError("dim must be divisible by 2 (even number)")

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (
            theta
            ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device) / dim)
        )
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    is_npu = freqs.device.type == "npu"
    if is_npu:
        freqs = freqs.float()
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = (
            freqs.cos()
            .T.repeat_interleave(2, dim=0, output_size=freqs.shape[1] * 2)
            .T.float()
            .contiguous()
        )  # [S, D]
        freqs_sin = (
            freqs.sin()
            .T.repeat_interleave(2, dim=0, output_size=freqs.shape[1] * 2)
            .T.float()
            .contiguous()
        )  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(
            torch.ones_like(freqs), freqs
        )  # complex64     # [S, D/2]
        return freqs_cis


class RMSNorm_npu(torch.nn.Module):
    """
    Patch the original torch.nn.RMSNorm to RMSNorm_npu.
    The overall performance will increase ~5%
    """

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            # Learnable weight parameter (same as PyTorch's RMSNorm)
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, x):
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]


class PatchedFluxAttention(Flux2Attention):
    _default_processor_cls = Flux2AttnProcessor
    _available_processors = [Flux2AttnProcessor]

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        processor=None,
    ):
        super(Flux2Attention, self).__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.added_kv_proj_dim = added_kv_proj_dim
        self.added_proj_bias = added_proj_bias

        self.to_q = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_k = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)
        self.to_v = torch.nn.Linear(query_dim, self.inner_dim, bias=bias)

        self.norm_q = RMSNorm_npu(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.norm_k = RMSNorm_npu(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.to_out = torch.nn.ModuleList([])
        self.to_out.append(torch.nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(torch.nn.Dropout(dropout))

        if added_kv_proj_dim is not None:
            self.norm_added_q = RMSNorm_npu(dim_head, eps=eps)
            self.norm_added_k = RMSNorm_npu(dim_head, eps=eps)
            self.add_q_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_k_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.add_v_proj = torch.nn.Linear(
                added_kv_proj_dim, self.inner_dim, bias=added_proj_bias
            )
            self.to_add_out = torch.nn.Linear(self.inner_dim, query_dim, bias=out_bias)

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            image_rotary_emb,
            **kwargs,
        )


class PatchedFluxParallelSelfAttention(Flux2ParallelSelfAttention):
    _default_processor_cls = Flux2ParallelSelfAttnProcessor
    _available_processors = [Flux2ParallelSelfAttnProcessor]
    # Does not support QKV fusion as the QKV projections are always fused
    _supports_qkv_fusion = False

    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        out_bias: bool = True,
        eps: float = 1e-5,
        out_dim: int = None,
        elementwise_affine: bool = True,
        mlp_ratio: float = 4.0,
        mlp_mult_factor: int = 2,
        processor=None,
    ):
        super(Flux2ParallelSelfAttention, self).__init__()

        self.head_dim = dim_head
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.heads = out_dim // dim_head if out_dim is not None else heads

        self.use_bias = bias
        self.dropout = dropout

        self.mlp_ratio = mlp_ratio
        self.mlp_hidden_dim = int(query_dim * self.mlp_ratio)
        self.mlp_mult_factor = mlp_mult_factor

        # Fused QKV projections + MLP input projection
        self.to_qkv_mlp_proj = torch.nn.Linear(
            self.query_dim,
            self.inner_dim * 3 + self.mlp_hidden_dim * self.mlp_mult_factor,
            bias=bias,
        )
        self.mlp_act_fn = Flux2SwiGLU()

        # QK Norm
        self.norm_q = RMSNorm_npu(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )
        self.norm_k = RMSNorm_npu(
            dim_head, eps=eps, elementwise_affine=elementwise_affine
        )

        # Fused attention output projection + MLP output projection
        self.to_out = torch.nn.Linear(
            self.inner_dim + self.mlp_hidden_dim, self.out_dim, bias=out_bias
        )

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(
            self, hidden_states, attention_mask, image_rotary_emb, **kwargs
        )


class PatchedFluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        This patch change the shape of freqs from [*, *, 1] -> [1, *, *] that can improve the performance about 5%+
        """
        # Expected ids shape: [S, len(self.axes_dim)]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        # Unlike Flux 1, loop over len(self.axes_dim) rather than ids.shape[-1]
        for i, dim in enumerate(self.axes_dim):
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                pos[..., i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin


def apply_patches():
    module = sys.modules["diffusers.models.transformers.transformer_flux2"]

    module.Flux2Attention = PatchedFluxAttention
    module.Flux2ParallelSelfAttention = PatchedFluxParallelSelfAttention
    module.Flux2PosEmbed = PatchedFluxPosEmbed
