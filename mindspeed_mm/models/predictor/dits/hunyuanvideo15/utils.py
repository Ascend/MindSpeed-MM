# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.
import collections.abc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import repeat

import torch
import torch.distributed as dist
from einops import rearrange
from torch import nn
from torch.distributed.device_mesh import init_device_mesh


def is_flash2_available():
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        return True
    except Exception:
        return False


def is_flash3_available():
    try:
        from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3  # noqa: F401
        return True
    except Exception:
        return False


def is_flash_available():
    return is_flash2_available() or is_flash3_available()


def is_sparse_attn_supported():
    return 'nvidia h' in torch.cuda.get_device_properties(0).name.lower()


def is_sparse_attn_available():
    if not is_sparse_attn_supported():
        return False
    try:
        from flex_block_attn import flex_block_attn_func  # noqa: F401
        return True
    except Exception:
        return False


def is_angelslim_available():
    try:
        import angelslim
        return True
    except Exception:
        return False


def maybe_fallback_attn_mode(attn_mode):
    """
    Determine the final attention mode based on configuration and availability.

    Args:
        attn_mode: Requested attention mode
        infer_state: Inference configuration object (optional)
        block_idx: Current block index (optional)

    Returns:
        Final attention mode to use
    """
    import warnings
    original_attn_mode = attn_mode

    if attn_mode in ('flex-block-attn'):
        if not is_sparse_attn_available():
            raise ValueError(f"{attn_mode} is not available for your GPU or flex-block-attn is not properly installed.")

    enable_sageattn = attn_mode == 'sageattn'

    if enable_sageattn and attn_mode == 'flex-block-attn':
        raise ValueError("SageAttention cannot be used with flex-block-attn mode. "
         "Please disable enable_sageattn or use a different attention mode.")

    # Use SageAttention if configured
    if attn_mode == 'sageattn':
        try:
            from sageattention import sageattn
        except Exception:
            attn_mode = 'flash'
    # Handle flash attention modes
    if attn_mode == 'flash':
        if is_flash3_available():
            attn_mode = 'flash3'
        elif is_flash2_available():
            attn_mode = 'flash2'
        else:
            attn_mode = 'torch'
    elif attn_mode == 'flash3':
        if not is_flash3_available():
            attn_mode = 'torch'
    elif attn_mode == 'flash2':
        if not is_flash2_available():
            attn_mode = 'torch'
    if attn_mode != original_attn_mode and not ('flash' in original_attn_mode and 'flash' in attn_mode):
        warnings.warn(
            f"Falling back from `{original_attn_mode}` to `{attn_mode}` because `{original_attn_mode}` is not properly installed.")
    return attn_mode


def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input, unpad_input
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(
        x, key_padding_mask
    )

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def flash_attn_no_pad_v3(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3

    if flash_attn_varlen_func_v3 is None:
        raise ImportError("FlashAttention V3 backend not available")

    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, max_seqlen_q, _ = unpad_input(
        rearrange(query, "b s h d -> b s (h d)"), key_padding_mask
    )
    key_unpad, _, cu_seqlens_k, _, _ = unpad_input(
        rearrange(key, "b s h d -> b s (h d)"), key_padding_mask
    )
    value_unpad, _, _, _, _ = unpad_input(
        rearrange(value, "b s h d -> b s (h d)"), key_padding_mask
    )

    query_unpad = rearrange(query_unpad, "nnz (h d) -> nnz h d", h=nheads)
    key_unpad = rearrange(key_unpad, "nnz (h d) -> nnz h d", h=nheads)
    value_unpad = rearrange(value_unpad, "nnz (h d) -> nnz h d", h=nheads)

    output_unpad = flash_attn_varlen_func_v3(
        query_unpad, key_unpad, value_unpad,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_q,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=deterministic
    )
    if isinstance(output_unpad, tuple):
        # output, softmax_lse
        output_unpad = output_unpad[0]

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads
    )
    return output


@dataclass
class ParallelDims:
    sp: int = 1
    world_size: int = -1
    dp_replicate: int = 1

    def __post_init__(self):
        if self.world_size == -1:
            if dist.is_initialized():
                self.world_size = dist.get_world_size()
            else:
                self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.build_mesh("cuda")

    def build_mesh(self, device_type):
        if self.dp_replicate == -1:
            if self.world_size % 8 != 0:
                raise ValueError("world_size must be divisible by 8 for dp_replicate==-1")
            self.dp_replicate = self.world_size // 8
        if self.world_size % self.sp != 0:
            raise ValueError("world_size must be divisible by sp")
        if self.world_size % self.dp_replicate != 0:
            raise ValueError("world_size must be divisible by dp_replicate")

        fsdp_shard = self.world_size // self.dp_replicate

        mesh = init_device_mesh(
            device_type,
            [self.world_size // self.sp, self.sp],
            mesh_dim_names=["dp", "sp"]
        )
        self.world_mesh = mesh
        self.fsdp_mesh = init_device_mesh(
            device_type,
            [self.dp_replicate, fsdp_shard],
            mesh_dim_names=["dp_replicate", "fsdp_shard"]
        )

        if self.sp_enabled:
            self.sp_rank = mesh['sp'].get_local_rank()
            self.sp_group = mesh['sp'].get_group()
        else:
            self.sp_rank = get_rank()
            self.sp_group = None

        return mesh

    @property
    def sp_enabled(self):
        return self.sp > 1

    @property
    def sp_mesh(self):
        return self.world_mesh['sp']

    @property
    def dp_enabled(self):
        return self.sp > 1


def get_parallel_state():
    return ParallelDims(sp=1, dp_replicate=1)


def _ntuple(n):
    """Create a function that converts input to n-tuple."""
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(repeat(x[0], n))
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)


@contextmanager
def auto_offload_model(models, device, enabled=True):
    if enabled:
        if isinstance(models, nn.Module):
            models = [models]
        for model in models:
            if model is not None:
                model.to(device)
    yield
    if enabled:
        for model in models:
            if model is not None:
                model.to(torch.device('cpu'))


def get_rank():
    return int(os.environ.get('RANK', '0'))
