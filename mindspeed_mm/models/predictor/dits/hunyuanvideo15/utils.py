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
import collections.abc
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import repeat as iter_repeat

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from einops import rearrange, repeat
from torch import nn
from torch.distributed.device_mesh import init_device_mesh


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
        raise AssertionError("sageattn have not been supported")
    # Handle flash attention modes
    if attn_mode == 'flash':
        attn_mode = 'flash2'
    if attn_mode != original_attn_mode and not ('flash' in original_attn_mode and 'flash' in attn_mode):
        warnings.warn(
            f"Falling back from `{original_attn_mode}` to `{attn_mode}` because `{original_attn_mode}` is not properly installed.")
    return attn_mode


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
        self.build_mesh("npu")

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
            self.sp_rank = dist.get_rank()
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


__parallel_dims = None


def initialize_parallel_state(
    sp: int = 1,
    dp_replicate: int = 1,
):
    global __parallel_dims
    __parallel_dims = ParallelDims(sp=sp, dp_replicate=dp_replicate)
    return __parallel_dims


def get_parallel_state():
    if __parallel_dims is None:
        # create default parallel states (without enabling any parallelism)
        initialize_parallel_state()
    return __parallel_dims


def _ntuple(n):
    """Create a function that converts input to n-tuple."""

    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(iter_repeat(x[0], n))
            return x
        return tuple(iter_repeat(x, n))

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


class HyIndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        if indices.ndim != 1:
            raise AssertionError("indices.ndim needs equal 1")
        if values.ndim < 2:
            raise AssertionError("values needs >= 2")
        output = torch.zeros(
            first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = HyIndexPutFirstAxis.apply


class HyIndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        if input.ndim < 2:
            raise AssertionError("input.ndim needs < 2")
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        if grad_output.ndim < 2:
            raise AssertionError("grad_output.ndim needs >= 2")
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = HyIndexFirstAxis.apply


def pad_input(hidden_states, indices, batch, seqlen):
    """
    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.
    Return:
        hidden_states: (batch, seqlen, ...)
    """
    dim = hidden_states.shape[-1]
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    """
    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.
        unused_mask: (batch, seqlen), bool / int, 1 means the element is allocated but unused.
    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int
        seqused: (batch), returns the number of tokens selected in attention_mask + unused_mask.
    """
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def flash_attn_no_pad(
        qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    batch_size, seqlen, _, nheads, head_dim = qkv.shape
    query, key, value = qkv.unbind(dim=2)

    query_unpad, indices, cu_seqlens_q, _, _ = unpad_input(
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

    head_num = query_unpad.shape[1]
    output_unpad = torch_npu.npu_fusion_attention(
        query_unpad, key_unpad, value_unpad, head_num,
        pse=None,
        atten_mask=None,
        scale=1.0 / math.sqrt(query_unpad.shape[-1]),
        keep_prob=1,
        input_layout="TND",
        actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
        actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()))[0]

    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d", h=nheads
    )
    return output


def flash_attn_no_pad_v3(
        qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None, deterministic=False
):
    output = flash_attn_no_pad(qkv, key_padding_mask, causal, dropout_p, softmax_scale, deterministic)
    return output
