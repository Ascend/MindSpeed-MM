import torch
from torch import Tensor
from torch.distributed import ProcessGroup

from megatron.core import mpu
from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import all_to_all


_TOTAL_SEQ_LEN = None
_VISUAL_SEQ_LEN = None


def get_seq_len(des: str = None) -> int:
    if des == "total":
        global _TOTAL_SEQ_LEN
        return _TOTAL_SEQ_LEN
    elif des == "visual":
        global _VISUAL_SEQ_LEN
        return _VISUAL_SEQ_LEN


def set_seq_len(des: str = None, seq_len: int = -1) -> None:
    if des == "total":
        global _TOTAL_SEQ_LEN
        _TOTAL_SEQ_LEN = seq_len
    elif des == "visual":
        global _VISUAL_SEQ_LEN
        _VISUAL_SEQ_LEN = seq_len


def gather_seq_scatter_heads(
    input: Tensor,
    seq_dim: int,
    head_dim: int,
    gather_size: int,
    group: ProcessGroup = None
) -> Tensor:
    group = mpu.get_context_parallel_group() if group is None else group
    if not group:
        return input

    return all_to_all(input, group, scatter_dim=head_dim, gather_dim=seq_dim, gather_size=gather_size)


def gather_heads_scatter_seq(
    input: Tensor, 
    head_dim: int, 
    seq_dim: int, 
    gather_size: int,
    group: ProcessGroup = None
) -> Tensor:
    group = mpu.get_context_parallel_group() if group is None else group
    if not group:
        return input

    return all_to_all(input, group, scatter_dim=seq_dim, gather_dim=head_dim, gather_size=gather_size)


def gather_seq_scatter_heads_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, seq_dim: int, head_dim: int, gather_size: int, group: ProcessGroup = None):
    q = gather_seq_scatter_heads(q, seq_dim, head_dim, gather_size, group)
    k = gather_seq_scatter_heads(k, seq_dim, head_dim, gather_size, group)
    v = gather_seq_scatter_heads(v, seq_dim, head_dim, gather_size, group)
    return q, k, v