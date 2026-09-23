# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
"""KV-AllGather context parallelism: the attention-side Q/K/V transforms.

U×K topology: total_cp = ulysses_size × kvallgather_size. The ulysses all-to-all
gathers its subgroup's sequence extent (S -> S/kv) and scatters heads; K/V are then
all-gathered across the kvallgather group to the FULL sequence (natural order via the
kvallgather-outer cp mesh), so every rank can run exact causal attention on its
local Q block against the full K/V.
"""

from typing import List, Optional, Union

import torch

from ...parallel_state import get_parallel_state
from ..communication import (
    all_to_all,
    gather_forward_split_backward,
    kv_all_gather_forward_reduce_scatter_backward,
)
from ..utils import cal_split_sizes


class _GradScaleBackwardOnly(torch.autograd.Function):
    """Identity forward; scales grad by `scale` in backward. Used to pre-scale K/V grads
    before kv_all_gather's reduce-scatter: in the packed-varlen (gather-full-Q) mode every
    kv rank computes the IDENTICAL complete dK/dV (same full Q), so the reduce-scatter SUM
    would multiply them by kv_size — pre-scaling by 1/kv_size makes the sum exact."""

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None


def kv_all_gather_attention_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    head_dim_index: int,
    seq_dim_index: int,
    total_seq_len: int,
    is_causal: bool,
    cu_seq_lens_q: Optional[Union[torch.Tensor, List[int]]],
    cu_seq_lens_k: Optional[Union[torch.Tensor, List[int]]],
):
    """Transform Q/K/V for the KV-AllGather attention path (the `if is_kvag` branch of
    ops/flash_attn/flash_attention_forward).

    USP + KV AllGather (local-Q mode, reverted from gather-full Q):
        ulysses a2a gathers the ulysses subgroup (seq -> S/kv), scatters heads.
        K/V are all-gathered across the kvallgather group to the FULL sequence
        (natural order via kvallgather-outer cp mesh).  Q stays local (S/kv) ->
        attention compute is 1/kv AND Q memory is S/kv (not full S).
        Causal: Q is contiguous block [a:b], K/V sliced to [0:b]; is_causal
        (end-aligned) gives Q[i]->K[0:a+i].

    Returns:
        (query, key, value, seq_len, q_head_num, kv_group, kv_split_sizes,
        varlen_full_q, cu_seq_lens_q, cu_seq_lens_k)

        - kv_group / kv_split_sizes: needed by the caller to split the attention
          output back to this rank's block in the varlen_full_q mode.
        - varlen_full_q: True when the packed-varlen gather-full-Q exact mode ran
          (output is the FULL sequence, replicated across the kv group).
        - cu_seq_lens_q/k: rewritten to per-segment local bounds in the local-Q
          varlen mode, passed through otherwise.
    """
    ps = get_parallel_state()
    kv_group = ps.get_kvallgather_group()
    kv_size = ps.get_kvallgather_group_size()
    kv_rank = ps.get_kvallgather_rank()
    # The a2a gather size and the K/V all-gather/reduce-scatter below all
    # assume the sequence is padded to a multiple of kv_size (upstream
    # contract: "the model pads to total SP size"). Without it the local
    # blocks get floor-truncated, the causal K-slice misaligns silently
    # (fixed-len) or downstream errors mention unrelated shapes. (PR review.)
    if total_seq_len % kv_size != 0:
        raise ValueError(
            f"KV-AllGather CP requires total_seq_len ({total_seq_len}) to be a "
            f"multiple of kvallgather_parallel_size ({kv_size}); pad the "
            f"sequence to the total CP size before the attention layers."
        )
    ulysses_gather = total_seq_len // kv_size
    query = all_to_all(
        query, ps.get_ulysses_group(),
        scatter_dim=head_dim_index, gather_dim=seq_dim_index, gather_size=ulysses_gather,
    )
    key = all_to_all(
        key, ps.get_ulysses_group(),
        scatter_dim=head_dim_index, gather_dim=seq_dim_index, gather_size=ulysses_gather,
    )
    value = all_to_all(
        value, ps.get_ulysses_group(),
        scatter_dim=head_dim_index, gather_dim=seq_dim_index, gather_size=ulysses_gather,
    )
    seq_len = query.shape[seq_dim_index]   # S/kv (Q stays local)
    q_head_num = query.shape[head_dim_index]
    # kv all-gather K/V to the full sequence (forward all-gather, backward
    # reduce-scatter sum -- correct because each rank uses a different local Q).
    key = kv_all_gather_forward_reduce_scatter_backward(key, kv_group, dim=seq_dim_index)
    value = kv_all_gather_forward_reduce_scatter_backward(value, kv_group, dim=seq_dim_index)
    _kv_split_sizes = cal_split_sizes(total_seq_len, kv_size)
    varlen_full_q = False
    if cu_seq_lens_q is not None and cu_seq_lens_k is not None:
        _local_q_ok = (
            cu_seq_lens_q == cu_seq_lens_k
            # _a comes from the kv-level split_sizes; it matches the ulysses
            # extent's true start only when the total length divides evenly
            # (always true for fixed-len data). Otherwise _a may be shifted
            # -> fall back to gather-full-Q (known correct) rather than guess.
            and seq_len == _kv_split_sizes[kv_rank]
        )
        if _local_q_ok:
            # local-Q varlenkv exact mode (unconditional when fa_pack_skip_q_gather holds; the memory fix for
            # 27B/35B 1M packs): Q stays local (ulysses extent = this kv
            # block [a,b)); K/V are narrowed to [_k0, b) and per-sample
            # causal is expressed via segment pairing:
            #   sample s=[s0,s1) intersecting block [a,b):
            #     Q seg  = [max(s0,a), min(s1,b))  -> qlen (Q-local coords)
            #     KV seg = [s0, min(s1,b))         -> kvlen (offset by _k0)
            #   where _k0 = the first intersecting sample's s0 -- dropping
            #   [0,_k0) from K/V is exact (Q inside the block causally cannot
            #   see earlier K) and varlen segments must start at 0; cumulative
            #   lengths would misalign global coords (every rank with a>0 wrong).
            #   sparse_mode=3 bottom-right offset = kvlen-qlen = a-s0, exactly
            #   the causal shift of "query starts mid-sample" (the standard
            #   varlenkv semantics of chunked prefill).
            #   Truncating K at b is exact: Q positions <= b-1 causally
            #   cannot see K beyond b.
            # Mathematically equal to gather-full-Q: per-row causal visibility
            #   is identical (in-segment local Q j sees local K <=
            #   j+(kvlen-qlen) <=> global [s0, q]).
            # dK/dV: each rank computes the dK/dV partial sum over its own
            #   (disjoint) Q block against the FULL gathered K/V -> the
            #   kv_all_gather reduce-scatter sums the per-rank partials
            #   exactly (vs gather-full-Q's replicated Q computing the SAME
            #   complete dK/dV on every rank, which needs the x1/kv_size
            #   pre-scale); the output is already local -> no split-back.
            # Memory: full-Q [S,H/U,D] + full output (~16GiB/layer at 27B@1M)
            #   -> two local-block copies (~1.5GiB); FA compute /kv_size.
            # Prerequisite: the NPU kernel supports per-segment qlen!=kvlen
            #   pairing -- verified on NPU (tests/test_fa_local_q.py, git history 3342d70f^).
            _a = sum(_kv_split_sizes[:kv_rank])
            _b = _a + seq_len
            _cu = [int(v) for v in cu_seq_lens_q]  # global sample bounds (last = padded total)
            _q_cum, _k_cum, _k0 = [0], [0], None
            for _i in range(1, len(_cu)):
                _s0, _s1 = _cu[_i - 1], _cu[_i]
                _q0, _q1 = max(_s0, _a), min(_s1, _b)
                if _q1 > _q0:
                    if _k0 is None:
                        _k0 = _s0
                    _q_cum.append(_q_cum[-1] + (_q1 - _q0))
                    # kvlen cumulative bound = the segment's global end - _k0
                    # (the K tensor starts at _k0; adjacent sample segments
                    # are naturally contiguous with no gap)
                    _k_cum.append(min(_s1, _b) - _k0)
            # Invariants: segments cover the whole Q block / K segments tile
            # [_k0, b) / cu's last value >= b. All guaranteed upstream
            # (cu[-1] = padded total, non-empty block); a violation means an
            # upstream contract broke -- raise explicitly, never compute wrong silently.
            if (
                _k0 is None
                or _q_cum[-1] != seq_len
                or _k_cum[-1] != _b - _k0
                or _cu[-1] < _b
            ):
                raise RuntimeError(
                    f"fa_pack_skip_q_gather segment-pairing invariant violated: "
                    f"block=[{_a},{_b}), seq_len={seq_len}, "
                    f"q_cum[-1]={_q_cum[-1]}, "
                    f"k0={_k0}, k_cum[-1]={_k_cum[-1] if _k0 is not None else None}, "
                    f"cu[-1]={_cu[-1]} -- check cu coverage vs kv split"
                )
            key = key.narrow(seq_dim_index, _k0, _b - _k0)
            value = value.narrow(seq_dim_index, _k0, _b - _k0)
            cu_seq_lens_q = _q_cum
            cu_seq_lens_k = list(_k_cum)
        else:
            # PACKED varlen: local-Q + plain causal over the K-slice treats the whole
            # packed bin as ONE sequence — every sample attends to all previous samples
            # (cross-sample leakage, ~2% loss offset; the model passes the GLOBAL cu
            # here exactly for this). Restore the exact gather-full-Q mode: all_gather
            # Q across the kv group, keep K/V FULL (no narrow — per-sample causality
            # comes from the actual_seq boundaries + sparse_mode=3 below, and padding
            # beyond cu[-1] is excluded by the last segment's boundary), then split the
            # output back to this rank's block. Numerically identical to the
            # ulysses16 full-sequence varlen path. Q blocks are DISJOINT across ranks
            # (not replicated) => grad_scale=None on both sides (the default up/down
            # scaling would mis-scale grads by kv_size).
            query = gather_forward_split_backward(
                query, kv_group, dim=seq_dim_index,
                gather_sizes=_kv_split_sizes, grad_scale=None,
            )
            seq_len = query.shape[seq_dim_index]   # full S
            varlen_full_q = True
            # K/V grad de-duplication: with the FULL (replicated) Q, every kv rank's
            # attention backward produces the IDENTICAL complete dK/dV, so the
            # reduce-scatter inside kv_all_gather_forward_reduce_scatter_backward
            # would SUM kv_size identical copies (kv_size-fold grad blow-up; observed
            # grad norm 102k vs expected ~50). Pre-scale by 1/kv_size so the sum is
            # exact. (In the old local-Q mode each rank's dK/dV covers disjoint Q
            # blocks — the plain sum is correct there, hence only scale here.)
            key = _GradScaleBackwardOnly.apply(key, 1.0 / kv_size)
            value = _GradScaleBackwardOnly.apply(value, 1.0 / kv_size)
    elif is_causal:
        # Q is the contiguous block [a:b] with a = kv_rank * seq_len.  Slice K/V
        # to [0:b]; is_causal (end-aligned, sparse_mode=3) gives Q[i]->K[0:a+i].
        k_len = (kv_rank + 1) * seq_len
        key = key.narrow(seq_dim_index, 0, k_len)
        value = value.narrow(seq_dim_index, 0, k_len)
    return query, key, value, seq_len, q_head_num, kv_group, _kv_split_sizes, varlen_full_q, cu_seq_lens_q, cu_seq_lens_k
