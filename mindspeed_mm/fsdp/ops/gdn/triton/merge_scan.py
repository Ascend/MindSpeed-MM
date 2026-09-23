# -*- coding: utf-8 -*-
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# Triton merge kernel for GDN scan context-parallel cross-rank prefix compose.
# Replaces the pure-Python for-loop in parallel_scan.py:prefix_scan_compose /
# prefix_scan_compose_bwd. Implements the cross-rank merge on MindSpeed-MM's Ascend
# triton dialect and [M|S_ext] packed layout.
#
# Math (forward, for cp_rank r >= 1):
#   S = 0; for j in 1..r: S = M_{j-1} @ S + S_ext_{j-1}
#   (M = A_p [B,H,K,K], S_ext = B_p [B,H,K,V], both fp32)
# Math (backward, for cp_rank r <= kv-2):
#   dht = 0; for j in kv-1 down to r+1: dht = dM_j @ dht + dS_ext_j
#
# NO gate/exp involvement (M, S_ext pre-computed) -> exp2-vs-tl.exp landmine does NOT apply.
# M@S_ext chain stays fp32 (b_h fp32, inputs cast fp32). Uses BK=next_pow2(K) single block,
# NOT the 64-unroll of fwd_h -- simpler and avoids the "compute all new_h from old
# b_h before assigning" hazard. BV in {32,64} (Ascend caps BV<=64).
#
# A `use_triton` flag in prefix_scan_compose[_bwd] selects triton vs the original Python path
# (Python retained for gradcheck diffing, fallback, non-NPU environments).

import torch
import triton
import triton.language as tl

from .utils import get_autotune_config, pin_autotune_configs


def _merge_autotune_configs():
    """Ascend-idiomatic configs: take get_autotune_config (multibuffer/tile_mix_* -- the knobs
    the Ascend cann triton compiler actually uses) and cross with BV in {32, 64} (the V tile,
    a kernel constexpr). num_warps/num_stages are CUDA-only and ignored on Ascend, so we do NOT
    use them (mirrors chunk_scaled_dot_kkt.py which uses bare triton.Config({'BK': ...}))."""
    base = get_autotune_config(multibuffer_list=(False,))
    configs = []
    for bv in [32, 64]:
        for c in base:
            kw = dict(c.kwargs)
            kw['BV'] = bv
            configs.append(triton.Config(kw))
    return configs


@triton.autotune(
    configs=pin_autotune_configs(_merge_autotune_configs()),
    key=['HV', 'K', 'V'],
)
@triton.jit(do_not_specialize=['num_ranks', 'rank', 'N'])
def merge_fwd_bwd_kernel(
    all_M,          # [cp_size, B, H, K, K] fp32 (contiguous — caller passes .contiguous())
    all_S_ext,      # [cp_size, B, H, K, V] fp32 (contiguous)
    out,            # [B, H, K, V] fp32 (initial_state for fwd, dht for bwd)
    num_ranks,      # int: forward = cp_rank (iterate idx=0..cp_rank-1); backward = cp_size-1-cp_rank
    rank,           # int: cp_rank
    N,              # int: B (batch) -- needed for the cp_size-dim stride (N*HV*K*K)
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,       # next_pow2(K)
    BV: tl.constexpr,
    FORWARD: tl.constexpr,  # True = forward, False = backward
):
    """Merge cross-rank (M, S_ext) pairs: S = M @ S + S_ext over the ranks this rank needs.

    Grid: (cdiv(V, BV), B*HV). Each program: one [BK, BV] column slice for one (n, h).
    Forward:  idx=0..num_ranks-1, cur_rank=idx            (S = M_0@0 + S_ext_0; M_1@S + S_ext_1; ...)
    Backward: idx=0..num_ranks-1, cur_rank=rank+num_ranks-idx  (dht = dM_{kv-1}@0 + dS_ext_{kv-1}; ...)
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV

    o_k = tl.arange(0, BK)
    m_k = o_k < K
    o_vb = i_v * BV + tl.arange(0, BV)
    m_v = o_vb < V

    # S state, single [BK, BV] fp32 block (no 64-unroll -- simpler, avoids update-order hazard).
    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # strides for contiguous [cp_size, B=N, H=HV, K, K] / [cp_size, B, H, K, V]:
    # dim-0 (cp_size) stride = N * dim-1 (B) stride. int64 to avoid overflow for large
    # cp_size*B*H*K*K. (Stride bugs only surface at B>=2 — B=1 hides them.)
    batch_stride_M = HV * K * K
    rank_stride_M = N.to(tl.int64) * batch_stride_M
    batch_stride_S = HV * K * V
    rank_stride_S = N.to(tl.int64) * batch_stride_S
    nh_off_M = i_n * batch_stride_M + i_h * K * K
    nh_off_S = i_n * batch_stride_S + i_h * K * V

    for idx in range(num_ranks):
        if FORWARD:
            cur_rank = idx
        else:
            cur_rank = rank + (num_ranks - idx)
        base_M = all_M + cur_rank.to(tl.int64) * rank_stride_M + nh_off_M
        base_S = all_S_ext + cur_rank.to(tl.int64) * rank_stride_S + nh_off_S

        # load M [BK, BK] (rows o_k, cols o_k) masked to K; S_ext [BK, BV] (rows o_k, cols o_vb)
        p_m = base_M + o_k[:, None] * K + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        p_s = base_S + o_k[:, None] * V + o_vb[None, :]
        b_s = tl.load(p_s, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        # S = M @ S + S_ext  (all from OLD b_h -- single assignment, no update-order hazard).
        # input_precision='ieee': both operands fp32 (large dynamic range from exp(g_last) in A_p);
        # Ascend's default fp32 dot downcasts to tf32/hf32 (~1e-3 relerr) which accumulates over
        # the M-chain. Force full fp32 (same convention as solve_tril).
        b_h = tl.dot(b_m, b_h, input_precision="ieee") + b_s

    # store out [B, H, K, V]: rows o_k, cols o_vb
    p_out = out + i_n * HV * K * V + i_h * K * V + o_k[:, None] * V + o_vb[None, :]
    tl.store(p_out, b_h, mask=m_k[:, None] & m_v[None, :])


def merge_fwd_triton(all_M, all_B_p, cp_rank, cp_size):
    """Forward merge: initial_state for cp_rank. Returns None for rank 0 (no triton launch)."""
    if cp_rank == 0:
        return None  # rank 0 initial_state = 0
    cp_size_, N, H, K, _ = all_M.shape
    V = all_B_p.shape[-1]
    HV = H
    assert all_M.dtype == torch.float32 and all_B_p.dtype == torch.float32
    out = all_M.new_zeros(N, HV, K, V, dtype=torch.float32)
    BK = triton.next_power_of_2(K)
    num_ranks = cp_rank  # iterate j = 0..cp_rank-1
    grid = lambda meta: (triton.cdiv(V, meta['BV']), N * HV)
    # .contiguous(): all_M/all_B_p are views of the packed all_hm_stacked [..., :K] / [..., K:]
    # with NON-contiguous strides (row stride = K+V, not K). The kernel does pointer arithmetic
    # assuming contiguous [cp_size,B,H,K,K] / [cp_size,B,H,K,V] layout, so materialize first.
    merge_fwd_bwd_kernel[grid](
        all_M.contiguous(), all_B_p.contiguous(), out, num_ranks, cp_rank, N,
        HV=HV, K=K, V=V, BK=BK, FORWARD=True,
    )
    return out


def merge_bwd_triton(all_dM, all_dB_p, cp_rank, cp_size):
    """Backward merge: dht for cp_rank. Returns zeros for the last rank
    (num_ranks==0 -> early return, no kernel launch)."""
    cp_size_, N, H, K, _ = all_dM.shape
    V = all_dB_p.shape[-1]
    HV = H
    assert all_dM.dtype == torch.float32 and all_dB_p.dtype == torch.float32
    out = all_dM.new_zeros(N, HV, K, V, dtype=torch.float32)
    BK = triton.next_power_of_2(K)
    num_ranks = cp_size - 1 - cp_rank  # iterate j = cp_size-1 down to cp_rank+1
    if num_ranks == 0:
        return out  # last rank: dht = 0, no launch
    grid = lambda meta: (triton.cdiv(V, meta['BV']), N * HV)
    merge_fwd_bwd_kernel[grid](
        all_dM.contiguous(), all_dB_p.contiguous(), out, num_ranks, cp_rank, N,
        HV=HV, K=K, V=V, BK=BK, FORWARD=False,
    )
    return out
