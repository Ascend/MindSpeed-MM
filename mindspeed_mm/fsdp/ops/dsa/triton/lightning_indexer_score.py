# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _lightning_indexer_score_kernel(
    q_ptr,          # [B, S, NH, D]
    k_ptr,          # [B, P, D]
    w_ptr,          # [B, S, NH]
    valid_ptr,      # [B, S, P]   bool / int8
    out_ptr,        # [B, S, P]   float32
    S,
    P,
    scale,
    neg_inf,
    NH: tl.constexpr,
    D: tl.constexpr,
    BS: tl.constexpr,
    BP: tl.constexpr,
    HAS_VALID: tl.constexpr,
):
    """Fused DSA (lightning) indexer scoring.

    Computes, without ever materialising the per-head score tensor::

        out[b, s, p] = sum_h w[b, s, h] * relu(scale * sum_d q[b, s, h, d] * k[b, p, d])

    and, when ``HAS_VALID``, replaces entries whose candidate mask is False with
    ``neg_inf`` so the caller can feed the result straight into ``topk``.

    The eager version keeps ``[B, S, NH, P]`` scores alive between the two
    matmuls; here that tensor only ever exists as a ``[BS, BP]`` tile on chip.
    """
    pid_s = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_b = tl.program_id(2)

    s0 = pid_s * BS
    p0 = pid_p * BP

    q_batch_off = pid_b * S * NH * D
    w_batch_off = pid_b * S * NH
    k_batch_off = pid_b * P * D
    o_batch_off = pid_b * S * P

    # k tile is reused by every head, so it is loaded once and kept on chip.
    p_k = tl.make_block_ptr(
        k_ptr + k_batch_off, (P, D), (D, 1), (p0, 0), (BP, D), (1, 0)
    )
    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_kt = tl.trans(b_k)

    acc = tl.zeros((BS, BP), dtype=tl.float32)
    for i_h in range(NH):
        # q[b, s0:s0+BS, i_h, :] -- token stride is NH * D because the head axis
        # sits between the sequence and feature axes.
        p_q = tl.make_block_ptr(
            q_ptr + q_batch_off + i_h * D, (S, D), (NH * D, 1), (s0, 0), (BS, D), (1, 0)
        )
        b_q = tl.load(p_q, boundary_check=(0, 1))

        # The D-wide reduction has to be fp32: the result feeds a topk, and a
        # bfloat16 accumulator would reorder the selected pools. It already is --
        # `tl.dot` returns fp32 for bfloat16 operands and the cube accumulates in
        # fp32, so this is exact to ~1e-7 relative against an fp32 matmul of the
        # same values (a true bfloat16 accumulation is off by ~2e-2). Upcasting
        # b_q / b_kt to fp32 first changes nothing but costs ~2.5x -- do not.
        b_s = tl.dot(b_q, b_kt)
        b_s = tl.maximum(b_s * scale, 0.0)

        p_w = tl.make_block_ptr(
            w_ptr + w_batch_off + i_h, (S,), (NH,), (s0,), (BS,), (0,)
        )
        b_w = tl.load(p_w, boundary_check=(0,)).to(tl.float32)

        acc += b_w[:, None] * b_s

    if HAS_VALID:
        p_v = tl.make_block_ptr(
            valid_ptr + o_batch_off, (S, P), (P, 1), (s0, p0), (BS, BP), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1))
        acc = tl.where(b_v != 0, acc, neg_inf)

    p_o = tl.make_block_ptr(
        out_ptr + o_batch_off, (S, P), (P, 1), (s0, p0), (BS, BP), (1, 0)
    )
    tl.store(p_o, acc, boundary_check=(0, 1))
