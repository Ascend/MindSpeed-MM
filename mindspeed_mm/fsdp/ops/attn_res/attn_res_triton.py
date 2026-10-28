# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
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
"""attn_res 融合 Triton 算子（前向 + 反向合一接口）。

Kimi K3 learned attn-residual：对每个 token 的 N+1 条 residual 流
（N 个 block_residual 快照 + 当前 prefix_sum）做 softmax 加权混合。

融合内容（kernel 内完成，无 host 侧临时张量）：
  - score_weight = norm_w * proj_w（kernel 内乘法）
  - v = cat(block_residual, prefix_sum.unsqueeze(1))（双指针读取）

反向通过 torch.autograd.Function 接入：前向把 online-softmax 的归约量
（score/rstd/max/sumexp）保存下来，反向 kernel 直接复用，避免重算。
"""
import torch
import triton
import triton.language as tl


_NUM_CORES = None


def _get_num_cores():
    global _NUM_CORES
    if _NUM_CORES is None:
        try:
            device = torch.npu.current_device()
            props = triton.runtime.driver.active.utils.get_device_properties(device)
            num_cores = props.get("num_vectorcore", -1)
            if num_cores <= 0:
                num_cores = props.get("num_aicore", 48)
            _NUM_CORES = max(int(num_cores), 1)
        except Exception:
            _NUM_CORES = 48
    return _NUM_CORES


@triton.jit
def _apply_attn_res_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    out_ptr,
    saved_score_ptr,
    saved_rstd_ptr,
    saved_max_ptr,
    saved_sumexp_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    block_size = (T - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= T:
        return
    tok1 = tl.minimum(tok0 + block_size, T)

    cols = tl.arange(0, H)                                   # full-row (non-pow2 OK on Ascend)
    tok_offs = tl.arange(0, SUB_BLOCK_SIZE)                  # sub-block token offsets

    # Fused mul: score_weight = norm_w * proj_w (computed once, resident)
    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    br_stride = N * H  # block_residual row stride
    inv_H = 1.0 / H

    for tok_base in range(tok0, tok1, SUB_BLOCK_SIZE):
        cur_toks = tok_offs + tok_base                       # [SUB_BLOCK_SIZE]
        tok_mask = cur_toks < tok1                            # boundary mask
        mask2d = tok_mask[:, None]                            # [SUB_BLOCK_SIZE, 1]

        # Online softmax: single-pass fused score + weighted sum (no stream reload)
        max_score = tl.full([SUB_BLOCK_SIZE], -float("inf"), dtype=tl.float32)
        exp_acc = tl.zeros([SUB_BLOCK_SIZE], dtype=tl.float32)
        out_acc = tl.zeros([SUB_BLOCK_SIZE, H], dtype=tl.float32)

        for s in range(N):
            ptrs = block_residual_ptr + cur_toks[:, None] * br_stride + s * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            vv_sum = tl.sum(v * v, axis=1)
            vw_sum = tl.sum(v * w[None, :], axis=1)
            rstd = tl.rsqrt(vv_sum * inv_H + EPS)
            score = vw_sum * rstd

            tl.store(saved_score_ptr + cur_toks * (N + 1) + s, score, mask=tok_mask)
            tl.store(saved_rstd_ptr + cur_toks * (N + 1) + s, rstd, mask=tok_mask)

            m_new = tl.maximum(max_score, score)
            correction = tl.exp(max_score - m_new)
            exp_score = tl.exp(score - m_new)
            out_acc = out_acc * correction[:, None] + exp_score[:, None] * v
            exp_acc = exp_acc * correction + exp_score
            max_score = m_new

        # prefix_sum stream (s == N)
        ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
        v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        vv_sum = tl.sum(v * v, axis=1)
        vw_sum = tl.sum(v * w[None, :], axis=1)
        rstd = tl.rsqrt(vv_sum * inv_H + EPS)
        score = vw_sum * rstd

        tl.store(saved_score_ptr + cur_toks * (N + 1) + N, score, mask=tok_mask)
        tl.store(saved_rstd_ptr + cur_toks * (N + 1) + N, rstd, mask=tok_mask)

        m_new = tl.maximum(max_score, score)
        correction = tl.exp(max_score - m_new)
        exp_score = tl.exp(score - m_new)
        out_acc = out_acc * correction[:, None] + exp_score[:, None] * v
        exp_acc = exp_acc * correction + exp_score
        max_score = m_new

        tl.store(saved_max_ptr + cur_toks, max_score, mask=tok_mask)
        tl.store(saved_sumexp_ptr + cur_toks, exp_acc, mask=tok_mask)

        out = out_acc / exp_acc[:, None]

        tl.store(
            out_ptr + cur_toks[:, None] * H + cols[None, :],
            out.to(out_ptr.dtype.element_ty),
            mask=mask2d,
        )


@triton.jit
def _apply_attn_res_backward_kernel(
    block_residual_ptr,
    prefix_sum_ptr,
    norm_w_ptr,
    proj_w_ptr,
    grad_out_ptr,
    grad_block_residual_ptr,
    grad_prefix_sum_ptr,
    grad_w_partials_ptr,
    saved_score_ptr,
    saved_rstd_ptr,
    saved_max_ptr,
    saved_sumexp_ptr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    NUM_CORES: tl.constexpr,
    NB: tl.constexpr,
    SUB_BLOCK_SIZE: tl.constexpr,
):
    block_size = (T - 1) // NUM_CORES + 1
    pid = tl.program_id(0)
    tok0 = pid * block_size
    if tok0 >= T:
        return
    tok1 = tl.minimum(tok0 + block_size, T)

    cols = tl.arange(0, H)
    s_idx = tl.arange(0, NB)
    tok_offs = tl.arange(0, SUB_BLOCK_SIZE)

    norm_w = tl.load(norm_w_ptr + cols).to(tl.float32)
    proj_w = tl.load(proj_w_ptr + cols).to(tl.float32)
    w = norm_w * proj_w

    br_stride = N * H
    inv_H = 1.0 / H
    grad_w_local = tl.zeros([H], dtype=tl.float32)

    for tok_base in range(tok0, tok1, SUB_BLOCK_SIZE):
        cur_toks = tok_offs + tok_base
        tok_mask = cur_toks < tok1
        mask2d = tok_mask[:, None]

        g = tl.load(
            grad_out_ptr + cur_toks[:, None] * H + cols[None, :],
            mask=mask2d, other=0.0,
        ).to(tl.float32)

        # Load saved softmax reduction values from forward (avoids recomputing
        # vv_sum, vw_sum, rstd, score, scores_max, sum_exp).
        scores_max = tl.load(saved_max_ptr + cur_toks, mask=tok_mask, other=-float("inf"))
        sum_exp = tl.load(saved_sumexp_ptr + cur_toks, mask=tok_mask, other=1.0)

        # Pass 1: compute gv_s only; use saved scores for weighted_gv accumulation.
        gv_arr = tl.full([NB, SUB_BLOCK_SIZE], 0.0, dtype=tl.float32)
        weighted_gv = tl.zeros([SUB_BLOCK_SIZE], dtype=tl.float32)

        for s in range(N):
            ptrs = block_residual_ptr + cur_toks[:, None] * br_stride + s * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            gv_s = tl.sum(g * v, axis=1)
            score_s = tl.load(saved_score_ptr + cur_toks * (N + 1) + s, mask=tok_mask, other=-float("inf"))
            weighted_gv += tl.exp(score_s - scores_max) * gv_s
            gv_arr = tl.where(s_idx[:, None] == s, gv_s[None, :], gv_arr)

        ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
        v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        gv_s = tl.sum(g * v, axis=1)
        score_s = tl.load(saved_score_ptr + cur_toks * (N + 1) + N, mask=tok_mask, other=-float("inf"))
        weighted_gv += tl.exp(score_s - scores_max) * gv_s
        gv_arr = tl.where(s_idx[:, None] == N, gv_s[None, :], gv_arr)

        weighted_gv = weighted_gv / sum_exp

        # Pass 2: compute gradients using saved rstd, score (no recompute).
        for s in range(N):
            ptrs = block_residual_ptr + cur_toks[:, None] * br_stride + s * H + cols[None, :]
            v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
            rstd_s = tl.load(saved_rstd_ptr + cur_toks * (N + 1) + s, mask=tok_mask, other=0.0)
            score_s = tl.load(saved_score_ptr + cur_toks * (N + 1) + s, mask=tok_mask, other=0.0)
            gv_s = tl.sum(tl.where(s_idx[:, None] == s, gv_arr, 0.0), axis=0)

            k = v * rstd_s[:, None]
            prob = tl.exp(score_s - scores_max) / sum_exp
            delta = gv_s - weighted_gv
            dscore_dv = (
                rstd_s[:, None] * w[None, :]
                - inv_H * v * rstd_s[:, None] * rstd_s[:, None] * score_s[:, None]
            )
            grad_v = prob[:, None] * (g + delta[:, None] * dscore_dv)

            tl.store(
                grad_block_residual_ptr + cur_toks[:, None] * br_stride + s * H + cols[None, :],
                grad_v.to(grad_block_residual_ptr.dtype.element_ty),
                mask=mask2d,
            )
            grad_w_local += tl.sum(prob[:, None] * delta[:, None] * k, axis=0)

        ptrs = prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :]
        v = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        rstd_s = tl.load(saved_rstd_ptr + cur_toks * (N + 1) + N, mask=tok_mask, other=0.0)
        score_s = tl.load(saved_score_ptr + cur_toks * (N + 1) + N, mask=tok_mask, other=0.0)
        gv_s = tl.sum(tl.where(s_idx[:, None] == N, gv_arr, 0.0), axis=0)

        k = v * rstd_s[:, None]
        prob = tl.exp(score_s - scores_max) / sum_exp
        delta = gv_s - weighted_gv
        dscore_dv = (
            rstd_s[:, None] * w[None, :]
            - inv_H * v * rstd_s[:, None] * rstd_s[:, None] * score_s[:, None]
        )
        grad_v = prob[:, None] * (g + delta[:, None] * dscore_dv)

        tl.store(
            grad_prefix_sum_ptr + cur_toks[:, None] * H + cols[None, :],
            grad_v.to(grad_prefix_sum_ptr.dtype.element_ty),
            mask=mask2d,
        )
        grad_w_local += tl.sum(prob[:, None] * delta[:, None] * k, axis=0)

    tl.store(grad_w_partials_ptr + pid * H + cols, grad_w_local)


class AttnResFunction(torch.autograd.Function):
    """Autograd wrapper for fused attention-residual forward and backward."""

    @staticmethod
    def forward(ctx, prefix_sum, block_residual, proj_w, norm_w, eps):
        prefix_sum = prefix_sum.contiguous()
        block_residual = block_residual.contiguous()
        proj_w = proj_w.contiguous()
        norm_w = norm_w.contiguous()
        T, H = prefix_sum.shape
        N = block_residual.shape[1]
        out = torch.empty_like(prefix_sum)
        NB = max(triton.next_power_of_2(N + 1), 8)  # 8 is to fix compile-err

        # Buffers for online-softmax reduction values reused by backward.
        saved_score = torch.empty((T, N + 1), dtype=torch.float32, device=prefix_sum.device)
        saved_rstd = torch.empty((T, N + 1), dtype=torch.float32, device=prefix_sum.device)
        saved_max = torch.empty((T,), dtype=torch.float32, device=prefix_sum.device)
        saved_sumexp = torch.empty((T,), dtype=torch.float32, device=prefix_sum.device)

        num_cores = _get_num_cores()
        _apply_attn_res_kernel[(num_cores,)](
            block_residual,
            prefix_sum,
            norm_w,
            proj_w,
            out,
            saved_score,
            saved_rstd,
            saved_max,
            saved_sumexp,
            T=T,
            H=H,
            N=N,
            EPS=eps,
            NUM_CORES=num_cores,
            NB=NB,
            multibuffer=True,
            SUB_BLOCK_SIZE=1
        )

        ctx.save_for_backward(
            prefix_sum, block_residual, proj_w, norm_w,
            saved_score, saved_rstd, saved_max, saved_sumexp,
        )
        ctx.eps = eps
        ctx.N = N
        ctx.NB = NB
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (
            prefix_sum, block_residual, proj_w, norm_w,
            saved_score, saved_rstd, saved_max, saved_sumexp,
        ) = ctx.saved_tensors
        T, H = prefix_sum.shape
        num_cores = _get_num_cores()

        grad_prefix_sum = torch.zeros_like(prefix_sum)
        grad_block_residual = torch.zeros_like(block_residual)
        grad_w_partials = torch.zeros(
            (num_cores, H),
            dtype=torch.float32,
            device=prefix_sum.device,
        )

        # fp32 inputs double the UB footprint of the stream buffers; with
        # multibuffer the triton-ascend memory planner (hivm-plan-memory) fails
        # to compile the backward kernel, so only bf16/fp16 use multibuffer.
        # 另外 T <= num_cores 时每核只有 1 个 token, 循环体被完全展开, UB 需求
        # 也会超限, 这种小 T 情形同样关闭 multibuffer。
        use_multibuffer = block_residual.dtype != torch.float32 and T > num_cores

        _apply_attn_res_backward_kernel[(num_cores,)](
            block_residual,
            prefix_sum,
            norm_w,
            proj_w,
            grad_out.contiguous(),
            grad_block_residual,
            grad_prefix_sum,
            grad_w_partials,
            saved_score,
            saved_rstd,
            saved_max,
            saved_sumexp,
            T=T,
            H=H,
            N=ctx.N,
            EPS=ctx.eps,
            NUM_CORES=num_cores,
            NB=ctx.NB,
            multibuffer=use_multibuffer,
            SUB_BLOCK_SIZE=1
        )

        grad_w = grad_w_partials.sum(dim=0)
        grad_proj_w = grad_w * norm_w.float()
        grad_norm_w = grad_w * proj_w.float()
        return (
            grad_prefix_sum,
            grad_block_residual,
            grad_proj_w,
            grad_norm_w,
            None,
        )


def attn_res_triton(
    prefix_sum: torch.Tensor,      # [T, H]
    block_residual: torch.Tensor,  # [T, N, H]
    proj: torch.nn.Linear,         # weight [1, H]
    norm: torch.nn.Module,         # weight [H], variance_epsilon (float)
) -> torch.Tensor:
    """K3 learned attn-residual: softmax-mix N+1 residual streams per token.

    Args:
        prefix_sum: [T, H] (current running sum).
        block_residual: [T, N, H] (N past block snapshots).
        proj: nn.Linear(H, 1) — learned per-channel scoring projection.
        norm: KimiRMSNorm-like — has .weight [H] and .variance_epsilon (float).

    Returns:
        [T, H] — the softmax-weighted mix of the N+1 streams.
    """
    proj_w = proj.weight.reshape(-1)
    norm_w = norm.weight
    eps = norm.variance_epsilon

    assert block_residual.dim() == 3 and block_residual.shape[0] == prefix_sum.shape[0]
    assert block_residual.shape[2] == prefix_sum.shape[1]
    assert proj_w.numel() == prefix_sum.shape[1]
    assert norm_w.numel() == prefix_sum.shape[1]

    return AttnResFunction.apply(prefix_sum, block_residual, proj_w, norm_w, eps)
