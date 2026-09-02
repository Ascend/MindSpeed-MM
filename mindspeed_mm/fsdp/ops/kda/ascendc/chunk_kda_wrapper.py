# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
"""AscendC KDA: AscendC forward plus the fused AscendC backward.

Supported: chunk_size=64, K=128, V in {128, 256}, H == HV, dense inputs,
post-sigmoid beta, no initial/final state.
"""

import torch

from triton_ascend_kernels.attention.fla.kda.fla_utils import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)
from triton_ascend_kernels.attention.fla.kda.l2norm_kda import l2norm_bwd, l2norm_fwd

from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd

try:
    from fla_npu.ops.ascendc import chunk_kda_bwd as ascendc_chunk_kda_bwd
except (AttributeError, ImportError) as exc:
    raise ImportError(
        "fla_npu does not expose chunk_kda_bwd. The installed wheel is either too "
        "old or was built for a different SoC. Inspect the installed operators with:\n"
        "  ls $(python -c 'import fla_npu, os; print(os.path.dirname(fla_npu.__file__))')"
        "/opp/vendors/*/op_impl/*/ | grep chunk_kda\n"
        "Set kda_implementation: fused to keep training on the Triton path."
    ) from exc

CHUNK_SIZE = 64


def _bsnd_to_bnsd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected a rank-4 BSND tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bnsd_to_bsnd(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected a rank-4 BNSD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected a rank-3 BSH tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor):
    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected a rank-3 BHS tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _check_intermediates(q, v, gk, Aqk, Akk, w, qg, kg, v_new, h):
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[3]
    chunks = (tokens + CHUNK_SIZE - 1) // CHUNK_SIZE
    expected = (
        ("g_cumsum", gk, (batch, heads, tokens, key_dim), torch.float32),
        ("Aqk", Aqk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("Akk", Akk, (batch, heads, tokens, CHUNK_SIZE), q.dtype),
        ("w", w, (batch, heads, tokens, key_dim), q.dtype),
        ("qg", qg, (batch, heads, tokens, key_dim), q.dtype),
        ("kg", kg, (batch, heads, tokens, key_dim), q.dtype),
        ("v_new", v_new, (batch, heads, tokens, value_dim), q.dtype),
        ("h", h, (batch, chunks, heads, key_dim, value_dim), q.dtype),
    )
    for name, tensor, shape, dtype in expected:
        if tensor is None:
            raise RuntimeError(f"AscendC forward did not produce {name}.")
        if tuple(tensor.shape) != shape:
            raise RuntimeError(f"{name} has shape {tuple(tensor.shape)}, expected {shape}.")
        if tensor.dtype != dtype:
            raise RuntimeError(f"{name} has dtype {tensor.dtype}, expected {dtype}.")
        if not tensor.is_contiguous():
            raise RuntimeError(f"{name} is not contiguous.")

_DUMPED = False


def _dump_once(path, **tensors):
    """Save one real call's inputs so the operator can be replayed offline."""
    global _DUMPED
    if _DUMPED:
        return
    _DUMPED = True
    import torch.distributed as dist
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    torch.save({k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in tensors.items()}, path)
    print(f"[kda-dump] wrote {path}", flush=True)

class ChunkKDAAscendCFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        safe_gate: bool = False,
        lower_bound: float | None = None,
    ):
        q_rstd = k_rstd = None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        # disable_recompute keeps w/qg/kg/v_new/h alive for the fused backward.
        (o, final_state, g_cumsum, Aqk, Akk, w, u, qg, kg, v_new, h,
         _) = ascendc_chunk_kda_fwd(
            q, k, v, g, beta,
            float(scale),
            CHUNK_SIZE,
            layout="BSND",
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            chunk_indices=None,
            safe_gate=bool(safe_gate),
            lower_bound=lower_bound,
            use_gate_in_kernel=bool(use_gate_in_kernel),
            A_log=A_log if use_gate_in_kernel else None,
            dt_bias=dt_bias if use_gate_in_kernel else None,
            disable_recompute=True,
            return_intermediate_states=False,
            state_v_first=False,
        )

        _check_intermediates(q, v, g_cumsum, Aqk, Akk, w, qg, kg, v_new, h)

        ctx.save_for_backward(
            q, q_rstd, k, k_rstd, v, g, beta, A_log, dt_bias,
            g_cumsum, Aqk, Akk, w, qg, kg, v_new, h,
        )
        ctx.scale = scale
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_gate_in_kernel = use_gate_in_kernel
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        (q, q_rstd, k, k_rstd, v, g_input, beta, A_log, dt_bias,
         g_cumsum, Aqk, Akk, w, qg, kg, v_new, h) = ctx.saved_tensors

        if dht is not None:
            raise RuntimeError("The fused AscendC KDA backward has no final-state gradient.")

        gate_dt_bias = None
        if ctx.use_gate_in_kernel and dt_bias is not None:
            gate_dt_bias = dt_bias.reshape(q.shape[2], q.shape[3]).contiguous()

        dq_h, dk_h, dv_h, db_h, dg_h, dA, dbias = ascendc_chunk_kda_bwd(
            _bsnd_to_bnsd(q),
            _bsnd_to_bnsd(k),
            _bsnd_to_bnsd(v),
            _bsh_to_bhs(beta),
            g_cumsum, Aqk, Akk, w, qg, kg, v_new, h,
            _bsnd_to_bnsd(do),
            float(ctx.scale),
            raw_g=_bsnd_to_bnsd(g_input) if ctx.use_gate_in_kernel else None,
            A_log=A_log if ctx.use_gate_in_kernel else None,
            dt_bias=gate_dt_bias,
            initial_state=None,
            dht=None,
            cu_seqlens=None,
            chunk_indices=None,
            chunk_size=CHUNK_SIZE,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=True,
            use_exp2=True,
            state_v_first=False,
        )

        dq, dk, dv = map(_bnsd_to_bsnd, (dq_h, dk_h, dv_h))
        db, dg = _bhs_to_bsh(db_h), _bnsd_to_bsnd(dg_h)
        if dbias is not None:
            dbias = dbias.reshape(dt_bias.shape)

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        # One gradient per positional argument of apply().
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g_input), db.to(beta),
                dA, dbias, None, None, None, None, None)


@torch.compiler.disable
def chunk_kda_ascendc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    **kwargs,
):
    if cu_seqlens is not None:
        raise NotImplementedError("chunk_kda_ascendc supports dense inputs only.")
    if initial_state is not None:
        raise NotImplementedError("chunk_kda_ascendc does not support initial_state.")
    if output_final_state:
        raise NotImplementedError("chunk_kda_ascendc does not support output_final_state.")
    if kwargs.get("use_beta_sigmoid_in_kernel"):
        raise NotImplementedError("chunk_kda_ascendc expects post-sigmoid beta.")
    chunk_size = kwargs.pop("chunk_size", CHUNK_SIZE)
    if chunk_size != CHUNK_SIZE:
        raise ValueError(f"chunk_size must be {CHUNK_SIZE}, got {chunk_size}.")

    A_log = dt_bias = None
    if use_gate_in_kernel:
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if A_log.dtype != torch.float32:
            raise TypeError(f"A_log must be float32, got {A_log.dtype}.")
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            raise TypeError(f"dt_bias must be float32, got {dt_bias.dtype}.")
        if safe_gate:
            if lower_bound is None:
                raise ValueError("lower_bound is required when safe_gate=True.")
            if not -5 <= lower_bound < 0:
                raise ValueError(f"lower_bound must be in [-5, 0), got {lower_bound}.")

    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if q.shape != k.shape:
        raise ValueError(f"q and k must match, got {tuple(q.shape)} vs {tuple(k.shape)}.")
    if H != HV:
        raise NotImplementedError(f"GVA is unsupported: H={H}, HV={HV}.")
    if K != 128 or V not in (128, 256):
        raise NotImplementedError(f"Requires K=128 and V in (128, 256), got K={K}, V={V}.")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"q must be float16 or bfloat16, got {q.dtype}.")
    if g.shape != (B, T, HV, K):
        raise ValueError(f"g must be {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if beta.shape != (B, T, HV):
        raise ValueError(f"beta must be {(B, T, HV)}, got {tuple(beta.shape)}.")

    if scale is None:
        scale = K ** -0.5

    if scale is None:
        scale = K ** -0.5

    return ChunkKDAAscendCFunction.apply(
        q, k, v, g, beta, A_log, dt_bias, scale,
        use_qk_l2norm_in_kernel, use_gate_in_kernel, safe_gate, lower_bound,
    )
