# Copyright © 2026 Huawei Technologies Co., Ltd.
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# This file contains code copied and/or modified from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li


import warnings

import torch

from triton_ascend_kernels.attention.fla.kda.chunk import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)


from fla_npu.ops.ascendc import chunk_kda_fwd as ascendc_chunk_kda_fwd
from fla_npu.ops.ascendc import chunk_kda_bwd as ascendc_chunk_kda_bwd


# ---------------------------------------------------------------------------
# Small host-side helpers kept from the previous integration.
# ---------------------------------------------------------------------------

def fused_beta_sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Chain rule using the effective beta already produced by fused Prepare.

    Here x is scale * sigmoid(beta_raw), not the raw logits.
    """
    return dy.float() * x.float() * (1.0 - x.float() / scale)


def _host_int_tuple(value):
    """Convert tensor metadata to one flat Host tuple with at most one D2H."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "reshape"):
        value = value.reshape(-1)
    if hasattr(value, "tolist"):
        value = value.tolist()
    return tuple(int(item) for item in value)


# ---------------------------------------------------------------------------
# Layout converters between the model layout (sequence-major) and the
# head-major layout consumed by the AscendC V2 kernels.
# ---------------------------------------------------------------------------

def _bsnd_to_bnsd(tensor):
    """[B, T, H, K] -> [B, H, T, K]"""

    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected one rank-4 BSND tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bnsd_to_bsnd(tensor):
    """[B, H, T, K] -> [B, T, H, K]"""

    if tensor is None:
        return None
    if tensor.dim() != 4:
        raise RuntimeError(f"Expected one rank-4 BNSD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1, 3).contiguous()


def _bsh_to_bhs(tensor):
    """[B, T, H] -> [B, H, T]"""

    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BSH tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bhs_to_bsh(tensor):
    """[B, H, T] -> [B, T, H]"""

    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 BHS tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(0, 2, 1).contiguous()


def _bsnd_to_ntd(tensor):
    """[1, T, H, K] -> [H, T, K] (packed varlen)"""

    if tensor is None:
        return None
    if tensor.dim() != 4 or tensor.shape[0] != 1:
        raise RuntimeError(
            "Varlen KDA tensors must use packed BSND with B=1, "
            f"but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).permute(1, 0, 2).contiguous()


def _bsh_to_nt(tensor):
    """[1, T, H] -> [H, T] (packed varlen)"""

    if tensor is None:
        return None
    if tensor.dim() != 3 or tensor.shape[0] != 1:
        raise RuntimeError(
            "Varlen KDA tensors must use packed BSH with B=1, "
            f"but received shape {tuple(tensor.shape)}."
        )
    return tensor.squeeze(0).transpose(0, 1).contiguous()


def _ntd_to_bsnd(tensor):
    """[H, T, K] -> [1, T, H, K] (packed varlen)"""

    if tensor is None:
        return None
    if tensor.dim() != 3:
        raise RuntimeError(f"Expected one rank-3 NTD tensor, got shape {tuple(tensor.shape)}.")
    return tensor.permute(1, 0, 2).unsqueeze(0).contiguous()


def _nt_to_bsh(tensor):
    """[H, T] -> [1, T, H] (packed varlen)"""

    if tensor is None:
        return None
    if tensor.dim() != 2:
        raise RuntimeError(f"Expected one rank-2 NT tensor, got shape {tuple(tensor.shape)}.")
    return tensor.transpose(0, 1).unsqueeze(0).contiguous()


def _packed_varlen_tensor(tensor):
    """Drop the leading singleton batch dim from packed varlen intermediates.

    The V2 kernels consume NTD-style tensors without a batch dim; some builds
    still return intermediates with a leading ``B=1``. Normalize defensively.
    """

    if tensor is not None and tensor.dim() >= 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


# ---------------------------------------------------------------------------
# Pre-flight validation.
# ---------------------------------------------------------------------------

def _require_ascendc_bwd(
    q,
    Aqk,
    Akk,
    *,
    disable_recompute,
    initial_state,
    output_final_state,
    state_v_first,
):
    """Fail fast when the AscendC backward cannot run.

    The Triton fallback has been removed, so any unsupported configuration
    raises here with an explicit error instead of silently degrading.
    """

    def _reject(reason):
        raise RuntimeError(
            "AscendC KDA backward does not support this configuration: "
            f"{reason} The Triton fallback has been removed."
        )

    if ascendc_chunk_kda_bwd is None:
        raise RuntimeError(
            "AscendC KDA backward kernel is unavailable "
            "(fla_npu.ops.ascendc.chunk_kda_bwd is None), "
            "and the Triton fallback has been removed."
        )

    if initial_state is not None:
        _reject("non-None `initial_state`.")
    if output_final_state:
        _reject("`output_final_state=True`.")
    if state_v_first:
        _reject("`state_v_first=True`.")

    # Aqk/Akk are required by the backward in both recompute modes.
    expected_last = 64
    for name, tensor in (("Aqk", Aqk), ("Akk", Akk)):
        if (
            tensor is None
            or tensor.dtype != q.dtype
            or tensor.shape[-1] != expected_last
            or not tensor.is_contiguous()
        ):
            _reject(
                f"malformed forward intermediate `{name}` "
                f"(shape={None if tensor is None else tuple(tensor.shape)}, "
                f"dtype={None if tensor is None else tensor.dtype})."
            )

    # The recompute mode recomputes gk/w/qg/kg/v_new/h inside the backward,
    # so their forward outputs are allowed (and expected) to be None.
    _ = disable_recompute


def _validate_recompute_support(q, cu_seqlens_host):
    """Check head-count and packed-batch constraints; sequence tails are allowed."""

    batch, _, heads, _ = q.shape
    if heads > 256 or heads % 8 != 0:
        raise ValueError(
            "AscendC KDA backward recompute (disable_recompute=False) requires "
            f"H <= 256 and H % 8 == 0, got H={heads}."
        )
    if cu_seqlens_host is not None and batch != 1:
        raise ValueError(
            "Packed varlen KDA tensors must use batch size 1, "
            f"but received shape {tuple(q.shape)}."
        )


class ChunkKDAFunction(torch.autograd.Function):
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
        initial_state: torch.Tensor,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        use_gate_in_kernel: bool = False,
        use_beta_sigmoid_in_kernel: bool = False,
        allow_neg_eigval: bool = False,
        state_v_first: bool = False,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        safe_gate: bool = False,
        lower_bound: float | None = None,
        chunk_size: int = 64,
        disable_recompute: bool = False,
        return_intermediate_states: bool = False,
        qkv_layout: str = "BSND",
        transpose_state_layout: bool = False,
    ):
        if chunk_size != 64:
            raise ValueError(
                f"AscendC KDA only supports `chunk_size=64`, got {chunk_size}. "
                "The Triton fallback for other chunk sizes has been removed."
            )
        if qkv_layout not in ("BSND", "BNSD"):
            raise ValueError(f"Unsupported qkv_layout: '{qkv_layout}'. Expected 'BSND' or 'BNSD'.")

        # The AscendC Python entry consumes Host metadata. Prefer the CPU
        # copy already prepared by the model to avoid another D2H.
        cu_seqlens_host = _host_int_tuple(
            cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens
        )
        packed = cu_seqlens_host is not None

        if not disable_recompute:
            _validate_recompute_support(q, cu_seqlens_host)

        # Convert model inputs to the head-major layout consumed by the V2
        # kernels. The Q/K l2norm and the beta sigmoid are fused into the
        # kernels, so the raw model tensors are passed straight through.
        # With qkv_layout="BNSD", q/k/v arrive head-major [B, H, T, K]
        # directly from the AscendC short conv, so their conversion is skipped;
        # g/beta still come from projections in the model layout.
        if packed:
            if qkv_layout == "BNSD":
                # [1, H, T, K] -> [H, T, K] (packed NTD), a view of the
                # contiguous conv output.
                q = q.squeeze(0)
                k = k.squeeze(0)
                v = v.squeeze(0)
            else:
                q = _bsnd_to_ntd(q)
                k = _bsnd_to_ntd(k)
                v = _bsnd_to_ntd(v)
            g = _bsnd_to_ntd(g)
            beta = _bsh_to_nt(beta)
        else:
            if qkv_layout == "BSND":
                q = _bsnd_to_bnsd(q)
                k = _bsnd_to_bnsd(k)
                v = _bsnd_to_bnsd(v)
            g = _bsnd_to_bnsd(g)
            beta = _bsh_to_bhs(beta)

        beta_dtype = beta.dtype
        if not return_intermediate_states:
            if initial_state is not None or output_final_state:
                raise ValueError("Optimized training does not support initial/final states.")
            if not safe_gate or not use_gate_in_kernel:
                raise ValueError("Optimized training requires safe_gate/use_gate_in_kernel=True.")
            if q.shape != k.shape or q.shape != v.shape or q.shape != g.shape or q.shape[-1] != 128:
                raise ValueError("Optimized training requires equal Q/K/V heads and K=V=128.")
            if any(x.dtype != torch.bfloat16 for x in (q, k, v)):
                raise ValueError("Optimized training requires BF16 Q/K/V.")
            cu = cu_seqlens_host
            if cu is not None and (len(cu) < 2 or cu[0] != 0 or cu[-1] != q.shape[-2]
                                or any(b <= a for a, b in zip(cu, cu[1:]))):
                raise ValueError("Packed boundaries must be strictly increasing from 0 to T.")

        use_l2norm = use_qk_l2norm_in_kernel and not return_intermediate_states
        use_beta_sigmoid = use_beta_sigmoid_in_kernel and not return_intermediate_states
        q_hat = torch.empty_like(q) if use_l2norm else q
        k_hat = torch.empty_like(k) if use_l2norm else k
        q_rstd = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device) if use_l2norm else None
        k_rstd = torch.empty(k.shape[:-1], dtype=torch.float32, device=k.device) if use_l2norm else None
        beta_eff = torch.empty_like(beta, dtype=torch.float32) if use_beta_sigmoid else (beta if return_intermediate_states else beta.float())
        (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, _) = ascendc_chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            layout="NTD" if packed else "BNSD",
            chunk_size=64,
            cu_seqlens=cu_seqlens_host,
            initial_state=initial_state,
            output_final_state=bool(output_final_state),
            safe_gate=bool(safe_gate),
            lower_bound=lower_bound,
            use_gate_in_kernel=bool(use_gate_in_kernel),
            A_log=A_log if use_gate_in_kernel else None,
            dt_bias=(dt_bias.reshape(-1) if not return_intermediate_states else dt_bias)
            if use_gate_in_kernel and dt_bias is not None else None,
            disable_recompute=bool(disable_recompute),
            return_intermediate_states=bool(return_intermediate_states),
            state_v_first=False,
            use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
            use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
            allow_neg_eigval=bool(allow_neg_eigval),
            use_exp2=True,
            q_hat_out=q_hat if use_l2norm else None,
            k_hat_out=k_hat if use_l2norm else None,
            q_rstd_out=q_rstd,
            k_rstd_out=k_rstd,
            beta_eff_out=beta_eff if use_beta_sigmoid else None,
        )
        if packed:
            o = o.unsqueeze(0)
        if not return_intermediate_states:
            q, k, beta = q_hat, k_hat, beta_eff

        if return_intermediate_states:
            assert torch.is_inference_mode_enabled(), "return_intermediate_states is only allowed in inference mode"
            assert disable_recompute is False, "return_intermediate_states must be used with disable_recompute=False"
            return o.type_as(q), final_state, h

        # The fused AscendC backward is the only backward path: any
        # unsupported configuration raises inside this check.
        _require_ascendc_bwd(
            q,
            Aqk,
            Akk,
            disable_recompute=disable_recompute,
            initial_state=initial_state,
            output_final_state=output_final_state,
            state_v_first=state_v_first,
        )

        ctx.save_for_backward(
            q, k, v, g, beta, A_log, dt_bias,
            Aqk, Akk, gk, w, qg, kg, v_new, h, q_rstd, k_rstd,
        )
        ctx.beta_dtype = beta_dtype
        ctx.chunk_size = 64
        ctx.safe_gate = safe_gate
        ctx.scale = scale
        ctx.lower_bound = lower_bound
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.allow_neg_eigval = allow_neg_eigval
        ctx.disable_recompute = disable_recompute
        ctx.cu_seqlens_host = cu_seqlens_host
        ctx.qkv_layout = qkv_layout
        return o.type_as(q), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        (q, k, v, g, beta, A_log, dt_bias,
        Aqk, Akk, gk, w, qg, kg, v_new, h, q_rstd, k_rstd) = (
            ctx.saved_tensors
        )

        # The fused AscendC backward is the only backward path.
        if dht is not None:
            raise RuntimeError("AscendC KDA backward does not support final-state gradients.")

        is_varlen = ctx.cu_seqlens_host is not None
        if is_varlen:
            # Keep the model's [1,T,H,K] output contract.
            do = _bsnd_to_ntd(do)
        else:
            # grad_out is [B, T, H, K]; the kernel consumes [B, H, T, K].
            do = do.transpose(1, 2).contiguous()

        gate_A_log = A_log if ctx.use_gate_in_kernel else None
        gate_dt_bias = dt_bias if ctx.use_gate_in_kernel else None

        dq, dk, dv, db, dg, dh0, dA, dbias = ascendc_chunk_kda_bwd(
            q,
            k,
            v,
            beta,
            gk,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
            do,
            float(ctx.scale),
            raw_g=g,
            A_log=gate_A_log,
            dt_bias=gate_dt_bias,
            cu_seqlens=ctx.cu_seqlens_host,
            chunk_size=64,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            disable_recompute=ctx.disable_recompute,
            use_exp2=True,
            state_v_first=False,
            implementation="optimized",
            q_rstd=q_rstd,
            k_rstd=k_rstd,
        )

        # beta is the effective value emitted by Prepare. Reuse it for the
        # sigmoid chain rule without running sigmoid forward again.
        if ctx.use_beta_sigmoid_in_kernel:
            db = fused_beta_sigmoid_bwd(beta, db, scale=2.0 if ctx.allow_neg_eigval else 1.0)

        if is_varlen:
            if ctx.qkv_layout == "BNSD":
                # Restore the [1, H, T, K] shape of the head-major forward inputs.
                dq, dk, dv = (t.unsqueeze(0) for t in (dq, dk, dv))
            else:
                dq, dk, dv = map(_ntd_to_bsnd, (dq, dk, dv))
            db, dg = _nt_to_bsh(db), _ntd_to_bsnd(dg)
        else:
            if ctx.qkv_layout == "BSND":
                dq, dk, dv = map(_bnsd_to_bsnd, (dq, dk, dv))
            db, dg = _bhs_to_bsh(db), _bnsd_to_bsnd(dg)

        # The kernels fuse the Q/K l2norm in both directions, so dq/dk are
        # already w.r.t. the raw model inputs; db was chained back to the raw
        # beta logits above.
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(ctx.beta_dtype), dA, dbias, None, dh0,
                None, None, None, None, None, None, None, None, None, None, None, None, None, None)


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
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    qkv_layout: str = "BSND",
    **kwargs,
):
    if 'transpose_state_layout' in kwargs:
        if state_v_first:
            raise ValueError("Cannot pass both `state_v_first` and the deprecated `transpose_state_layout`.")
        warnings.warn(
            "`transpose_state_layout` is deprecated and renamed to `state_v_first`.",
            DeprecationWarning,
            stacklevel=2,
        )
        state_v_first = kwargs.pop('transpose_state_layout')

    if state_v_first:
        raise ValueError("Optimized KDA requires state_v_first=False.")

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if initial_state is not None:
        assert initial_state.dtype == torch.float32, "initial_state must be in float32."

    A_log, dt_bias = None, None
    if use_gate_in_kernel:
        assert "A_log" in kwargs, "A_log must be provided when use_gate_in_kernel=True."
        A_log, dt_bias = kwargs["A_log"], kwargs.get("dt_bias")
        if A_log.dtype != torch.float32:
            A_log = A_log.float()
        if dt_bias is not None and dt_bias.dtype != torch.float32:
            dt_bias = dt_bias.float()

    chunk_size = kwargs.pop("chunk_size", 64)
    if chunk_size != 64:
        raise ValueError(
            f"`chunk_size` must be 64 for the AscendC KDA kernel, got {chunk_size}. "
            "The Triton fallback for other chunk sizes has been removed."
        )

    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError("`lower_bound` must be specified when `safe_gate=True` and `use_gate_in_kernel=True`.")
        if not (-5 <= lower_bound < 0):
            raise ValueError(f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}.")

    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")

    if qkv_layout not in ("BSND", "BNSD"):
        raise ValueError(f"Unsupported qkv_layout: '{qkv_layout}'. Expected 'BSND' or 'BNSD'.")

    # Validate head dimensions for GVA
    if qkv_layout == "BNSD":
        # q/k/v arrive head-major [B, H, T, K] from the AscendC short conv;
        # g/beta stay in the model layout [B, T, HV, K] / [B, T, HV].
        B, H, T, K = q.shape
        HV = v.shape[1]
        for name, tensor in (("q", q), ("k", k), ("v", v)):
            if not tensor.is_contiguous():
                raise ValueError(f"head-major `{name}` must be contiguous, got shape {tuple(tensor.shape)}.")
    else:
        B, T, H, K, HV = *q.shape, v.shape[2]
    assert q.shape == k.shape, f"q and k must have the same shape, got q={q.shape} vs k={k.shape}"
    assert K <= 256, f"Currently we only support key headdim <=256 for KDA, got {K}."
    assert HV % H == 0, (
        f"For GVA, num_v_heads (HV={HV}) must be evenly divisible by num_qk_heads (H={H}), "
        f"but got HV % H = {HV % H}"
    )
    assert g.shape == (B, T, HV, K), f"g must have shape [B, T, HV, K]={[B, T, HV, K]}, got {list(g.shape)}"
    assert beta.shape == (B, T, HV), f"beta must have shape [B, T, HV]={[B, T, HV]}, got {list(beta.shape)}"

    if scale is None:
        scale = K ** -0.5
    return ChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        scale,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        state_v_first,
        cu_seqlens,
        cu_seqlens_cpu,
        safe_gate,
        lower_bound,
        chunk_size,
        disable_recompute,
        return_intermediate_states,
        qkv_layout,
    )
