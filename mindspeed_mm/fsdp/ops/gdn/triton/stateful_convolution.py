# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Differentiable stateful causal convolution for BSND and packed TND."""

from typing import Optional

import torch
import triton
import triton.language as tl

from .convolution import (
    causal_conv1d_bwd_impl,
    causal_conv1d_fwd_impl,
    causal_conv1d_update_states,
)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["bias"] is not None,
        "HAS_GRAD_OUTPUT": lambda args: args["grad_output"] is not None,
        "HAS_RESIDUAL": lambda args: args["residual"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_SILU": lambda args: args["ACTIVATION"] is not None,
    }
)
@triton.jit
def _stateful_head_fwd_kernel(
    x,
    weight,
    bias,
    residual,
    initial_state,
    y,
    grad_output,
    cu_seqlens,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    ACTIVATION: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_GRAD_OUTPUT: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SILU: tl.constexpr,
    NUM_HEAD_TASKS: tl.int32,
):
    i_d = tl.program_id(0)
    offsets_d = i_d * BD + tl.arange(0, BD)
    mask_d = offsets_d < D

    # Persist one program per D tile and walk the tiny set of sequence heads.
    # This avoids launching one program for every (sequence, head-token) pair.
    for head_task in range(0, NUM_HEAD_TASKS):
        i_n = head_task // (W - 1)
        offset_t = head_task % (W - 1)
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            sequence_length = eos - bos
        else:
            bos = (i_n * T).to(tl.int64)
            sequence_length = T
        mask_t = offset_t < sequence_length
        accumulator = tl.zeros((BD,), dtype=tl.float32)

        for i_w in tl.static_range(0, W):
            source_t = offset_t - W + 1 + i_w
            safe_source_t = tl.maximum(source_t, 0)
            source = tl.load(
                x + (bos + safe_source_t) * D + offsets_d,
                mask=mask_t & (source_t >= 0) & mask_d,
                other=0.0,
            ).to(tl.float32)
            state_index = source_t + W
            safe_state_index = tl.minimum(
                tl.maximum(state_index, 0),
                W - 1,
            )
            source += tl.load(
                initial_state
                + i_n * D * W
                + offsets_d * W
                + safe_state_index,
                mask=(
                    mask_t
                    & (source_t < 0)
                    & (state_index >= 0)
                    & (state_index < W)
                    & mask_d
                ),
                other=0.0,
            ).to(tl.float32)
            w = tl.load(
                weight + i_w * D + offsets_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            accumulator += source * w

        if HAS_BIAS:
            accumulator += tl.load(
                bias + offsets_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        if HAS_GRAD_OUTPUT:
            gradient = tl.load(
                grad_output + (bos + offset_t) * D + offsets_d,
                mask=mask_t & mask_d,
                other=0.0,
            ).to(tl.float32)
            sigmoid = tl.sigmoid(accumulator)
            accumulator = gradient * sigmoid * (
                1.0 + accumulator * (1.0 - sigmoid)
            )
        else:
            if USE_SILU:
                accumulator *= tl.sigmoid(accumulator)
            if HAS_RESIDUAL:
                accumulator += tl.load(
                    residual + (bos + offset_t) * D + offsets_d,
                    mask=mask_t & mask_d,
                    other=0.0,
                ).to(tl.float32)
        tl.store(
            y + (bos + offset_t) * D + offsets_d,
            accumulator.to(y.dtype.element_ty, fp_downcast_rounding="rtne"),
            mask=mask_t & mask_d,
        )


@triton.heuristics(
    {"IS_VARLEN": lambda args: args["cu_seqlens"] is not None}
)
@triton.jit
def _stateful_causal_conv1d_head_bwd_kernel(
    initial_state,
    weight,
    gradient,
    d_initial_state,
    state_dw,
    cu_seqlens,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_SEQUENCES: tl.int32,
):
    i_d = tl.program_id(0)
    offsets_d = i_d * BD + tl.arange(0, BD)
    mask_d = offsets_d < D

    for i_n in range(0, NUM_SEQUENCES):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            sequence_length = eos - bos
        else:
            bos = (i_n * T).to(tl.int64)
            sequence_length = T

        for state_index in tl.static_range(0, W):
            dstate = tl.zeros((BD,), dtype=tl.float32)
            for output_t in tl.static_range(0, W - 1):
                if output_t < state_index:
                    dz_row = tl.load(
                        gradient + (bos + output_t) * D + offsets_d,
                        mask=mask_d & (output_t < sequence_length),
                        other=0.0,
                    ).to(tl.float32)
                    w = tl.load(
                        weight
                        + (state_index - 1 - output_t) * D
                        + offsets_d,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    dstate += dz_row * w
            tl.store(
                d_initial_state
                + i_n * D * W
                + offsets_d * W
                + state_index,
                dstate.to(
                    d_initial_state.dtype.element_ty,
                    fp_downcast_rounding="rtne",
                ),
                mask=mask_d,
            )

    for i_w in tl.static_range(0, W):
        dw = tl.zeros((BD,), dtype=tl.float32)
        for i_n in range(0, NUM_SEQUENCES):
            if IS_VARLEN:
                bos = tl.load(cu_seqlens + i_n).to(tl.int64)
                eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
                sequence_length = eos - bos
            else:
                bos = (i_n * T).to(tl.int64)
                sequence_length = T
            for output_t in tl.static_range(0, W - 1):
                if output_t + i_w < W - 1:
                    dz_row = tl.load(
                        gradient + (bos + output_t) * D + offsets_d,
                        mask=mask_d & (output_t < sequence_length),
                        other=0.0,
                    ).to(tl.float32)
                    state = tl.load(
                        initial_state
                        + i_n * D * W
                        + offsets_d * W
                        + output_t
                        + i_w
                        + 1,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    dw += dz_row * state
        tl.store(
            state_dw + i_w * D + offsets_d,
            dw,
            mask=mask_d,
        )


@triton.heuristics(
    {
        "HAS_INITIAL_STATE": lambda args: args["d_initial_state"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit
def _stateful_causal_conv1d_final_state_bwd_kernel(
    dx,
    d_initial_state,
    d_final_state,
    cu_seqlens,
    T,
    D: tl.constexpr,
    W: tl.constexpr,
    BD: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    NUM_SEQUENCES: tl.int32,
):
    i_d = tl.program_id(0)
    offsets_d = i_d * BD + tl.arange(0, BD)
    mask_d = offsets_d < D

    for i_n in range(0, NUM_SEQUENCES):
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int64)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
            sequence_length = eos - bos
        else:
            bos = (i_n * T).to(tl.int64)
            sequence_length = T

        for i_w in tl.static_range(0, W):
            source_t = sequence_length - W + i_w
            gradient = tl.load(
                d_final_state + i_n * D * W + offsets_d * W + i_w,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            if source_t >= 0:
                current = tl.load(
                    dx + (bos + source_t) * D + offsets_d,
                    mask=mask_d,
                    other=0.0,
                ).to(tl.float32)
                tl.store(
                    dx + (bos + source_t) * D + offsets_d,
                    (current + gradient).to(
                        dx.dtype.element_ty,
                        fp_downcast_rounding="rtne",
                    ),
                    mask=mask_d,
                )
            else:
                if HAS_INITIAL_STATE:
                    state_index = source_t + W
                    current = tl.load(
                        d_initial_state
                        + i_n * D * W
                        + offsets_d * W
                        + state_index,
                        mask=mask_d,
                        other=0.0,
                    ).to(tl.float32)
                    tl.store(
                        d_initial_state
                        + i_n * D * W
                        + offsets_d * W
                        + state_index,
                        (current + gradient).to(
                            d_initial_state.dtype.element_ty,
                            fp_downcast_rounding="rtne",
                        ),
                        mask=mask_d,
                    )


def _native_stateful_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
):
    """Apply initial-state corrections to the first W-1 outputs."""
    if activation not in (None, "silu", "swish"):
        raise ValueError(f"unsupported activation: {activation}")
    _, time, dimension = x.shape
    width = weight.shape[0]
    y, _ = causal_conv1d_fwd_impl(
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        cu_seqlens=cu_seqlens,
    )
    if width <= 1:
        return y
    block_d = 256
    num_d_blocks = triton.cdiv(dimension, block_d)
    num_sequences = initial_state.shape[0]
    _stateful_head_fwd_kernel[(num_d_blocks,)](
        x=x,
        weight=weight,
        bias=bias,
        residual=residual,
        initial_state=initial_state,
        y=y,
        grad_output=None,
        cu_seqlens=cu_seqlens,
        T=time,
        D=dimension,
        W=width,
        BD=block_d,
        ACTIVATION=activation,
        NUM_HEAD_TASKS=num_sequences * (width - 1),
    )
    return y


def _native_stateful_gradient(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    grad_output: torch.Tensor,
    activation: Optional[str],
    cu_seqlens: Optional[torch.Tensor],
):
    """Compute the activation gradient with initial-state corrections."""
    if activation is None:
        return grad_output
    _, time, dimension = x.shape
    width = weight.shape[0]
    gradient, _ = causal_conv1d_fwd_impl(
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=None,
        output_final_state=False,
        activation=activation,
        cu_seqlens=cu_seqlens,
        grad_output=grad_output,
    )
    if width <= 1:
        return gradient
    block_d = 256
    num_d_blocks = triton.cdiv(dimension, block_d)
    num_sequences = initial_state.shape[0]
    _stateful_head_fwd_kernel[(num_d_blocks,)](
        x=x,
        weight=weight,
        bias=bias,
        residual=None,
        initial_state=initial_state,
        y=gradient,
        grad_output=grad_output,
        cu_seqlens=cu_seqlens,
        T=time,
        D=dimension,
        W=width,
        BD=block_d,
        ACTIVATION=activation,
        NUM_HEAD_TASKS=num_sequences * (width - 1),
    )
    return gradient


def _native_stateful_head_backward(
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    gradient: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
):
    """Compute gradients contributed by the differentiable initial state."""
    _, time, dimension = gradient.shape
    width = weight.shape[0]
    num_sequences = (
        initial_state.shape[0]
        if cu_seqlens is None
        else len(cu_seqlens) - 1
    )
    block_d = 256
    num_d_blocks = triton.cdiv(dimension, block_d)
    d_initial_state = torch.empty_like(initial_state)
    state_dw = torch.empty(
        width,
        dimension,
        dtype=torch.float32,
        device=weight.device,
    )
    _stateful_causal_conv1d_head_bwd_kernel[(num_d_blocks,)](
        initial_state=initial_state,
        weight=weight,
        gradient=gradient,
        d_initial_state=d_initial_state,
        state_dw=state_dw,
        cu_seqlens=cu_seqlens,
        T=time,
        D=dimension,
        W=width,
        BD=block_d,
        NUM_SEQUENCES=num_sequences,
    )
    return d_initial_state, state_dw


def _add_final_state_gradient(
    dx: torch.Tensor,
    d_initial_state: Optional[torch.Tensor],
    d_final_state: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
    width: int,
):
    """Accumulate final-state gradients into input and initial-state gradients."""
    _, time, dimension = dx.shape
    num_sequences = d_final_state.shape[0]
    block_d = 256
    num_d_blocks = triton.cdiv(dimension, block_d)
    _stateful_causal_conv1d_final_state_bwd_kernel[(num_d_blocks,)](
        dx=dx,
        d_initial_state=d_initial_state,
        d_final_state=d_final_state,
        cu_seqlens=cu_seqlens,
        T=time,
        D=dimension,
        W=width,
        BD=block_d,
        NUM_SEQUENCES=num_sequences,
    )


class NativeStatefulCausalConv1dFunction(torch.autograd.Function):
    """Autograd entry point shared by BSND and packed TND."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        initial_state: Optional[torch.Tensor],
        activation: Optional[str],
        cu_seqlens: Optional[torch.Tensor],
        output_final_state: bool,
    ):
        ctx.set_materialize_grads(False)
        if x.ndim != 3:
            raise ValueError("x must have shape [B, T, D]")
        if weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
            raise ValueError("weight must have shape [W, D]")
        num_sequences = (
            x.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1
        )
        if initial_state is not None:
            expected_state = (num_sequences, x.shape[-1], weight.shape[0])
            if tuple(initial_state.shape) != expected_state:
                raise ValueError(
                    f"initial_state shape {tuple(initial_state.shape)} "
                    f"must be {expected_state}"
                )

        x = x.contiguous()
        weight = weight.contiguous()
        initial_state = (
            initial_state.contiguous() if initial_state is not None else None
        )
        bias = bias.contiguous() if bias is not None else None
        residual = residual.contiguous() if residual is not None else None
        cu_seqlens = (
            cu_seqlens.contiguous() if cu_seqlens is not None else None
        )
        if initial_state is None:
            y, _ = causal_conv1d_fwd_impl(
                x=x,
                weight=weight,
                bias=bias,
                residual=residual,
                initial_state=None,
                output_final_state=False,
                activation=activation,
                cu_seqlens=cu_seqlens,
            )
        else:
            y = _native_stateful_forward(
                x,
                weight,
                bias,
                residual,
                initial_state,
                activation,
                cu_seqlens,
            )
        final_state = (
            causal_conv1d_update_states(
                x=x,
                state_len=weight.shape[0],
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )
            if output_final_state
            else None
        )

        placeholder = x.new_empty(0)
        ctx.save_for_backward(
            x,
            weight,
            bias if bias is not None else placeholder,
            initial_state if initial_state is not None else placeholder,
            cu_seqlens if cu_seqlens is not None else placeholder,
        )
        ctx.has_bias = bias is not None
        ctx.has_initial_state = initial_state is not None
        ctx.has_residual = residual is not None
        ctx.has_cu_seqlens = cu_seqlens is not None
        ctx.activation = activation
        return y, final_state

    @staticmethod
    def backward(
        ctx,
        dy: torch.Tensor,
        d_final_state: Optional[torch.Tensor] = None,
    ):
        x, weight, bias, initial_state, cu_seqlens = ctx.saved_tensors
        bias = bias if ctx.has_bias else None
        initial_state = initial_state if ctx.has_initial_state else None
        cu_seqlens = cu_seqlens if ctx.has_cu_seqlens else None
        dy = torch.zeros_like(x) if dy is None else dy.contiguous()

        if ctx.activation is None:
            gradient = dy
        elif initial_state is None:
            gradient, _ = causal_conv1d_fwd_impl(
                x=x,
                weight=weight,
                bias=bias,
                residual=None,
                initial_state=None,
                output_final_state=False,
                activation=ctx.activation,
                cu_seqlens=cu_seqlens,
                grad_output=dy,
            )
        else:
            gradient = _native_stateful_gradient(
                x,
                weight,
                bias,
                initial_state,
                dy,
                ctx.activation,
                cu_seqlens,
            )
        dx, dw, db, _, _ = causal_conv1d_bwd_impl(
            x=x,
            dy=gradient,
            dht=None,
            weight=weight,
            bias=bias,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
        )
        if initial_state is not None:
            d_initial_state, state_dw = _native_stateful_head_backward(
                initial_state,
                weight,
                gradient,
                cu_seqlens,
            )
            dw = (dw.float() + state_dw).to(weight)
        else:
            d_initial_state = None

        if d_final_state is not None:
            _add_final_state_gradient(
                dx,
                d_initial_state,
                d_final_state.contiguous(),
                cu_seqlens,
                weight.shape[0],
            )
        d_residual = dy if ctx.has_residual else None
        return (
            dx,
            dw,
            db,
            d_residual,
            d_initial_state,
            None,
            None,
            None,
        )


def native_stateful_causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
):
    """Run native stateful causal convolution with optional final-state output."""
    return NativeStatefulCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )
