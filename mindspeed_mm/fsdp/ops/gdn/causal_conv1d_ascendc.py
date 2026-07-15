from typing import Dict, Optional

import torch
import torch.nn.functional as F
import torch_npu
import fla_npu

from fla_npu.ops.ascendc import (
    causal_conv1d as ascendc_causal_conv1d,
    causal_conv1d_bwd as ascendc_causal_conv1d_bwd,
)


def _prepare_conv_states(
    x: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    *,
    num_sequences: int,
    width: int,
    dim: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    state_len = width - 1
    if initial_state is None:
        conv_states = torch.zeros(num_sequences, state_len, dim, dtype=x.dtype, device=x.device)
        return conv_states, None

    if initial_state.ndim != 3 or initial_state.shape[0] != num_sequences:
        raise ValueError(
            "initial_state must be rank-3 and match the sequence count, "
            f"got shape={tuple(initial_state.shape)} and num_sequences={num_sequences}."
        )

    if initial_state.shape[1] == dim and initial_state.shape[2] >= width:
        state_for_bwd = initial_state.transpose(1, 2).contiguous()
    elif initial_state.shape[2] == dim and initial_state.shape[1] >= width:
        state_for_bwd = initial_state.contiguous()
    else:
        raise ValueError(
            "initial_state must use [N, D, W] or [N, W, D] layout with W >= kernel width; "
            f"got shape={tuple(initial_state.shape)}, dim={dim}, width={width}."
        )

    conv_states = state_for_bwd[:, -state_len:, :].contiguous()
    return conv_states, state_for_bwd


def _activation_mode(activation: Optional[str]) -> int:
    if activation is None or activation == "":
        return 0
    if activation in ("silu", "swish"):
        return 1
    raise ValueError(f"Unsupported causal_conv1d activation: {activation}")


def _as_int_list(value: Optional[list[int] | torch.Tensor]) -> Optional[list[int]]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return [int(x) for x in value.detach().cpu().flatten().tolist()]
    return [int(x) for x in value]


def _causal_conv1d_final_state(
    x: torch.Tensor,
    *,
    width: int,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
) -> torch.Tensor:
    dim = x.shape[-1]
    cu_list = _as_int_list(cu_seqlens)
    if cu_list is None:
        sequences = [x[i] for i in range(x.shape[0])]
    else:
        flat_x = x.reshape(-1, dim) if x.ndim == 3 else x
        sequences = [flat_x[cu_list[i] : cu_list[i + 1]] for i in range(len(cu_list) - 1)]

    states = []
    for idx, seq in enumerate(sequences):
        prev = None
        if initial_state is not None:
            prev = initial_state[idx]
            if prev.shape[0] != dim:
                prev = prev.transpose(0, 1).contiguous()
        hist = seq.transpose(0, 1).contiguous()
        if prev is not None:
            hist = torch.cat([prev[:, -width:], hist], dim=-1)
        if hist.shape[-1] < width:
            hist = F.pad(hist, (width - hist.shape[-1], 0))
        states.append(hist[:, -width:])
    return torch.stack(states, dim=0)


def _flat_to_head_layout(x: torch.Tensor, head_num: int, *, is_varlen: bool) -> torch.Tensor:
    if head_num <= 0:
        return x
    if x.shape[-1] % head_num != 0:
        raise ValueError(f"last dimension must be divisible by head_num, got shape={tuple(x.shape)}, head_num={head_num}.")
    head_dim = x.shape[-1] // head_num
    if is_varlen:
        flat_x = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x
        return flat_x.reshape(flat_x.shape[0], head_num, head_dim).transpose(0, 1).contiguous()
    return x.reshape(x.shape[0], x.shape[1], head_num, head_dim).transpose(1, 2).contiguous()


def _flatten_varlen_x(
    x: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor],
) -> tuple[torch.Tensor, Optional[list[int]], bool]:
    if cu_seqlens is None:
        return x, None, False

    cu_list = _as_int_list(cu_seqlens)
    if x.ndim == 3:
        if x.shape[0] != 1:
            raise ValueError("causal_conv1d varlen path expects x.shape[0] == 1 for [1, T, D] input.")
        return x.reshape(x.shape[1], x.shape[2]).contiguous(), cu_list, True
    if x.ndim == 2:
        return x.contiguous(), cu_list, True
    raise ValueError(f"causal_conv1d varlen path expects rank-2 or rank-3 input, got shape={tuple(x.shape)}.")


def _silu_backward(grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sigmoid = torch.sigmoid(x)
    return grad * sigmoid * (1.0 + x * (1.0 - sigmoid))


def _head_to_flat_layout(x: torch.Tensor, *, is_varlen: bool, batch_size: int) -> torch.Tensor:
    if is_varlen:
        x_head = x.squeeze(0) if x.ndim == 4 and x.shape[0] == 1 else x
        return x_head.transpose(0, 1).reshape(-1, x_head.shape[0] * x_head.shape[-1]).contiguous()
    return x.transpose(1, 2).reshape(batch_size, x.shape[2], x.shape[1] * x.shape[-1]).contiguous()


class AscendCCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        H: int,
        bias: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        activation: Optional[str] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ):
        activation_mode = _activation_mode(activation)
        op_weight = weight.transpose(-1, -2).contiguous()
        width, dim = op_weight.shape
        op_x, query_start_loc, is_varlen = _flatten_varlen_x(x, cu_seqlens)
        num_sequences = len(query_start_loc) - 1 if query_start_loc is not None else int(x.shape[0])
        conv_states, initial_state_bwd = _prepare_conv_states(
            x,
            initial_state,
            num_sequences=num_sequences,
            width=width,
            dim=dim,
        )
        initial_state_mode = [1] * num_sequences if initial_state is not None else None

        preactivation = ascendc_causal_conv1d(
            op_x,
            op_weight,
            bias,
            conv_states,
            query_start_loc=query_start_loc,
            initial_state_mode=initial_state_mode,
            activation_mode=0,
            pad_slot_id=-1,
            run_mode=0,
            head_num=H,
        )
        if is_varlen:
            preactivation = preactivation.unsqueeze(0)
        if residual is not None:
            preactivation = preactivation + _flat_to_head_layout(residual, H, is_varlen=is_varlen)

        y = F.silu(preactivation) if activation_mode != 0 else preactivation
        final_state = None
        if output_final_state:
            final_state = _causal_conv1d_final_state(
                x,
                width=width,
                initial_state=initial_state,
                cu_seqlens=cu_seqlens,
            )

        ctx.save_for_backward(x, op_weight, bias, residual, initial_state_bwd, preactivation)
        ctx.activation_mode = activation_mode
        ctx.query_start_loc = query_start_loc
        ctx.is_varlen = is_varlen
        ctx.head_num = H
        ctx.batch_size = x.shape[0] if x.ndim == 3 else 1
        ctx.had_bias = bias is not None
        ctx.had_residual = residual is not None
        ctx.had_initial_state = initial_state is not None
        return y, final_state

    @staticmethod
    def backward(ctx, dy: torch.Tensor, dht: Optional[torch.Tensor] = None):
        x, op_weight, bias, residual, initial_state_bwd, preactivation = ctx.saved_tensors
        op_x = x.reshape(-1, x.shape[-1]).contiguous() if ctx.is_varlen and x.ndim == 3 else x.contiguous()
        op_dy = dy.squeeze(0).contiguous() if ctx.is_varlen and dy.ndim == 4 else dy.contiguous()
        op_y = preactivation.squeeze(0).contiguous() if ctx.is_varlen and preactivation.ndim == 4 else preactivation.contiguous()
        dht_bwd = None
        if dht is not None:
            dht_bwd = dht.transpose(1, 2).contiguous() if dht.ndim == 3 and dht.shape[1] == x.shape[-1] else dht

        dx, dw, db, dh0 = ascendc_causal_conv1d_bwd(
            x=op_x,
            y=op_y if ctx.activation_mode != 0 else None,
            weight=op_weight,
            dy=op_dy,
            initial_state=initial_state_bwd if ctx.had_initial_state else None,
            dht=dht_bwd,
            query_start_loc=ctx.query_start_loc,
            activation=ctx.activation_mode,
            input_layout="NTD" if ctx.is_varlen else "BNSD",
        )

        dx = dx.reshape_as(x)
        dw = dw.transpose(0, 1).contiguous()
        db = db if ctx.had_bias else None
        dr = None
        if ctx.had_residual:
            dr_head = _silu_backward(dy, preactivation) if ctx.activation_mode != 0 else dy
            dr = _head_to_flat_layout(dr_head, is_varlen=ctx.is_varlen, batch_size=ctx.batch_size)
        dh0 = dh0.transpose(1, 2).contiguous() if ctx.had_initial_state else None
        return dx, dw, None, db, dr, dh0, None, None, None


def causal_conv1d_ascendc(
    x: torch.Tensor,
    weight: torch.Tensor,
    H: int,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    return AscendCCausalConv1dFunction.apply(
        x,
        weight,
        H,
        bias,
        residual,
        initial_state,
        activation,
        cu_seqlens,
        output_final_state,
    )
