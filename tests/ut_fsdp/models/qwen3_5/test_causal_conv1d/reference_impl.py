# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Pure-PyTorch reference implementations for causal_conv1d kernels.

These functions intentionally run on whichever device the inputs live on
(CPU is the typical caller); they are used as numerical golden references
for the triton-on-NPU implementations under test.
"""

import torch
import torch.nn.functional as F


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor = None,
    residual: torch.Tensor = None,
    initial_state: torch.Tensor = None,
    activation: str = None,
):
    """Reference causal conv1d.

    Args:
        x:             [B, T, D]
        weight:        [W, D]
        bias:          [D] or None
        initial_state: [B, D, W] or None.
            initial_state[b, d, k] represents the value at step (k - W) before
            the sequence start:
                k=0    -> furthest (W steps before, not actually used)
                k=1    -> position -(W-1) (earliest step that contributes)
                k=W-1  -> position -1 (most recent step)
    """
    _, T, _ = x.shape
    W = weight.shape[0]

    if initial_state is not None:
        # permute [B, D, W] -> [B, W, D], take [1:] to get positions -(W-1)..-1 (W-1 steps)
        pad = initial_state.permute(0, 2, 1)[:, 1:, :].float()  # [B, W-1, D]
        x_padded = torch.cat([pad, x.float()], dim=1)  # [B, W-1+T, D]
    else:
        x_padded = torch.nn.functional.pad(x.float(), (0, 0, W - 1, 0), value=0.0)

    y = torch.zeros_like(x, dtype=torch.float32)
    for w in range(W):
        y += x_padded[:, w: w + T, :] * weight[w].float()

    if bias is not None:
        y = y + bias.float()
    if activation in ("swish", "silu"):
        y = y * torch.sigmoid(y)
    if residual is not None:
        y = y + residual.float()
    return y


def causal_conv1d_ref_no_state(x, weight, bias=None, residual=None, activation=None):
    """Simpler reference impl without initial_state support."""
    _, T, _ = x.shape
    W = weight.shape[0]

    x_padded = torch.nn.functional.pad(x.float(), (0, 0, W - 1, 0), value=0.0)
    y = torch.zeros_like(x, dtype=torch.float32)
    for w in range(W):
        y += x_padded[:, w: w + T, :] * weight[w].float()

    if bias is not None:
        y = y + bias.float()
    if activation in ("swish", "silu"):
        y = y * torch.sigmoid(y)
    if residual is not None:
        y = y + residual.float()
    return y


def causal_conv1d_update_states_ref(x, state_len, initial_state=None):
    """Reference for causal_conv1d_update_states.

    Args:
        x:             [B, T, D]
        state_len:     W
        initial_state: [B, D, W] or None
    Returns:
        final_state [B, D, W]
    """
    B, T, D = x.shape
    W = state_len
    final_state = torch.zeros(B, D, W, dtype=torch.float32)
    for w in range(W):
        src = T - W + w
        if 0 <= src < T:
            final_state[:, :, w] = x[:, src, :].float()
        elif src < 0 and initial_state is not None:
            state_idx = src + W  # == T + w
            if 0 <= state_idx < W:
                final_state[:, :, w] = initial_state[:, :, state_idx].float()
    return final_state


def causal_conv1d_update_bdt_ref(x, conv_state, weight, bias=None, activation=None):
    """Reference for causal_conv1d_update_bdt_impl, built on top of F.conv1d depthwise.

    Args:
        x:          [B, D, T]
        conv_state: [B, D, state_len]
        weight:     [D, W]
        bias:       [D] or None
    """
    B, D, _ = x.shape
    W = weight.shape[1]
    state_len = conv_state.shape[2]

    prefix_len = min(W - 1, state_len)
    prefix = conv_state[:, :, state_len - prefix_len:].float()  # [B, D, prefix_len]
    if prefix_len < W - 1:
        prefix = F.pad(prefix, (W - 1 - prefix_len, 0))  # [B, D, W-1]

    x_cat = torch.cat([prefix, x.float()], dim=2)  # [B, D, W-1+T]
    out = F.conv1d(x_cat, weight.unsqueeze(1).float(), bias.float() if bias is not None else None, padding=0, groups=D)

    if activation in ("silu", "swish"):
        out = out * torch.sigmoid(out)
    return out
