# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Backward-path tests: gradient correctness for bwd_impl, varlen, state, and autograd wrapper."""

# pylint: disable=duplicate-code,possibly-used-before-assignment
# - duplicate-code: kwargs-heavy API calls inevitably repeat across test files
# - possibly-used-before-assignment: arch32 names are bound only when not is_arch35();
#   conftest auto-skips this whole package on arch35 so the names are never accessed there

import pytest
import torch

from mindspeed_mm.fsdp.models.qwen3_5.causal_conv1d import causal_conv1d
from mindspeed_mm.fsdp.ops.gdn.triton.utils import is_arch35

from .conftest import DEVICE
from .reference_impl import (
    causal_conv1d_ref,
    causal_conv1d_ref_no_state,
    causal_conv1d_update_states_ref,
)

if not is_arch35():
    from mindspeed_mm.fsdp.ops.gdn.triton.convolution import (
        causal_conv1d_bwd_impl,
        causal_conv1d_fwd_impl,
    )


# ============================================================
# 1. Backward gradient correctness
# ============================================================


class TestCausalConv1dBackward:
    """Backward kernel gradient correctness."""

    def _run_bwd_test(self, B, T, D, W, dtype, bias=False, activation=None, initial_state=False):
        torch.manual_seed(42)

        x = torch.randn(B, T, D, dtype=dtype, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=dtype, device=DEVICE, requires_grad=True)
        b = torch.randn(D, dtype=dtype, device=DEVICE, requires_grad=True) if bias else None
        h0 = torch.randn(B, D, W, dtype=dtype, device=DEVICE, requires_grad=True) if initial_state else None

        # forward
        y_npu, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=b,
            residual=None,
            initial_state=h0,
            activation=activation,
            output_final_state=False,
        )

        # synthetic upstream gradient
        dy = torch.randn_like(y_npu)

        # NPU backward
        dx_npu, dw_npu, db_npu, _, dh0_npu = causal_conv1d_bwd_impl(
            x=x.detach(),
            dy=dy,
            dht=None,
            weight=w.detach(),
            bias=b.detach() if b is not None else None,
            residual=None,
            initial_state=h0.detach() if h0 is not None else None,
            activation=activation,
        )

        # Reference autograd
        x_ref = x.detach().clone().float().requires_grad_(True)
        w_ref = w.detach().clone().float().requires_grad_(True)
        b_ref = b.detach().clone().float().requires_grad_(True) if b is not None else None
        h0_ref = h0.detach().clone().float().requires_grad_(True) if h0 is not None else None

        if initial_state:
            y_ref = causal_conv1d_ref(x_ref, w_ref, bias=b_ref, initial_state=h0_ref, activation=activation)
        else:
            y_ref = causal_conv1d_ref_no_state(x_ref, w_ref, bias=b_ref, activation=activation)
        y_ref.backward(dy.float())

        atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4

        torch.testing.assert_close(dx_npu.float(), x_ref.grad.to(dtype).float(), atol=atol, rtol=rtol)
        if dw_npu is not None:
            torch.testing.assert_close(dw_npu.float(), w_ref.grad.to(dtype).float(), atol=atol, rtol=rtol)
        if db_npu is not None and b_ref is not None:
            torch.testing.assert_close(db_npu.float(), b_ref.grad.to(dtype).float(), atol=atol, rtol=rtol)
        if dh0_npu is not None and h0_ref is not None:
            torch.testing.assert_close(dh0_npu.float(), h0_ref.grad.to(dtype).float(), atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (1, 64, 256, 4),
            (2, 128, 256, 4),
            (2, 256, 512, 4),
        ],
    )
    def test_grad_basic(self, B, T, D, W):
        """Basic gradient correctness."""
        self._run_bwd_test(B, T, D, W, torch.float32)

    def test_grad_with_bias(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, bias=True)

    def test_grad_with_silu(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, activation="silu")

    def test_grad_bias_and_silu(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, bias=True, activation="silu")

    def test_grad_fp16(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float16)

    def test_grad_with_initial_state(self):
        """initial_state, also verifies dh0."""
        self._run_bwd_test(2, 128, 256, 4, torch.float32, initial_state=True)

    def test_grad_with_initial_state_and_bias(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, bias=True, initial_state=True)

    def test_grad_with_initial_state_and_silu(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, activation="silu", initial_state=True)

    def test_grad_initial_state_all_options(self):
        self._run_bwd_test(2, 128, 256, 4, torch.float32, bias=True, activation="silu", initial_state=True)


# ============================================================
# 2. Variable-length sequence (cu_seqlens)
# ============================================================


class TestCausalConv1dVarlen:
    """Packed varlen scenarios."""

    def _make_varlen_inputs(self, seq_lens, D, W, dtype):
        total_T = sum(seq_lens)
        B_packed = 1
        N = len(seq_lens)

        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=DEVICE)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        x = torch.randn(B_packed, total_T, D, dtype=dtype, device=DEVICE)
        w = torch.randn(W, D, dtype=dtype, device=DEVICE)
        return x, w, cu_seqlens

    @pytest.mark.parametrize(
        "seq_lens",
        [
            [32, 64, 48],
            [128, 128],
            [16, 256, 32, 64],
            [16, 256, 32, 64],
            [4096, 2048, 5111],
            [1000, 2000, 3000],
            [1273, 5133, 3333],
        ],
    )
    def test_varlen_fwd(self, seq_lens):
        D, W = 256, 4
        x, w, cu_seqlens = self._make_varlen_inputs(seq_lens, D, W, torch.float32)

        y, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )

        # Segment-by-segment validation
        for i, _ in enumerate(seq_lens):
            start = int(cu_seqlens[i])
            end = int(cu_seqlens[i + 1])
            x_seg = x[:, start:end, :]
            y_seg = y[:, start:end, :]
            y_ref = causal_conv1d_ref_no_state(x_seg, w.float())
            torch.testing.assert_close(y_seg.float(), y_ref.to(y_seg.dtype).float(), atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize(
        "seq_lens",
        [
            [1, 2, 3],  # very short sequences
            [3, 3, 3],  # all sequences shorter than W
        ],
    )
    def test_varlen_short_seqs(self, seq_lens):
        D, W = 256, 4
        x, w, cu_seqlens = self._make_varlen_inputs(seq_lens, D, W, torch.float32)

        y, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )
        assert y.shape == x.shape

    @pytest.mark.parametrize(
        "seq_lens",
        [
            [32, 64, 48],
            [128, 128],
        ],
    )
    def test_varlen_bwd(self, seq_lens):
        D, W = 256, 4
        x, w, cu_seqlens = self._make_varlen_inputs(seq_lens, D, W, torch.float32)
        x.requires_grad_(True)
        w.requires_grad_(True)

        y, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
            output_final_state=False,
        )

        dy = torch.randn_like(y)
        dx_npu, dw_npu, _, _, _ = causal_conv1d_bwd_impl(
            x=x.detach(),
            dy=dy,
            dht=None,
            weight=w.detach(),
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
        )

        # Reference: per-segment autograd
        x_ref = x.detach().clone().float().requires_grad_(True)
        w_ref = w.detach().clone().float().requires_grad_(True)

        y_ref_parts = []
        for i, _ in enumerate(seq_lens):
            start = int(cu_seqlens[i])
            end = int(cu_seqlens[i + 1])
            y_ref_parts.append(causal_conv1d_ref_no_state(x_ref[:, start:end, :], w_ref))
        y_ref = torch.cat(y_ref_parts, dim=1)
        y_ref.backward(dy.float())

        torch.testing.assert_close(dx_npu.float(), x_ref.grad.to(torch.float32), atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dw_npu.float(), w_ref.grad.to(torch.float32), atol=1e-4, rtol=1e-4)


# ============================================================
# 3. initial_state / final_state
# ============================================================


class TestCausalConv1dState:
    """initial_state / final_state semantics and gradient propagation."""

    def test_final_state_shape(self):
        B, T, D, W = 2, 128, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)

        y, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            output_final_state=True,
        )

        assert final_state is not None
        assert final_state.shape == (B, D, W)

    def test_final_state_values(self):
        """final_state must hold the last W timesteps of x, value-wise."""
        B, T, D, W = 1, 32, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)

        _, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            output_final_state=True,
        )

        assert not torch.isnan(final_state).any()

        final_state_ref = causal_conv1d_update_states_ref(x.cpu(), W)
        torch.testing.assert_close(final_state.cpu().float(), final_state_ref, atol=1e-5, rtol=1e-5)

    def test_initial_state_effect(self):
        """initial_state should change early timesteps only."""
        B, T, D, W = 1, 16, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)

        # Without initial_state
        y_no_state, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=None,
            output_final_state=False,
        )

        # With (non-zero) initial_state
        h0 = torch.randn(B, D, W, dtype=torch.float32, device=DEVICE)
        y_with_state, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=h0,
            output_final_state=False,
        )

        # First W-1 timesteps must differ
        diff = (y_no_state[:, : W - 1, :] - y_with_state[:, : W - 1, :]).abs().sum()
        assert diff > 0, "initial_state should affect early timesteps"

        # Far-enough timesteps should be identical (when T >> W)
        if T > 2 * W:
            tail_diff = (y_no_state[:, 2 * W :, :] - y_with_state[:, 2 * W :, :]).abs().max()
            torch.testing.assert_close(tail_diff, torch.tensor(0.0, device=DEVICE), atol=1e-5, rtol=0)

        # Numerically match the reference (initial_state path)
        y_ref = causal_conv1d_ref(x.cpu(), w.cpu(), initial_state=h0.cpu())
        torch.testing.assert_close(y_with_state.cpu().float(), y_ref, atol=1e-4, rtol=1e-4)

    def test_state_continuity(self):
        """Splitting a sequence and chaining final_state -> initial_state must equal full pass."""
        B, D, W = 1, 256, 4
        T1, T2 = 64, 64
        T = T1 + T2

        x_full = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)

        # Single-pass
        y_full, _ = causal_conv1d_fwd_impl(
            x=x_full,
            weight=w,
            bias=None,
            residual=None,
            output_final_state=False,
        )

        # Two-segment
        x1, x2 = x_full[:, :T1, :], x_full[:, T1:, :]

        y1, state1 = causal_conv1d_fwd_impl(
            x=x1,
            weight=w,
            bias=None,
            residual=None,
            output_final_state=True,
        )

        y2, _ = causal_conv1d_fwd_impl(
            x=x2,
            weight=w,
            bias=None,
            residual=None,
            initial_state=state1,
            output_final_state=False,
        )

        y_concat = torch.cat([y1, y2], dim=1)
        torch.testing.assert_close(y_concat, y_full, atol=1e-5, rtol=1e-5)

        # Cross-check against the PyTorch reference for absolute correctness
        y_ref = causal_conv1d_ref(x_full.cpu(), w.cpu())
        torch.testing.assert_close(y_full.cpu().float(), y_ref, atol=1e-5, rtol=1e-5)

    def test_backward_with_output_final_state(self):
        """dht must propagate correctly to dx when output_final_state=True."""
        B, T, D, W = 2, 64, 256, 4  # T >> W so final_state depends only on x
        torch.manual_seed(42)

        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        h0 = torch.randn(B, D, W, dtype=torch.float32, device=DEVICE, requires_grad=True)

        # NPU forward with output_final_state
        y_npu, final_state = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=None,
            residual=None,
            initial_state=h0,
            activation="silu",
            output_final_state=True,
        )

        dy = torch.randn_like(y_npu)
        dht = torch.randn_like(final_state)

        # NPU backward with dht
        dx_npu, dw_npu, _, _, dh0_npu = causal_conv1d_bwd_impl(
            x=x.detach(),
            dy=dy,
            dht=dht,
            weight=w.detach(),
            bias=None,
            residual=None,
            initial_state=h0.detach(),
            activation="silu",
        )

        # CPU autograd reference (final_state depends only on x, not on weight/h0)
        x_ref = x.detach().clone().cpu().float().requires_grad_(True)
        w_ref = w.detach().clone().cpu().float().requires_grad_(True)
        h0_ref = h0.detach().clone().cpu().float().requires_grad_(True)

        y_ref = causal_conv1d_ref(x_ref, w_ref, initial_state=h0_ref, activation="silu")
        # T >= W: final_state[b, d, w] = x[b, T-W+w, d]
        final_state_ref = x_ref[:, -W:, :].permute(0, 2, 1).contiguous()  # [B, D, W]

        loss = (y_ref * dy.cpu().float()).sum() + (final_state_ref * dht.cpu().float()).sum()
        grads = torch.autograd.grad(loss, (x_ref, w_ref, h0_ref))

        torch.testing.assert_close(dx_npu.cpu().float(), grads[0], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dw_npu.cpu().float(), grads[1], atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(dh0_npu.cpu().float(), grads[2], atol=1e-4, rtol=1e-4)


# ============================================================
# 4. CausalConv1dFunction autograd wrapper
# ============================================================


class TestCausalConv1dAutograd:
    """End-to-end tests on the `causal_conv1d` autograd wrapper."""

    def test_wrapper_matches_impl_all_options(self):
        """wrapper forward output must be bit-exact vs direct fwd_impl, across all options."""
        B, T, D, W = 2, 128, 256, 4
        torch.manual_seed(42)
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        b = torch.randn(D, dtype=torch.float32, device=DEVICE)
        r = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        h0 = torch.randn(B, D, W, dtype=torch.float32, device=DEVICE)

        y_wrap, fs_wrap = causal_conv1d(
            x,
            w,
            bias=b,
            residual=r,
            initial_state=h0,
            activation="silu",
            output_final_state=True,
        )
        y_impl, fs_impl = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=b,
            residual=r,
            initial_state=h0,
            activation="silu",
            output_final_state=True,
        )
        torch.testing.assert_close(y_wrap, y_impl, atol=0, rtol=0)
        torch.testing.assert_close(fs_wrap, fs_impl, atol=0, rtol=0)

    # ---- Reverse: autograd gradient vs reference ----

    def _grad_test(self, bias=False, activation=None, initial_state=False, residual=False, dtype=torch.float32):
        B, T, D, W = 2, 64, 256, 4
        torch.manual_seed(42)
        atol = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        rtol = atol

        # ---- through wrapper -> autograd (NPU) ----
        x1 = torch.randn(B, T, D, dtype=dtype, device=DEVICE, requires_grad=True)
        w1 = torch.randn(W, D, dtype=dtype, device=DEVICE, requires_grad=True)
        b1 = torch.randn(D, dtype=dtype, device=DEVICE, requires_grad=True) if bias else None
        r1 = torch.randn(B, T, D, dtype=dtype, device=DEVICE, requires_grad=True) if residual else None
        h1 = torch.randn(B, D, W, dtype=dtype, device=DEVICE, requires_grad=True) if initial_state else None

        y1, _ = causal_conv1d(x1, w1, bias=b1, residual=r1, initial_state=h1, activation=activation)
        loss1 = y1.sum()
        loss1.backward()

        # ---- reference -> autograd (CPU float32) ----
        x2 = x1.detach().cpu().float().requires_grad_(True)
        w2 = w1.detach().cpu().float().requires_grad_(True)
        b2 = b1.detach().cpu().float().requires_grad_(True) if bias else None
        r2 = r1.detach().cpu().float().requires_grad_(True) if residual else None
        h2 = h1.detach().cpu().float().requires_grad_(True) if initial_state else None

        y2 = causal_conv1d_ref(x2, w2, bias=b2, residual=r2, initial_state=h2, activation=activation)
        loss2 = y2.sum()
        loss2.backward()

        torch.testing.assert_close(x1.grad.cpu().float(), x2.grad, atol=atol, rtol=rtol)
        torch.testing.assert_close(w1.grad.cpu().float(), w2.grad, atol=atol, rtol=rtol)
        if bias:
            torch.testing.assert_close(b1.grad.cpu().float(), b2.grad, atol=atol, rtol=rtol)
        if residual:
            torch.testing.assert_close(r1.grad.cpu().float(), r2.grad, atol=atol, rtol=rtol)
        if initial_state:
            torch.testing.assert_close(h1.grad.cpu().float(), h2.grad, atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"bias": True, "activation": "silu", "residual": True, "initial_state": True},
            {"dtype": torch.float16},
            {"dtype": torch.bfloat16},
        ],
        ids=["basic", "all_options", "fp16", "bf16"],
    )
    def test_autograd(self, kwargs):
        """wrapper autograd across basic / all_options / fp16 / bf16."""
        self._grad_test(**kwargs)

    def test_double_backward_no_crash(self):
        """Two consecutive forward+backward must not crash (ctx lifetime sanity)."""
        B, T, D, W = 2, 64, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)

        for _ in range(2):
            y, _ = causal_conv1d(x, w)
            y.sum().backward()
            x.grad = None
            w.grad = None

    def test_autograd_with_final_state_grad(self):
        """dht must propagate through autograd back to dx."""
        B, T, D, W = 2, 64, 256, 4
        torch.manual_seed(42)

        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)

        y, fs = causal_conv1d(x, w, output_final_state=True)
        assert fs is not None

        loss = y.sum() + fs.sum()
        loss.backward()

        assert x.grad is not None
        assert w.grad is not None
        assert x.grad.abs().sum() > 0

    @pytest.mark.parametrize(
        "seq_lens",
        [
            [32, 64, 48],
            [128, 128],
        ],
    )
    def test_autograd_varlen(self, seq_lens):
        """varlen via wrapper autograd; cross-check against direct bwd_impl."""
        D, W = 256, 4
        total_T = sum(seq_lens)
        N = len(seq_lens)

        cu_seqlens = torch.zeros(N + 1, dtype=torch.int32, device=DEVICE)
        for i, sl in enumerate(seq_lens):
            cu_seqlens[i + 1] = cu_seqlens[i] + sl

        x = torch.randn(1, total_T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)

        y, _ = causal_conv1d(x, w, cu_seqlens=cu_seqlens)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert w.grad is not None
        assert w.grad.shape == w.shape

        # Bit-exact match vs direct bwd_impl
        dy = torch.ones_like(y)
        dx_impl, dw_impl, _, _, _ = causal_conv1d_bwd_impl(
            x=x.detach(),
            dy=dy,
            dht=None,
            weight=w.detach(),
            bias=None,
            residual=None,
            initial_state=None,
            activation=None,
            cu_seqlens=cu_seqlens,
        )
        torch.testing.assert_close(x.grad, dx_impl, atol=0, rtol=0)
        torch.testing.assert_close(w.grad, dw_impl, atol=0, rtol=0)

    def test_return_none_final_state(self):
        """output_final_state=False -> final_state must be None."""
        B, T, D, W = 2, 64, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)

        y, fs = causal_conv1d(x, w, output_final_state=False)
        assert fs is None
        assert y.shape == (B, T, D)
