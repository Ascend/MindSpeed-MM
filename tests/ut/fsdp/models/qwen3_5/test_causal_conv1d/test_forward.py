# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang, Wenshuo Zhao
# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Forward-path tests: correctness of fwd_impl + UpdateBDT path + edge cases."""

# pylint: disable=duplicate-code,possibly-used-before-assignment
# - duplicate-code: kwargs-heavy API calls inevitably repeat across test files
# - possibly-used-before-assignment: arch32 names are bound only when not is_arch35();
#   conftest auto-skips this whole package on arch35 so the names are never accessed there

import pytest
import torch

from mindspeed_mm.fsdp.models.qwen3_5.triton.utils import is_arch35

from .conftest import DEVICE
from .reference_impl import (
    causal_conv1d_ref,
    causal_conv1d_ref_no_state,
    causal_conv1d_update_bdt_ref,
)

if not is_arch35():
    from mindspeed_mm.fsdp.models.qwen3_5.triton.convolution import (
        causal_conv1d_fwd_impl,
        causal_conv1d_update_bdt_impl,
    )


# ============================================================
# 1. Forward correctness vs PyTorch reference
# ============================================================


class TestCausalConv1dForward:
    """Forward kernel correctness."""

    def _run_fwd_test(self, B, T, D, W, dtype, bias=False, residual=False, activation=None, initial_state=False):
        torch.manual_seed(42)

        x = torch.randn(B, T, D, dtype=dtype, device=DEVICE)
        w = torch.randn(W, D, dtype=dtype, device=DEVICE)
        b = torch.randn(D, dtype=dtype, device=DEVICE) if bias else None
        r = torch.randn(B, T, D, dtype=dtype, device=DEVICE) if residual else None
        h0 = torch.randn(B, D, W, dtype=dtype, device=DEVICE) if initial_state else None

        y_npu, _ = causal_conv1d_fwd_impl(
            x=x,
            weight=w,
            bias=b,
            residual=r,
            initial_state=h0,
            activation=activation,
            output_final_state=False,
        )

        if initial_state:
            y_ref = causal_conv1d_ref(
                x.float(),
                w.float(),
                bias=b.float() if b is not None else None,
                residual=r.float() if r is not None else None,
                initial_state=h0.float(),
                activation=activation,
            )
        else:
            y_ref = causal_conv1d_ref_no_state(
                x.float(),
                w.float(),
                bias=b.float() if b is not None else None,
                residual=r.float() if r is not None else None,
                activation=activation,
            )

        atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        rtol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
        torch.testing.assert_close(y_npu.float(), y_ref.to(y_npu.dtype).float(), atol=atol, rtol=rtol)

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (1, 64, 256, 4),
            (2, 128, 256, 4),
            (4, 256, 512, 4),
            (1, 32, 256, 2),
        ],
    )
    def test_basic_shapes(self, B, T, D, W):
        """Basic shape combinations."""
        self._run_fwd_test(B, T, D, W, torch.float32)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_dtypes(self, dtype):
        """Different dtypes."""
        self._run_fwd_test(2, 128, 256, 4, dtype)

    def test_with_bias(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, bias=True)

    def test_with_residual(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, residual=True)

    def test_with_silu_activation(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, activation="silu")

    def test_with_swish_activation(self):
        """swish is an alias of silu."""
        self._run_fwd_test(2, 128, 256, 4, torch.float32, activation="swish")

    def test_all_options_combined(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, bias=True, residual=True, activation="silu")

    def test_with_bias_fp16(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float16, bias=True)

    def test_large_kernel_width(self):
        self._run_fwd_test(2, 128, 256, 8, torch.float32)

    @pytest.mark.parametrize("T", [1, 3, 7, 15, 31])
    def test_small_seq_len(self, T):
        """Sequence shorter than BT."""
        self._run_fwd_test(1, T, 256, 4, torch.float32)

    def test_with_initial_state(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, initial_state=True)

    def test_with_initial_state_and_activation(self):
        self._run_fwd_test(2, 128, 256, 4, torch.float32, initial_state=True, activation="silu")

    def test_with_initial_state_all_options(self):
        self._run_fwd_test(
            2, 128, 256, 4, torch.float32, bias=True, residual=True, activation="silu", initial_state=True
        )

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_with_initial_state_fp16_bf16(self, dtype):
        self._run_fwd_test(2, 128, 256, 4, dtype, initial_state=True)


# ============================================================
# 2. BDT update path (causal_conv1d_update_bdt_impl)
# ============================================================


class TestCausalConv1dUpdateBDT:
    """causal_conv1d_update_bdt_impl (BDT layout)."""

    def test_bdt_basic(self):
        """BDT layout basic correctness vs reference."""
        B, D, T = 2, 256, 64
        W = 4
        x = torch.randn(B, D, T, dtype=torch.float32, device=DEVICE)
        w = torch.randn(D, W, dtype=torch.float32, device=DEVICE)
        conv_state = torch.randn(B, D, T, dtype=torch.float32, device=DEVICE)
        conv_state_orig = conv_state.clone()

        out = causal_conv1d_update_bdt_impl(
            x=x,
            conv_state=conv_state,
            weight=w,
            bias=None,
            activation=None,
        )
        assert out.shape == x.shape

        out_ref = causal_conv1d_update_bdt_ref(x.cpu(), conv_state_orig.cpu(), w.cpu())
        torch.testing.assert_close(out.cpu().float(), out_ref, atol=1e-4, rtol=1e-4)

        # conv_state must be updated in-place: last state_len columns come from x
        state_len = T
        if T >= state_len:
            expected_tail = x[:, :, -state_len:].cpu()
            torch.testing.assert_close(conv_state.cpu().float(), expected_tail, atol=1e-4, rtol=1e-4)

    def test_bdt_with_silu(self):
        """BDT + SiLU activation."""
        B, D, T = 2, 256, 64
        W = 4
        x = torch.randn(B, D, T, dtype=torch.float32, device=DEVICE)
        w = torch.randn(D, W, dtype=torch.float32, device=DEVICE)
        conv_state = torch.randn(B, D, T, dtype=torch.float32, device=DEVICE)
        conv_state_orig = conv_state.clone()

        out = causal_conv1d_update_bdt_impl(
            x=x,
            conv_state=conv_state,
            weight=w,
            bias=None,
            activation="silu",
        )
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

        out_ref = causal_conv1d_update_bdt_ref(x.cpu(), conv_state_orig.cpu(), w.cpu(), activation="silu")
        torch.testing.assert_close(out.cpu().float(), out_ref, atol=1e-4, rtol=1e-4)

    def test_bdt_2d_input(self):
        """2D input auto-unsqueezed inside the kernel."""
        B, D = 2, 256
        W = 4
        x = torch.randn(B, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(D, W, dtype=torch.float32, device=DEVICE)
        conv_state = torch.randn(B, D, W, dtype=torch.float32, device=DEVICE)
        conv_state_orig = conv_state.clone()

        out = causal_conv1d_update_bdt_impl(
            x=x,
            conv_state=conv_state,
            weight=w,
        )
        assert out.shape == (B, D)

        # Kernel unsqueezes(-1) internally and squeezes(-1) on the way out; the ref does the same
        out_ref = causal_conv1d_update_bdt_ref(x.cpu().unsqueeze(-1), conv_state_orig.cpu(), w.cpu()).squeeze(-1)
        torch.testing.assert_close(out.cpu().float(), out_ref, atol=1e-4, rtol=1e-4)


# ============================================================
# 3. Edge cases
# ============================================================


class TestCausalConv1dEdgeCases:
    """Boundary conditions and degenerate inputs."""

    def test_seq_len_equals_one(self):
        """T=1"""
        B, T, D, W = 2, 1, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert y.shape == (B, T, D)

    def test_seq_len_equals_width(self):
        """T=W"""
        B, D, W = 2, 256, 4
        T = W
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert y.shape == (B, T, D)

    def test_seq_len_less_than_width(self):
        """T < W"""
        B, D, W = 2, 256, 4
        T = 2
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert y.shape == (B, T, D)

    def test_width_one(self):
        """W=1 degenerates to pointwise multiply."""
        B, T, D, W = 2, 64, 256, 1
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        y_ref = causal_conv1d_ref_no_state(x.float(), w.float())
        torch.testing.assert_close(y.float(), y_ref.to(y.dtype).float(), atol=1e-5, rtol=1e-5)

    def test_large_batch(self):
        B, T, D, W = 32, 64, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert y.shape == (B, T, D)

    def test_no_nan_in_output(self):
        B, T, D, W = 2, 128, 256, 4
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    def test_zero_input(self):
        """All-zero input -> all-zero output."""
        B, T, D, W = 2, 64, 256, 4
        x = torch.zeros(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        torch.testing.assert_close(y, torch.zeros_like(y), atol=1e-7, rtol=0)

    def test_d_not_multiple_of_bd_should_fail(self):
        """D not divisible by BD must raise."""
        B, T, D, W = 2, 64, 192, 4  # 192 % 256 != 0
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        with pytest.raises(ValueError):
            causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)


class TestCausalConv1dPrecision:
    """不同精度下的误差范围"""

    @pytest.mark.parametrize(
        "dtype,atol,rtol",
        [
            (torch.float32, 1e-5, 1e-5),
            (torch.float16, 5e-3, 5e-3),
            (torch.bfloat16, 1e-2, 1e-2),
        ],
    )
    def test_precision_bounds(self, dtype, atol, rtol):
        B, T, D, W = 2, 128, 256, 4
        torch.manual_seed(0)
        x = torch.randn(B, T, D, dtype=dtype, device=DEVICE)
        w = torch.randn(W, D, dtype=dtype, device=DEVICE)

        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        y_ref = causal_conv1d_ref_no_state(x.float(), w.float())

        torch.testing.assert_close(y.float(), y_ref.to(dtype).float(), atol=atol, rtol=rtol)
