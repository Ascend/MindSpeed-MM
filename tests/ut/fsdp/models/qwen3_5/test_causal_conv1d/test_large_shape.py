# Copyright (c) 2026, Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Large-shape / real-world scenarios + performance benchmarks.

All tests in this file are marked `slow` and skipped by default
(deselect with `-m 'not slow'`).
"""

# pylint: disable=duplicate-code,possibly-used-before-assignment
# - duplicate-code: kwargs-heavy API calls inevitably repeat across test files
# - possibly-used-before-assignment: arch32 names are bound only when not is_arch35();
#   conftest auto-skips this whole package on arch35 so the names are never accessed there

import time

import pytest
import torch

from mindspeed_mm.fsdp.models.qwen3_5.causal_conv1d import causal_conv1d
from mindspeed_mm.fsdp.models.qwen3_5.triton.utils import is_arch35

from .conftest import DEVICE
from .reference_impl import causal_conv1d_ref

if not is_arch35():
    from mindspeed_mm.fsdp.models.qwen3_5.triton.convolution import (
        causal_conv1d_bwd_impl,
        causal_conv1d_fwd_impl,
    )


def _device_sync():
    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


# ============================================================
# 1. Performance benchmarks
# ============================================================


class TestCausalConv1dPerformance:
    """Perf benchmarks (slow, opt-in)."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (8, 2048, 256, 4),
            (8, 4096, 512, 4),
            (16, 2048, 2048, 4),
            (8, 2048, 3072, 4),
        ],
    )
    def test_throughput(self, B, T, D, W):
        """Forward throughput."""
        x = torch.randn(B, T, D, dtype=torch.float16, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float16, device=DEVICE)
        for _ in range(5):
            causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        _device_sync()
        N = 100
        start = time.perf_counter()
        for _ in range(N):
            causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        _device_sync()
        elapsed = time.perf_counter() - start
        print(f"\n[Perf] B={B}, T={T}, D={D}, W={W}: {N / elapsed:.1f} iters/s, {elapsed / N * 1000:.2f} ms/iter")

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (8, 2048, 1536, 4),
            (4, 2048, 4096, 4),
            (1, 4096, 8192, 4),
        ],
    )
    def test_fwd_bwd_e2e_perf(self, B, T, D, W):
        """End-to-end fwd+bwd perf via wrapper."""
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        for _ in range(5):
            y, _ = causal_conv1d(x, w, activation="silu")
            y.sum().backward()
            x.grad = w.grad = None
        _device_sync()

        N = 10
        fwd_times, bwd_times = [], []
        for _ in range(N):
            x_i = x.detach().requires_grad_(True)
            w_i = w.detach().requires_grad_(True)
            _device_sync()
            t0 = time.perf_counter()
            y, _ = causal_conv1d(x_i, w_i, activation="silu")
            _device_sync()
            t1 = time.perf_counter()
            y.sum().backward()
            _device_sync()
            fwd_times.append((t1 - t0) * 1000)
            bwd_times.append((time.perf_counter() - t1) * 1000)

        fwd, bwd = sum(fwd_times) / N, sum(bwd_times) / N
        print(f"\n[Perf E2E] B={B}, T={T}, D={D}, W={W}: fwd={fwd:.2f}ms bwd={bwd:.2f}ms total={fwd + bwd:.2f}ms")


# ============================================================
# 2. Real-model shape coverage
# ============================================================
# Covers typical causal conv1d configs in SSM-style architectures
# (Mamba / Jamba / Qwen3-Omni). Primary goal: no UB overflow, no precision
# regression, no OOM at production-scale shapes.


@pytest.mark.slow
class TestCausalConv1dRealShapes:
    """Realistic / production-scale shape coverage."""

    def _run_fwd_bwd_test(self, B, T, D, W, dtype=torch.float32, bias=False, activation=None, initial_state=False):
        """End-to-end fwd+bwd, compared against the CPU PyTorch reference."""
        torch.manual_seed(42)
        atol_fwd = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
        atol_bwd = 5e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-3

        x = torch.randn(B, T, D, dtype=dtype, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=dtype, device=DEVICE, requires_grad=True)
        b = torch.randn(D, dtype=dtype, device=DEVICE, requires_grad=True) if bias else None
        h0 = torch.randn(B, D, W, dtype=dtype, device=DEVICE, requires_grad=True) if initial_state else None

        # triton forward
        y, _ = causal_conv1d(x, w, bias=b, initial_state=h0, activation=activation)

        # ref forward (CPU)
        x_ref = x.detach().cpu().float().requires_grad_(True)
        w_ref = w.detach().cpu().float().requires_grad_(True)
        b_ref = b.detach().cpu().float().requires_grad_(True) if bias else None
        h0_ref = h0.detach().cpu().float().requires_grad_(True) if initial_state else None
        y_ref = causal_conv1d_ref(x_ref, w_ref, bias=b_ref, initial_state=h0_ref, activation=activation)

        # forward compare
        torch.testing.assert_close(y.detach().cpu().float(), y_ref.detach(), atol=atol_fwd, rtol=atol_fwd)

        # backward
        dy = torch.randn_like(y)
        y.backward(dy)
        y_ref.backward(dy.cpu().float())

        # dx
        torch.testing.assert_close(x.grad.cpu().float(), x_ref.grad, atol=atol_bwd, rtol=atol_bwd)
        # dw (accumulation noise grows with B*T, loosen atol proportionally)
        dw_atol = max(atol_bwd, 1e-3 * (B * T / 256))
        torch.testing.assert_close(w.grad.cpu().float(), w_ref.grad, atol=dw_atol, rtol=atol_bwd)

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (32, 2048, 1536, 4),
            (16, 2048, 2048, 4),
            (8, 2048, 3072, 4),
            (4, 2048, 4096, 4),
            (4, 8192, 2048, 4),
            (1, 2048, 768, 4),
            (1, 2048, 1536, 4),
            (1, 2048, 3072, 4),
            (1, 2048, 4096, 4),
        ],
    )
    def test_doc_training_shapes(self, B, T, D, W):
        self._run_fwd_bwd_test(B, T, D, W, bias=True, activation="silu")

    @pytest.mark.parametrize("T", [4096, 8192, 16384, 65536])
    def test_long_sequence(self, T):
        self._run_fwd_bwd_test(1, T, 256, 4)

    @pytest.mark.parametrize("D", [4096, 8192])
    def test_large_hidden_dim(self, D):
        self._run_fwd_bwd_test(1, 2048, D, 4, activation="silu")

    @pytest.mark.parametrize("B", [4, 8, 16])
    def test_large_batch(self, B):
        self._run_fwd_bwd_test(B, 2048, 256, 4)

    @pytest.mark.parametrize("W", [2, 3, 8])
    def test_kernel_widths(self, W):
        self._run_fwd_bwd_test(1, 2048, 256, W)

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (1, 65536, 8192, 4),  # single batch, ultra-long seq, large D
            (2, 4096, 4096, 4),
            (4, 8192, 3072, 4),
        ],
    )
    def test_production_combos(self, B, T, D, W):
        self._run_fwd_bwd_test(B, T, D, W, activation="silu")

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_large_shape_low_precision(self, dtype):
        self._run_fwd_bwd_test(1, 4096, 4096, 4, dtype=dtype, activation="silu")

    def test_large_shape_all_options(self):
        self._run_fwd_bwd_test(1, 4096, 1536, 4, bias=True, activation="silu", initial_state=True)

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (1, 131072, 256, 4),
            (1, 65536, 8192, 4),
        ],
    )
    def test_ultra_large_no_crash(self, B, T, D, W):
        """Smoke test: ultra-large shapes must not crash; skip accuracy compare to avoid CPU OOM."""
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE)
        y, _ = causal_conv1d_fwd_impl(x=x, weight=w, bias=None, residual=None)
        assert y.shape == (B, T, D)
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()

    @pytest.mark.parametrize(
        "B,T,D,W",
        [
            (1, 4096, 4096, 4),
            (2, 2048, 1536, 4),
        ],
    )
    def test_wrapper_large_shape(self, B, T, D, W):
        """wrapper autograd correctness at large shapes."""
        torch.manual_seed(42)
        x = torch.randn(B, T, D, dtype=torch.float32, device=DEVICE, requires_grad=True)
        w = torch.randn(W, D, dtype=torch.float32, device=DEVICE, requires_grad=True)

        y, _ = causal_conv1d(x, w, activation="silu")
        y.sum().backward()

        dy = torch.ones_like(y)
        dx_impl, dw_impl, _, _, _ = causal_conv1d_bwd_impl(
            x=x.detach(),
            dy=dy,
            dht=None,
            weight=w.detach(),
            bias=None,
            residual=None,
            initial_state=None,
            activation="silu",
        )

        torch.testing.assert_close(x.grad, dx_impl, atol=0, rtol=0)
        torch.testing.assert_close(w.grad, dw_impl, atol=0, rtol=0)
