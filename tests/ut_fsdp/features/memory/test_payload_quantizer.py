"""Payload quantizer UTs: int8 per-token and fp8 block-wise swap payloads.

Covers the quantizer contract (roundtrip fidelity, bitwise-correct N-dim
reshape bookkeeping through the quantize/dequantize facade, identity fallback,
fail-fast on an unknown quantizer name, no RNG consumption, no autograd state
on dequantized outputs), the swap-cache wiring (payload-byte accounting,
handle release via pop, the replay mode's unconsumed-route cleanup) and the
op-replay plumbing (scope config -> put(quantizer=...)).

The dequant kernels themselves (dispatch tiers, geometry, arch profiles) are
tested in tests/ut_fsdp/ops/test_int8_dequant.py.
"""
import logging
import sys

import pytest
import torch
from pydantic import ValidationError

from mindspeed_mm.fsdp.features.memory import op_replay as opr
from mindspeed_mm.fsdp.features.memory import swap_core as swap_core_mod
from mindspeed_mm.fsdp.features.memory.quantizer import (
    QUANTIZERS,
    Fp8BlockwiseQuantizer,
    Int8PerTokenQuantizer,
)
from mindspeed_mm.fsdp.features.memory.slab_arena import SlabArena
from mindspeed_mm.fsdp.features.memory.swap_core import SwapCache, SwapState
from mindspeed_mm.fsdp.params.feature_args import OpReplayScopeConfig
from mindspeed_mm.fsdp.utils.device import get_device_type, is_arch35


pytestmark = pytest.mark.skipif(
    not swap_core_mod.IS_DEVICE_AVAILABLE,
    reason="swap requires a usable device (fast fail on CPU-only hosts)")

MODE = Int8PerTokenQuantizer.name
INT8 = QUANTIZERS[MODE]


def _get_device():
    return torch.device(get_device_type())


def _make_cache(*args, **kwargs):
    """SwapCache with a default on-demand arena, as the SwapManager composes it."""
    kwargs.setdefault("cpu_arena", SlabArena.on_demand())
    return SwapCache(*args, **kwargs)


def _mean_rel_err(out, ref):
    out32, ref32 = out.float(), ref.float()
    return ((out32 - ref32).abs() / ref32.abs().clamp_min(1e-6)).mean().item()


def _device_rng_state():
    device = _get_device()
    if device.type == 'npu':
        return torch.npu.get_rng_state()
    if device.type == 'cuda':
        return torch.cuda.get_rng_state()
    return None


def _bits(t):
    """Raw-bit view of a bf16/fp16 tensor (torch.equal on values is not a
    bitwise comparison: it reports NaN != NaN and treats -0.0 == 0.0).
    """
    return t.reshape(-1).view(torch.int16)


def _assert_bitwise_equal(got, expected):
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert torch.equal(_bits(got), _bits(expected))


def _dequant_reference(payload, meta):
    """Independent reference: exact int8 -> fp32, per-token multiply, cast back."""
    return (payload.reshape(meta.reduced_shape).float()
            * meta.scale.reshape(*meta.reduced_shape[:-1], 1)
            ).to(meta.orig_dtype).reshape(meta.orig_shape)


class _CountingQuantizer:
    """Quantizer stub with call counters: the payload is a clone of the input."""
    name = 'stub'

    def __init__(self):
        self.quantize_calls = 0
        self.dequantize_calls = 0

    def quantize(self, t):
        self.quantize_calls += 1
        return t.clone(), {'shape': t.shape}

    def dequantize(self, payload, meta):
        self.dequantize_calls += 1
        return payload.clone()


class _PutSpyCache:
    """Stub swap cache recording every put(quantizer=...) call."""

    def __init__(self):
        self.puts = []

    def put(self, tensor, quantizer=None):
        self.puts.append((tensor, quantizer))
        return object()


class _FakeController:
    """OpReplaySession only shares _scope_stack / op_cache by reference."""

    def __init__(self, scope, op_cache):
        self._scope_stack = [scope]
        self.op_cache = op_cache


class TestInt8PerTokenRoundtrip:
    """int8 per-token payload: quantize at put, dequantize at pop"""

    def test_2d_roundtrip_shape_dtype_and_accuracy(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)  # deterministic: full D2H then H2D
        x = torch.randn(256, 512, dtype=torch.bfloat16, device=device)

        handle = cache.put(x, quantizer=INT8)
        # the cache stores the int8 payload (half the bytes), not the bf16 input
        assert handle.bytes() == x.numel()
        out = cache.pop(handle)

        assert out is not x  # a dequantized payload is a new tensor
        assert out.dtype == x.dtype and out.shape == x.shape
        assert out.stride() == x.stride()
        assert _mean_rel_err(out, x) < 0.08  # measured ~3.2%

    def test_1d_roundtrip_uses_2d_shim(self):
        """A 1D tensor has no token axis: the [1, numel] shim must be recorded
        and undone, so shape/dtype come back exactly
        """
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(4096, dtype=torch.bfloat16, device=device)

        # Slow path by construction (measured ~57 ms/GiB, ~40x the 2D form, and
        # per-token quantization degenerates to one scale for the whole tensor):
        # whether a scope quantizes 1D payloads is the scope configuration's
        # call, so the shim must stay correct.
        handle = cache.put(x, quantizer=INT8)
        assert handle.bytes() == x.numel()
        out = cache.pop(handle)

        assert out.dtype == x.dtype and out.shape == x.shape
        assert _mean_rel_err(out, x) < 0.08  # measured ~3.9%

    def test_fp16_roundtrip(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(64, 128, dtype=torch.float16, device=device)

        out = cache.pop(cache.put(x, quantizer=INT8))

        assert out.dtype == x.dtype and out.shape == x.shape
        assert _mean_rel_err(out, x) < 0.08

    def test_bitwise_matches_reference_across_dims_and_dtypes(self):
        """The facade's N-dim <-> 2D reshape bookkeeping must be invisible:
        dequantize(quantize(x)) is bitwise identical to the straight reference
        for every rank (1D shim included).
        """
        device = _get_device()
        quantizer = Int8PerTokenQuantizer()
        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((4096,), (64, 128), (4, 16, 32), (2, 3, 4, 8), (1, 1024)):
                x = torch.randn(*shape, dtype=dtype, device=device)
                payload, meta = quantizer.quantize(x)

                got = quantizer.dequantize(payload, meta)

                _assert_bitwise_equal(got, _dequant_reference(payload, meta))
                assert got.shape == x.shape and got.dtype == dtype


class TestUnquantizedPath:
    """No quantizer: unchanged behavior — same object back, no extra work"""

    def test_pop_returns_the_same_object(self):
        device = _get_device()
        # pass-through: the tensor stays in HBM, so pop's identity is observable
        cache = _make_cache(capacity_bytes=None)
        x = torch.randn(64, 128, dtype=torch.bfloat16, device=device)

        handle = cache.put(x)
        assert handle.quantizer is None and handle.meta is None
        assert cache.pop(handle) is x

    def test_bare_cache_still_serves_unquantized_puts(self):
        """A plain cache is the pre-quantizer cache; quantizers arrive per put"""
        device = _get_device()
        cache = SwapCache(capacity_bytes=None, cpu_arena=SlabArena.on_demand())
        x = torch.randn(8, 8, dtype=torch.bfloat16, device=device)
        assert cache.pop(cache.put(x)) is x

    def test_evicted_tensor_comes_back_as_a_rebuilt_object(self):
        """Pre-existing swap-core semantics, untouched by quantizers: once the
        tensor went HBM -> DDR -> HBM, the H2D landing rebuilds it via
        as_strided, so pop returns a new object. The no-quantizer identity
        guarantee applies while nothing was evicted (test above).
        """
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(64, 128, dtype=torch.bfloat16, device=device)

        out = cache.pop(cache.put(x))

        assert out is not x
        torch.testing.assert_close(out, x)


class TestQuantizerFallbackAndErrors:
    """skip = identity + one warning; unknown name = ValueError at scope construction"""

    def test_unsupported_tensor_stored_as_is_with_one_warning(self, caplog):
        device = _get_device()
        # pass-through: the identity fallback is observable as object identity
        cache = _make_cache(capacity_bytes=None)
        # A fresh quantizer instance: the warn-once memory is the quantizer's.
        quantizer = Int8PerTokenQuantizer()

        fp32 = torch.randn(64, 128, dtype=torch.float32, device=device)
        base = torch.randn(64, 128, dtype=torch.bfloat16, device=device)
        view = base.t()
        assert not view.is_contiguous()
        empty = torch.randn(0, 128, dtype=torch.bfloat16, device=device)

        with caplog.at_level(
                logging.WARNING,
                logger='mindspeed_mm.fsdp.features.memory.quantizer'):
            h0 = cache.put(fp32, quantizer=quantizer)        # unsupported dtype
            cache.put(fp32.clone(), quantizer=quantizer)     # same reason again
            h2 = cache.put(view, quantizer=quantizer)        # non-contiguous bf16
            cache.put(empty, quantizer=quantizer)            # empty bf16

        # passed through: the handle still carries the quantizer, meta is None
        assert h0.quantizer is quantizer and h0.meta is None
        assert h2.quantizer is quantizer and h2.meta is None
        warns = [r for r in caplog.records if 'does not support' in r.message]
        assert len(warns) == 3  # one per reject reason (dtype / contiguity / size)
        assert sum('non-contiguous' in r.message for r in warns) == 1
        assert sum('empty tensor' in r.message for r in warns) == 1
        # nothing was quantized: the very objects put in come back out
        assert cache.pop(h0) is fp32
        assert cache.pop(h2) is view

    def test_unknown_quantizer_name_raises_at_scope_construction(self):
        with pytest.raises(ValueError, match="unknown payload quantizer"):
            opr.OpReplayScope(quantizer="int4_imaginary")


class TestQuantizerContracts:
    """Contracts the swap path relies on: no RNG use, no autograd state"""

    def test_quantizer_consumes_no_rng(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(256, 256, dtype=torch.bfloat16, device=device)

        cpu_before = torch.get_rng_state()
        device_before = _device_rng_state()
        out = cache.pop(cache.put(x, quantizer=INT8))
        assert out is not None
        assert torch.equal(torch.get_rng_state(), cpu_before)
        if device_before is not None:
            assert torch.equal(_device_rng_state(), device_before)

    def test_dequantized_tensor_carries_no_autograd_state(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        with torch.enable_grad():
            x = torch.randn(32, 64, dtype=torch.bfloat16, device=device,
                            requires_grad=True)
            out = cache.pop(cache.put(x, quantizer=INT8))
        assert out.requires_grad is False
        assert out.grad_fn is None


class TestPopDequantizeAndAccounting:
    """pop dequantizes quantized handles; capacity is charged by payload bytes"""

    def test_pop_dequantizes_once_and_recycles_the_handle(self):
        """A quantized handle is dequantized exactly once by pop; the handle is
        released and the cache no longer serves it
        """
        device = _get_device()
        stub = _CountingQuantizer()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(64, 64, dtype=torch.float32, device=device)

        handle = cache.put(x, quantizer=stub)
        assert stub.quantize_calls == 1
        out = cache.pop(handle)
        assert stub.dequantize_calls == 1
        assert out is not x
        torch.testing.assert_close(out, x)

        assert handle.state() is SwapState.EMPTY
        assert handle.quantizer is None and handle.meta is None
        assert handle not in cache.handles
        assert cache.pop(handle) is None  # recycled, not double-served

    def test_cleanup_path_releases_unconsumed_routes(self):
        """Early stop: the replaying mode's exit pops every unconsumed route
        without error, leaving no handle behind in the shared cache. The
        payloads are dequantized even though nobody reads them — the known
        small waste of routing cleanup through pop (a release(handle) is the
        follow-up candidate).
        """
        device = _get_device()
        stub = _CountingQuantizer()
        cache = _make_cache(capacity_bytes=None)
        scope = opr.OpReplayScope(cache_ops=opr._parse_op_list(['aten.mm.default']),
                                  name='cleanup', quantizer=stub)
        session = opr.OpReplaySession(_FakeController(scope, cache))

        # routes exactly as OpReplayCachingMode would have recorded them
        leaves = []
        for _ in range(3):
            x = torch.randn(8, 8, dtype=torch.float32, device=device)
            leaves.append(opr._HandleInfo(cache.put(x, quantizer=stub)))
        session.token_routes[torch.ops.aten.mm.default].append((leaves, None, None))
        assert len(cache.handles) == 3

        with opr.OpReplayReplayingMode(session):
            pass  # early stop: the checkpoint exits without consuming the route

        assert len(cache.handles) == 0  # every handle released
        assert not session.token_routes  # route table drained
        assert stub.dequantize_calls == 3  # one dequantize per payload, unread

    def test_capacity_is_charged_by_payload_bytes(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)  # zero capacity: every put evicts
        x = torch.randn(1024, 1024, dtype=torch.bfloat16, device=device)
        payload_bytes = x.numel() * 1  # int8
        orig_bytes = x.numel() * x.element_size()  # bf16

        handle = cache.put(x, quantizer=INT8)
        assert handle.bytes() == payload_bytes
        assert cache._iter_stats['peak_ddr_bytes'] == payload_bytes
        cache.end_iteration()
        assert cache.stats['peak_ddr_bytes'] == payload_bytes  # the D2H moved half the bytes
        cache.pop(handle)
        cache.clear()

        # the same tensor unquantized accounts its full size
        handle2 = cache.put(x)
        assert handle2.bytes() == orig_bytes
        cache.end_iteration()
        assert cache.stats['peak_ddr_bytes'] == orig_bytes
        cache.pop(handle2)
        cache.clear()


class TestOpReplayWiring:
    """Scope config -> put(quantizer=...) plumbing"""

    def test_scope_resolves_quantizer_from_config(self):
        cfg = OpReplayScopeConfig(name='q', apply_modules=['0'], quantizer=MODE)
        assert opr.OpReplayScope.from_config(cfg).quantizer is INT8
        assert opr.OpReplayScope.from_config(OpReplayScopeConfig()).quantizer is None

    def test_config_rejects_an_unknown_quantizer_name(self):
        """A typo'd quantizer name fails at config parse, before any wiring."""
        with pytest.raises(ValidationError, match="int4_imaginary"):
            OpReplayScopeConfig(name='q', apply_modules=['0'],
                                quantizer='int4_imaginary')

    def test_config_rejects_invalid_fp8_block_size(self):
        with pytest.raises(ValidationError):
            OpReplayScopeConfig(name='q8', apply_modules=['0'],
                                quantizer=FP8_MODE, fp8_block_size=96)

    def test_config_rejects_fp8_block_size_for_int8(self):
        with pytest.raises(ValidationError, match='fp8_block_size'):
            OpReplayScopeConfig(name='q', apply_modules=['0'],
                                quantizer=MODE, fp8_block_size=64)

    def _run_mm_through_caching_mode(self, scope, op_cache):
        session = opr.OpReplaySession(_FakeController(scope, op_cache))
        a = torch.randn(4, 8, dtype=torch.bfloat16)
        b = torch.randn(8, 4, dtype=torch.bfloat16)
        with opr.OpReplayCachingMode(session):
            torch.mm(a, b)
        return [quantizer for _, quantizer in op_cache.puts]

    def test_caching_mode_puts_with_the_scope_quantizer(self):
        op_cache = _PutSpyCache()
        scope = opr.OpReplayScope(cache_ops=opr._parse_op_list(['aten.mm.default']),
                                  name='q', quantizer=MODE)
        assert self._run_mm_through_caching_mode(scope, op_cache) == [INT8]

        # a scope without a quantizer puts unquantized
        plain = opr.OpReplayScope(cache_ops=opr._parse_op_list(['aten.mm.default']),
                                  name='plain')
        assert self._run_mm_through_caching_mode(plain, _PutSpyCache()) == [None]


FP8_MODE = Fp8BlockwiseQuantizer.name
FP8 = QUANTIZERS[FP8_MODE]


def _fp8_dequant_reference(payload, meta):
    rows, cols = int(meta.reduced_shape[0]), int(meta.reduced_shape[-1])
    q = payload.reshape(rows, cols // meta.block_size, meta.block_size).float()
    return (q * meta.scale.unsqueeze(-1)).to(meta.orig_dtype).reshape(meta.orig_shape)


@pytest.mark.skipif(not is_arch35(), reason="fp8 block-wise quantization is only tested on A5/950")
class TestFp8BlockwiseRoundtrip:
    """fp8-e4m3 1x128 block-wise payload: quantize at put, dequantize at pop"""

    def test_2d_roundtrip_shape_dtype_and_accuracy(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(256, 512, dtype=torch.bfloat16, device=device)

        handle = cache.put(x, quantizer=FP8)
        assert handle.bytes() == x.numel()
        out = cache.pop(handle)

        assert out is not x
        assert out.dtype == x.dtype and out.shape == x.shape
        assert out.stride() == x.stride()
        assert _mean_rel_err(out, x) < 0.08

    def test_quantize_layout_scales_and_payload_dtype(self):
        device = _get_device()
        x = torch.randn(64, 1024, dtype=torch.bfloat16, device=device)

        payload, meta = FP8.quantize(x)

        assert payload.dtype is torch.float8_e4m3fn
        assert payload.shape == x.shape
        assert meta.scale.dtype is torch.float32
        assert meta.scale.shape == (64, 1024 // 128)
        amax = x.float().reshape(64, -1, 128).abs().amax(dim=-1)
        torch.testing.assert_close(meta.scale, amax / 448)

    @pytest.mark.parametrize('block_size', [64, 128, 192, 256])
    def test_configurable_block_size(self, block_size):
        device = _get_device()
        quantizer = Fp8BlockwiseQuantizer(block_size=block_size)
        x = torch.randn(8, block_size * 3, dtype=torch.bfloat16, device=device)

        payload, meta = quantizer.quantize(x)
        out = quantizer.dequantize(payload, meta)

        assert meta.block_size == block_size
        assert meta.scale.shape == (8, 3)
        amax = x.float().reshape(8, -1, block_size).abs().amax(dim=-1)
        torch.testing.assert_close(meta.scale, amax / 448)
        _assert_bitwise_equal(out, _fp8_dequant_reference(payload, meta))

    def test_invalid_block_size_is_rejected(self):
        with pytest.raises(ValueError, match='fp8 block size'):
            Fp8BlockwiseQuantizer(block_size=96)

    def test_1d_roundtrip_uses_2d_shim(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(4096, dtype=torch.bfloat16, device=device)

        handle = cache.put(x, quantizer=FP8)
        out = cache.pop(handle)

        assert handle.bytes() == x.numel()
        assert out.dtype == x.dtype and out.shape == x.shape
        assert _mean_rel_err(out, x) < 0.08

    def test_fp16_roundtrip(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=0)
        x = torch.randn(64, 128, dtype=torch.float16, device=device)

        out = cache.pop(cache.put(x, quantizer=FP8))

        assert out.dtype == x.dtype and out.shape == x.shape
        assert _mean_rel_err(out, x) < 0.08

    def test_bitwise_matches_reference_across_dims_and_dtypes(self):
        device = _get_device()
        quantizer = Fp8BlockwiseQuantizer()
        for dtype in (torch.bfloat16, torch.float16):
            for shape in ((4096,), (64, 128), (4, 16, 256), (2, 3, 4, 128), (1, 1024)):
                x = torch.randn(*shape, dtype=dtype, device=device)
                payload, meta = quantizer.quantize(x)

                got = quantizer.dequantize(payload, meta)

                _assert_bitwise_equal(got, _fp8_dequant_reference(payload, meta))
                assert got.shape == x.shape and got.dtype == dtype

    def test_128_scales_use_per_row_dequantization(self):
        device = _get_device()
        x = torch.randn(64, 256, dtype=torch.bfloat16, device=device)
        payload, meta = FP8.quantize(x)
        assert meta.scale.numel() == 128

        out = FP8.dequantize(payload, meta)

        _assert_bitwise_equal(out, _fp8_dequant_reference(payload, meta))

    def test_last_dim_not_multiple_of_128_passes_through(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=None)
        x = torch.randn(64, 100, dtype=torch.bfloat16, device=device)

        handle = cache.put(x, quantizer=FP8)

        assert handle.quantizer is FP8 and handle.meta is None
        assert cache.pop(handle) is x

    def test_scalar_passes_through(self):
        device = _get_device()
        cache = _make_cache(capacity_bytes=None)
        x = torch.randn((), dtype=torch.bfloat16, device=device)

        handle = cache.put(x, quantizer=FP8)

        assert handle.quantizer is FP8 and handle.meta is None
        assert cache.pop(handle) is x

    def test_scope_resolves_fp8_quantizer_from_config(self):
        cfg = OpReplayScopeConfig(name='q8', apply_modules=['0'], quantizer=FP8_MODE)
        assert opr.OpReplayScope.from_config(cfg).quantizer is FP8

    def test_scope_uses_configured_fp8_block_size(self):
        cfg = OpReplayScopeConfig(name='q8', apply_modules=['0'],
                                  quantizer=FP8_MODE, fp8_block_size=192)

        scope = opr.OpReplayScope.from_config(cfg)

        assert scope.quantizer is not FP8
        assert isinstance(scope.quantizer, Fp8BlockwiseQuantizer)
        assert scope.quantizer.block_size == 192
