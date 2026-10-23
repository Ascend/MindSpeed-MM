"""Fused int8 per-row dequantization UTs (fsdp/ops/int8_dequant.py).

Covers the dispatch contract (native anti_quant op -> Triton kernel -> torch
composition, all three bitwise identical), the Triton kernels themselves
(masked vs exact, shape boundaries, launch-geometry legality for every planned
launch under every arch profile), and the fallback triggers (no Triton on the
host, unlaunchable geometry, payloads below the measured fuse crossover).

The tenant-facing layer around this op (Int8PerTokenQuantizer, the meta
record, the registry) is tested in
tests/ut_fsdp/features/memory/test_payload_quantizer.py.
"""
import builtins
import logging

import pytest
import torch

from mindspeed_mm.fsdp.ops import int8_dequant as ops_mod
from mindspeed_mm.fsdp.ops.int8_dequant import dequant_int8_per_row
from mindspeed_mm.fsdp.utils.device import IS_DEVICE_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(
    not IS_DEVICE_AVAILABLE,
    reason="swap requires a usable device (fast fail on CPU-only hosts)")

OPS_LOGGER = 'mindspeed_mm.fsdp.ops.int8_dequant'


def _get_device():
    return torch.device(get_device_type())


def _bits(t):
    """Raw-bit view of a bf16/fp16 tensor (torch.equal on values is not a
    bitwise comparison: it reports NaN != NaN and treats -0.0 == 0.0).
    """
    return t.reshape(-1).view(torch.int16)


def _assert_bitwise_equal(got, expected):
    assert got.dtype == expected.dtype and got.shape == expected.shape
    assert torch.equal(_bits(got), _bits(expected))


def _dequant_reference(q2d, scale, dtype):
    """Independent reference: exact int8 -> fp32, per-row multiply, cast back."""
    return (q2d.float() * scale[:, None]).to(dtype)


def _synthetic_case(rows, cols):
    """Payload + scales without the NPU quantize op, so shape boundaries can be
    probed independently of what npu_dynamic_quant accepts.
    """
    device = _get_device()
    q2d = torch.randint(-128, 128, (rows, cols), dtype=torch.int8, device=device)
    scale = torch.rand(rows, dtype=torch.float32, device=device) * 0.05 + 1e-4
    return q2d, scale


class _RaisingKernel:
    """Stands in for a jitted kernel: reaching it is a test failure, because a
    launch that the geometry check did not authorize can only crash the device.
    """

    def __getitem__(self, grid):
        raise AssertionError("kernel launched although the geometry check refused")


# The shapes the e2e profiler counted, with the rows-per-program that must keep
# grid dim 0 launchable (rows 524288 / 262144 exceed the 65535 limit).
REAL_E2E_SHAPES = ((16384, 8192), (16384, 9216), (16384, 4096), (524288, 128),
                   (16384, 2560), (524288, 64), (262144, 256), (16384, 1024),
                   (16384, 32))


class TestInt8PerRowDequant:
    """dequant_int8_per_row: bitwise equal to the reference on every tier

    The kernel is a pure re-association of the reference computation (int8 ->
    fp32 is exact, one multiply, one cast), so "same bits" is the right
    assertion, not a tolerance.
    """

    def test_bitwise_matches_reference_across_dtypes(self):
        for dtype in (torch.bfloat16, torch.float16):
            for rows, cols in ((128, 256), (1024, 4096), (24, 8)):
                q2d, scale = _synthetic_case(rows, cols)

                got = dequant_int8_per_row(q2d, scale, dtype)

                _assert_bitwise_equal(got, _dequant_reference(q2d, scale, dtype))
                _assert_bitwise_equal(got, ops_mod._dequantize_torch(q2d, scale, dtype))

    def test_bitwise_on_boundary_shapes(self):
        """cols=1 / 8192 / 100 (non power of two) / rows=1 / cols below and
        above BLOCK: the column mask and the per-row scale indexing must hold
        """
        for rows, cols in ((1024, 1), (64, 8192), (37, 100), (1, 4096),
                           (512, 32), (16, 5000)):
            q2d, scale = _synthetic_case(rows, cols)
            payload_copy = q2d.clone()

            got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

            _assert_bitwise_equal(got, _dequant_reference(q2d, scale, torch.bfloat16))
            _assert_bitwise_equal(
                got, ops_mod._dequantize_torch(q2d, scale, torch.bfloat16))
            assert torch.equal(q2d, payload_copy)  # payload never mutated

    def test_dispatch_tiers_agree_bitwise(self, monkeypatch):
        """native anti_quant -> Triton kernel -> torch composition: every
        dispatch tier must be indistinguishable in its output.
        """
        q2d, scale = _synthetic_case(128, 256)
        top = dequant_int8_per_row(q2d, scale, torch.float16)  # native op when probed OK

        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # native op off
        with_kernel = dequant_int8_per_row(q2d, scale, torch.float16)
        assert ops_mod._triton_dequant() is not None  # kernel path was taken

        monkeypatch.setattr(ops_mod, "_TRITON_DEQUANT", False)  # Triton off too
        fallback = dequant_int8_per_row(q2d, scale, torch.float16)

        _assert_bitwise_equal(fallback, with_kernel)
        _assert_bitwise_equal(fallback, top)
        _assert_bitwise_equal(fallback, _dequant_reference(q2d, scale, torch.float16))

    def test_falls_back_once_when_triton_is_missing(self, monkeypatch, caplog):
        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # reach Triton
        q2d, scale = _synthetic_case(64, 64)
        expected = dequant_int8_per_row(q2d, scale, torch.bfloat16)  # via the real kernel

        real_import = builtins.__import__

        def _no_triton(name, *args, **kwargs):
            if name == "triton" or name.startswith("triton."):
                raise ImportError("simulated: triton not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(ops_mod, "_TRITON_DEQUANT", None)  # re-probe
        monkeypatch.setattr(builtins, "__import__", _no_triton)
        with caplog.at_level(logging.WARNING, logger=OPS_LOGGER):
            got = dequant_int8_per_row(q2d, scale, torch.bfloat16)
            again = dequant_int8_per_row(q2d, scale, torch.bfloat16)

        assert ops_mod._TRITON_DEQUANT is False
        _assert_bitwise_equal(got, expected)
        _assert_bitwise_equal(again, expected)
        warns = [r for r in caplog.records if 'fused Triton dequant' in r.message]
        assert len(warns) == 1  # warned, but only once

    def test_native_op_is_preferred_when_probed(self, monkeypatch):
        """With the native op probed available, dequantize must not touch the
        Triton kernels (a raising stub stands guard)."""
        q2d, scale = _synthetic_case(128, 256)
        if ops_mod._anti_quant_dequant(q2d.device) is None:
            pytest.skip("npu_anti_quant with per-row scale unavailable on this stack")
        monkeypatch.setattr(ops_mod, "_TRITON_DEQUANT",
                            (object(), _RaisingKernel()))
        got = dequant_int8_per_row(q2d, scale, torch.bfloat16)
        _assert_bitwise_equal(got, _dequant_reference(q2d, scale, torch.bfloat16))

    def test_bitwise_on_real_e2e_shapes(self):
        """Every shape the e2e profiler counted, including the three row-heavy
        ones (524288x64 / 524288x128 / 262144x256) whose row-per-program grid
        exceeded the 65535 launch limit and crashed the device queue.
        """
        for rows, cols in REAL_E2E_SHAPES:
            q2d, scale = _synthetic_case(rows, cols)
            payload_copy = q2d.clone()

            got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

            _assert_bitwise_equal(got, _dequant_reference(q2d, scale, torch.bfloat16))
            _assert_bitwise_equal(
                got, ops_mod._dequantize_torch(q2d, scale, torch.bfloat16))
            assert torch.equal(q2d, payload_copy)  # payload never mutated

    def test_int64_offsets_above_the_int32_range(self, monkeypatch):
        """Payloads past 2**31 elements: offset arithmetic must not wrap.

        int32 offsets wrap to negative past 2**31 elements and silently read
        out of bounds — the geometry check proves grid legality but never
        bounds numel. (524416, 4096) exercises the exact kernel past the
        boundary, (524288, 4098) and (524287, 4098) the masked one (their
        narrow pow2 divisors floor to the 32-lane vector width, so these also
        pin the padded-tile geometry on wide payloads).
        """
        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # force the kernel
        if ops_mod._triton_dequant() is None:
            pytest.skip("Triton unavailable on this host")
        for rows, cols in ((524416, 4096), (524288, 4098), (524287, 4098)):
            assert rows * cols > 2**31
            q2d, scale = _synthetic_case(rows, cols)

            got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

            _assert_bitwise_equal(
                got, ops_mod._dequantize_torch(q2d, scale, torch.bfloat16))
            del q2d, scale, got
            torch.npu.empty_cache()

    def test_narrow_strips_floor_to_the_vector_width(self, monkeypatch):
        """cols whose largest pow2 divisor is below the 32-lane vector width
        widen to it and mask the tail: padded narrow tiles charge UB per row,
        not per element (see _kernel_geometry). The floored geometry must
        compile and stay bitwise correct; (524288, 1) covers the sub-vector
        branch (block stays 1, ROWS_PER_PROG capped by the padded tile).
        """
        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # force the kernel
        if ops_mod._triton_dequant() is None:
            pytest.skip("Triton unavailable on this host")
        profile = ops_mod._profile()
        for rows, cols, want_block in ((32768, 97, 32), (524288, 1, 1)):
            rpp, block, _ = ops_mod._kernel_geometry(rows, cols, profile)
            assert block == want_block
            assert rpp * max(block, 32) <= profile.ub_tile_elems
            q2d, scale = _synthetic_case(rows, cols)

            got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

            _assert_bitwise_equal(
                got, ops_mod._dequantize_torch(q2d, scale, torch.bfloat16))
            del q2d, scale, got
            torch.npu.empty_cache()

    def test_geometry_stays_launchable_for_every_planned_launch(self):
        """The launch error is asynchronous (uncatchable), so legality has to be
        proven up front: see the launch-geometry note in the module. Every arch
        profile is checked, not just this machine's — the geometry is pure
        arithmetic, so the other arch's profile is testable here.
        """
        for profile in ops_mod._ARCH_PROFILES.values():
            for rows, cols in REAL_E2E_SHAPES:
                geometry = ops_mod._kernel_geometry(rows, cols, profile)
                assert geometry is not None, (profile.name, rows, cols)
                rows_per_prog, block, exact = geometry
                assert rows_per_prog * block <= profile.ub_tile_elems
                # the PADDED tile fits too: narrow blocks pad up to the vector
                # width and their UB charge scales per row, not per element
                assert (rows_per_prog * max(block, ops_mod._VECTOR_LANES)
                        <= profile.ub_tile_elems)
                assert rows_per_prog * block >= profile.ub_tile_elems // 2  # budget-filling
                assert -(-rows // rows_per_prog) <= ops_mod._MAX_GRID_DIM
                assert -(-cols // block) <= ops_mod._MAX_GRID_DIM
                assert block >= 1 and rows_per_prog >= 1
                if exact:
                    assert rows % rows_per_prog == 0 and cols % block == 0
            # row-per-program (the geometry that crashed) is never chosen when
            # the row count exceeds the limit
            assert ops_mod._kernel_geometry(524288, 128, profile)[0] > 1
            # the tile fits the buffer the compiler charges it against, at the
            # charge that arch's compiler reported (its ub-overflow line)
            assert profile.ub_bytes_per_elem > 0 and profile.ub_budget_bytes > 0
            assert (profile.ub_tile_elems * profile.ub_bytes_per_elem
                    <= profile.ub_budget_bytes), profile.name
        # narrow rows take the exact strip at the vector width
        for profile in ops_mod._ARCH_PROFILES.values():
            narrow_rpp, narrow_block, narrow_exact = ops_mod._kernel_geometry(
                16384, 32, profile)
            assert narrow_block == 32 and narrow_exact is True, profile.name

    def test_geometry_denies_shapes_it_cannot_launch(self):
        # rows so large that even the widest allowed strip needs more rows per
        # program than the tile budget can hold (both profiles)
        for profile in ops_mod._ARCH_PROFILES.values():
            assert ops_mod._kernel_geometry(67108864, 32, profile) is None
            assert ops_mod._kernel_geometry(0, 64, profile) is None
            assert ops_mod._kernel_geometry(64, 0, profile) is None

    def test_exact_and_masked_kernels_agree_bitwise(self):
        """Both launch variants must produce the same bits, so the geometry may
        switch between them on divisibility alone.
        """
        kernels = ops_mod._triton_dequant()[1]
        for rows, cols, rows_per_prog, block in ((256, 512, 16, 512),
                                                 (1024, 128, 8, 128),
                                                 (512, 32, 16, 32)):
            q2d, scale = _synthetic_case(rows, cols)
            expected = _dequant_reference(q2d, scale, torch.bfloat16)
            device = q2d.device

            out = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
            kernels['exact'][(rows // rows_per_prog, cols // block)](
                q2d, out, scale, cols,
                ROWS_PER_PROG=rows_per_prog, BLOCK=block)
            _assert_bitwise_equal(out, expected)

            out_masked = torch.empty((rows, cols), dtype=torch.bfloat16, device=device)
            kernels['masked'][(-(-rows // rows_per_prog), -(-cols // block))](
                q2d, out_masked, scale, rows, cols,
                ROWS_PER_PROG=rows_per_prog, BLOCK=block)
            _assert_bitwise_equal(out_masked, expected)

    def test_over_limit_geometry_takes_the_torch_path_without_launching(
            self, monkeypatch):
        """An unlaunchable geometry must be detected before any launch: with the
        grid limit squeezed to 2, a small payload has no legal geometry and the
        torch path serves it — a kernel stub that explodes on indexing proves
        nothing was launched. The profile is pinned so the check means the same
        thing on an arch5 host (its wider tile does have a legal geometry here).
        """
        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # reach the kernel
        monkeypatch.setattr(ops_mod, "_PROFILE",
                            ops_mod._ARCH_PROFILES["ascend910_93"])
        q2d, scale = _synthetic_case(1024, 64)
        expected = dequant_int8_per_row(q2d, scale, torch.bfloat16)  # real kernel

        monkeypatch.setattr(ops_mod, "_MAX_GRID_DIM", 2)
        assert ops_mod._kernel_geometry(1024, 64) is None
        monkeypatch.setattr(ops_mod, "_TRITON_DEQUANT",
                            (object(), _RaisingKernel()))

        got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

        _assert_bitwise_equal(got, expected)
        _assert_bitwise_equal(got, _dequant_reference(q2d, scale, torch.bfloat16))

    def test_small_payloads_take_the_torch_path_where_measured_a_loss(
            self, monkeypatch):
        """Below profile.fuse_min_elems the kernel is launch-bound and loses to
        the composition (measured on 910_93 at 16384x32 = 524288 elements:
        0.7-0.9x), so those payloads must go to torch without launching either.
        """
        profile = ops_mod._ARCH_PROFILES["ascend910_93"]
        monkeypatch.setattr(ops_mod, "_ANTI_QUANT_DEQUANT", False)  # reach kernel/torch
        monkeypatch.setattr(ops_mod, "_PROFILE", profile)
        rows, cols = 16384, 32                                  # 524288 elements
        assert rows * cols < profile.fuse_min_elems
        q2d, scale = _synthetic_case(rows, cols)
        assert not ops_mod._kernel_applicable(q2d, scale)
        # a payload past the crossover still fuses
        big_q2d, big_scale = _synthetic_case(16384, 1024)       # 16.8M elements
        assert ops_mod._kernel_applicable(big_q2d, big_scale)

        monkeypatch.setattr(ops_mod, "_TRITON_DEQUANT",
                            (object(), _RaisingKernel()))
        got = dequant_int8_per_row(q2d, scale, torch.bfloat16)

        _assert_bitwise_equal(got, _dequant_reference(q2d, scale, torch.bfloat16))
        _assert_bitwise_equal(
            got, ops_mod._dequantize_torch(q2d, scale, torch.bfloat16))

    def test_arch_profile_resolution_and_override(self, monkeypatch):
        """The device picks the profile, MINDSPEED_MM_QUANT_PROFILE forces one
        (a smoke run on the other arch), an unknown value is ignored, and both
        tiles are the measured ones inside their arch's reported buffer.
        """
        saved = ops_mod._PROFILE
        try:
            assert ops_mod._use_profile("ascend910_95").name == "ascend910_95"
        finally:
            ops_mod._PROFILE = saved
        monkeypatch.setenv("MINDSPEED_MM_QUANT_PROFILE", "ascend910_93")
        assert ops_mod._detect_profile() == "ascend910_93"
        monkeypatch.setenv("MINDSPEED_MM_QUANT_PROFILE", "no-such-profile")
        assert ops_mod._detect_profile() in ops_mod._ARCH_PROFILES
        monkeypatch.delenv("MINDSPEED_MM_QUANT_PROFILE")
        assert ops_mod._detect_profile() in ops_mod._ARCH_PROFILES
        tiles = {name: prof.ub_tile_elems
                 for name, prof in ops_mod._ARCH_PROFILES.items()}
        assert tiles == {"ascend910_93": 16384, "ascend910_95": 32768}
