"""Fused int8 per-row dequantization for the swap cache's payload quantizer.

One entry point: dequant_int8_per_row(q2d, scale, dtype) — a contiguous int8
(rows, cols) payload plus one fp32 scale per row, back to (rows, cols) in the
original dtype, on the caller's stream. Dispatch over three bitwise-identical
paths: the native npu_anti_quant (per-process probe — op registration alone
says nothing about per-row scale support), then a fused Triton kernel, then
the equivalent torch composition. Triton is imported lazily and must not
become a module-level dependency.

The tenant-facing contract (PayloadQuantizer, the meta record, QUANTIZERS)
lives in mindspeed_mm/fsdp/features/memory/quantizer.py.
"""
from dataclasses import dataclass
import logging
import os

import torch

from mindspeed_mm.fsdp.log import print_rank


logger = logging.getLogger(__name__)

# ---- Fused dequant launch geometry ----------------------------------------
# Forward-compatibility guard: old stacks (910_93 / CANN 9.0) reject a grid
# dim over 65535 asynchronously and fatally (uncatchable from Python), so the
# geometry is proven legal before launching; 910_95 / CANN 9.2 lifts the limit.
_MAX_GRID_DIM = 65535
# The compiler pads a narrower BLOCK up to this width, turning the tile's UB
# charge from per-element into per-row (see _kernel_geometry).
_VECTOR_LANES = 32
# ---- Per-architecture budget and geometry rules ---------------------------
# Everything in a profile is measured on the arch itself, never extrapolated.
# The compiler's "ub overflow, requires N bits while M bits available" error
# yields the budget (M) and per-element charge (N/tile) directly:
#   * 910_93: 192 KB buffer, 7 B/element (int8 + fp32 product + bf16 + mask).
#   * 910_95: 248 KB buffer, 3 B/element (int8 + bf16; fp32 product stays off
#     the unified buffer).
# The tile in use is the measured optimum, not the largest that compiles
# (910_93: 16384 of 28079 modelled elements; 910_95: 32768 of 84650).
@dataclass(frozen=True)
class _ArchProfile:
    """The fused dequant's budget and geometry rules for one arch family."""
    name: str
    ub_tile_elems: int        # tile that measures fastest (not the ceiling)
    ub_bytes_per_elem: int    # what the compiler charges per tile element
    ub_budget_bytes: int      # the buffer it charges against (its reported M)
    block_cap: int            # widest column strip an exact kernel may use
    fuse_min_elems: int       # smaller payloads take the torch path (0 = always)

_ARCH_PROFILES = {
    # 910B/910_93. Wide strips are fine (block 2048 measures 0.6% ahead of 256
    # at the same tile area). Below the torch/triton crossover the fused kernel
    # loses (16384x32 = 0.5M elements: 0.7-0.9x; 16384x1024 = 16.8M: 1.5x), so
    # fuse_min_elems sits at the conservative 1M.
    "ascend910_93": _ArchProfile("ascend910_93", 16384, 7, 192 * 1024,
                                 2048, 1 << 20),
    # 910_95/950. Column strips >= 512 collapse (1.7x), 256 is the measured
    # optimum. The fused kernel wins from the smallest measured payload up
    # (16384x32: 1.65x), so it always fuses.
    "ascend910_95": _ArchProfile("ascend910_95", 32768, 3, 248 * 1024,
                                 256, 0),
}
# Unknown devices (and hosts without an NPU) get the conservative arch3
# profile: its tile fits either UB.
_DEFAULT_PROFILE = "ascend910_93"
_PROFILE_ENV = "MINDSPEED_MM_QUANT_PROFILE"
# None = not resolved yet; lazy because this module is imported on hosts that
# may have no NPU.
_PROFILE = None


def _detect_profile() -> str:
    """Profile name for this device: A5 (Ascend910_95*/Ascend950*) has its own,
    everything else takes the arch3 one. Same device-name test as the other A5
    branches in the repo; MINDSPEED_MM_QUANT_PROFILE overrides it (smoke tests
    on a machine of the other arch)."""
    forced = os.environ.get(_PROFILE_ENV, "").strip()
    if forced:
        if forced in _ARCH_PROFILES:
            return forced
        logger.warning("swap payload quantizer: ignoring %s=%r, known profiles: %s",
                       _PROFILE_ENV, forced, ", ".join(sorted(_ARCH_PROFILES)))
    try:
        import torch_npu
        device_name = torch_npu.npu.get_device_name()
    except Exception:
        return _DEFAULT_PROFILE
    if "Ascend910_95" in device_name or "Ascend950" in device_name:
        return "ascend910_95"
    return _DEFAULT_PROFILE


def _profile() -> _ArchProfile:
    """The resolved profile, cached — the geometry is recomputed on every pop."""
    global _PROFILE
    if _PROFILE is None:
        _PROFILE = _ARCH_PROFILES[_detect_profile()]
    return _PROFILE


def _use_profile(name):
    """Pin a profile by name (tests, smoke runs); None re-resolves from device."""
    global _PROFILE
    _PROFILE = None if name is None else _ARCH_PROFILES[name]
    return _profile()

# None = not probed yet, False = Triton unavailable, else (triton, kernels).
# Triton is a heavy optional dependency — never a module-level import.
_TRITON_DEQUANT = None

# None = not probed yet, False = unusable, else the op itself. Registration
# says nothing about per-row scale support, so the probe runs the real call
# and checks the bits.
_ANTI_QUANT_DEQUANT = None


def _anti_quant_dequant(device: torch.device):
    """torch.ops.npu.npu_anti_quant, or None when this stack cannot run the
    exact production call (int8 (rows, cols), per-row fp32 scales, bf16/fp16).

    The probe issues the real call on tiny tensors, synchronizes (a launch
    failure can surface asynchronously), and bitwise-compares against the
    reference composition. RNG-free by contract; probed once per process.
    """
    global _ANTI_QUANT_DEQUANT
    if _ANTI_QUANT_DEQUANT is None:
        _ANTI_QUANT_DEQUANT = False
        npu_ops = getattr(torch.ops, 'npu', None)
        op = getattr(npu_ops, 'npu_anti_quant', None) if npu_ops is not None else None
        if op is not None:
            try:
                q = (torch.arange(3 * 17, device=device).reshape(3, 17)
                     % 255 - 127).to(torch.int8)
                s = torch.arange(1, 4, device=device, dtype=torch.float32) / 7
                for dtype in (torch.bfloat16, torch.float16):
                    out = op(q, s, dst_dtype=dtype)
                    torch.npu.synchronize(device)
                    if not torch.equal(out, (q.float() * s[:, None]).to(dtype)):
                        return None
                _ANTI_QUANT_DEQUANT = op
            except Exception as err:
                logger.debug("swap payload quantizer: npu_anti_quant probe "
                             "failed, fused dequant will not use it: %r", err)
    return _ANTI_QUANT_DEQUANT or None


def _disable_triton_dequant(reason) -> None:
    """Drop the fused kernel for the rest of the process and say so once —
    a silent fallback would hide a broken environment."""
    global _TRITON_DEQUANT
    _TRITON_DEQUANT = False
    print_rank(
        logger.warning,
        f"swap payload quantizer: fused Triton dequant is unavailable "
        f"({reason}); dequantizing with the torch composition instead "
        f"(correct, but slower).",
        ranks=0)


# Bound to triton.language by _triton_dequant() on first use: the kernels
# below are Triton source whose `tl.*` names resolve in this module's globals
# at JIT time.
tl = None


def _dequant_int8_rows(Q, Y, S, ROWS, COLS,
                       ROWS_PER_PROG: "tl.constexpr", BLOCK: "tl.constexpr"):
    """Fused int8 per-row dequant tile (masked): ROWS_PER_PROG rows x BLOCK
    columns per program — per-program overhead tracks program count, not byte
    count, so a row-per-program grid would both exhaust the grid limit and
    waste the UB.

    rows/cols are int64: int32 offsets wrap past 2**31 elements into silent
    out-of-bounds reads. Measured on 910_95: per-shape time ratios 0.99-1.02
    vs int32 addressing — no consistent cost.

    Triton source, not a callable: wrapped in triton.jit() on first use (see
    _triton_dequant).
    """
    rows = (tl.program_id(0) * ROWS_PER_PROG
            + tl.arange(0, ROWS_PER_PROG)).to(tl.int64)
    cols = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    m = (rows[:, None] < ROWS) & (cols[None, :] < COLS)
    q = tl.load(Q + rows[:, None] * COLS + cols[None, :], mask=m, other=0)
    s = tl.load(S + rows, mask=rows < ROWS, other=0.0)
    tl.store(Y + rows[:, None] * COLS + cols[None, :],
             (q.to(tl.float32) * s[:, None]).to(Y.dtype.element_ty), mask=m)


def _dequant_int8_rows_exact(Q, Y, S, COLS,
                             ROWS_PER_PROG: "tl.constexpr", BLOCK: "tl.constexpr"):
    """Maskless twin of _dequant_int8_rows for exact tiles (ROWS_PER_PROG | rows,
    BLOCK | cols, guaranteed by _kernel_geometry) — measured ~5-15% faster on
    wide shapes, more on narrow ones."""
    rows = (tl.program_id(0) * ROWS_PER_PROG
            + tl.arange(0, ROWS_PER_PROG)).to(tl.int64)
    cols = (tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    q = tl.load(Q + rows[:, None] * COLS + cols[None, :])
    s = tl.load(S + rows)
    tl.store(Y + rows[:, None] * COLS + cols[None, :],
             (q.to(tl.float32) * s[:, None]).to(Y.dtype.element_ty))


def _triton_dequant():
    """Resolve (triton, kernels) lazily, or None when Triton cannot be used."""
    global _TRITON_DEQUANT
    if _TRITON_DEQUANT is None:
        try:
            import triton
            import triton.language as tl

            # triton.jit resolves `tl` in this module's globals at JIT time and
            # on every new specialization, so it must stay visible there.
            globals()['tl'] = tl
            _TRITON_DEQUANT = (triton, {
                'masked': triton.jit(_dequant_int8_rows),
                'exact': triton.jit(_dequant_int8_rows_exact),
            })
        except Exception as err:
            _disable_triton_dequant(f"import failed: {err}")
    return _TRITON_DEQUANT or None


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length()


def _largest_pow2_divisor(n: int, cap: int) -> int:
    """Largest power of two that divides n, capped (0 when none fits the cap)."""
    power = 1
    while power * 2 <= min(cap, n) and n % (power * 2) == 0:
        power *= 2
    return power


def _kernel_geometry(rows: int, cols: int, profile=None):
    """(ROWS_PER_PROG, BLOCK, exact) for the fused kernel, or None when no
    geometry is both launchable and UB-safe.

    BLOCK is the widest power-of-two divisor of cols up to profile.block_cap
    (exact kernel, no tail lanes), floored at _VECTOR_LANES: narrower strips
    pad to the vector width and charge UB per row, so a tall narrow tile
    overflows at compile time even when the data fits. ROWS_PER_PROG absorbs
    the remaining budget, bounded by _MAX_GRID_DIM and the padded-tile budget.
    exact selects the maskless kernel.
    """
    prof = profile if profile is not None else _profile()
    if rows <= 0 or cols <= 0:
        return None
    block = _largest_pow2_divisor(cols, prof.block_cap)
    if block < _VECTOR_LANES:
        block = _VECTOR_LANES if cols >= _VECTOR_LANES else _next_pow2(cols)
    block = min(block, prof.ub_tile_elems)
    rows_per_prog = max(1, min(prof.ub_tile_elems // block, _next_pow2(rows)))
    # Few rows in play: widen the strip to fill the budget.
    block = min(prof.ub_tile_elems, max(block, prof.ub_tile_elems // rows_per_prog))
    # Grid dim 0 = ceil(rows / ROWS_PER_PROG) must stay launchable.
    rows_per_prog = max(rows_per_prog, _next_pow2(-(-rows // _MAX_GRID_DIM)))
    # Cap ROWS_PER_PROG so the padded tile stays within budget.
    rows_per_prog = min(rows_per_prog,
                        max(1, prof.ub_tile_elems // max(block, _VECTOR_LANES)))
    # Grid dim 1 = ceil(cols / BLOCK) is bounded by widening the strip.
    while -(-cols // block) > _MAX_GRID_DIM and block < prof.ub_tile_elems:
        block *= 2
    if rows_per_prog * block > prof.ub_tile_elems:
        return None
    if (-(-rows // rows_per_prog) > _MAX_GRID_DIM
            or -(-cols // block) > _MAX_GRID_DIM):
        return None
    exact = (rows % rows_per_prog == 0 and cols % block == 0)
    return rows_per_prog, block, exact


def _kernel_applicable(q2d: torch.Tensor, scale: torch.Tensor) -> bool:
    """Pre-launch pre-conditions of the fused kernel, or False to take the
    torch path: 2D contiguous int8 payload, contiguous fp32 per-row scales, a
    launchable geometry, and a payload above profile.fuse_min_elems.
    """
    if q2d.dim() != 2:
        return False
    rows, cols = q2d.shape
    if rows <= 0 or cols <= 0:
        return False
    profile = _profile()
    if rows * cols < profile.fuse_min_elems:
        # Below this the fused kernel is launch-bound and loses to the
        # composition's cheap launches (910_93: 0.5M elements -> 0.7-0.9x,
        # 16.8M -> 1.5x).
        return False
    if _kernel_geometry(rows, cols, profile) is None:
        return False
    return (q2d.is_contiguous() and q2d.dtype is torch.int8
            and scale.is_contiguous() and scale.dtype is torch.float32
            and scale.numel() == rows)


def _dequantize_torch(q2d: torch.Tensor, scale: torch.Tensor,
                      dtype: torch.dtype) -> torch.Tensor:
    """Torch composition: cast, scale, cast back — three kernels and a
    4-byte-per-element fp32 intermediate, but bitwise identical to the fused
    kernel and Triton-free. The fallback path."""
    # int8 x fp32 promotes to fp32; cast back so the dtype matches put's.
    return (q2d.to(dtype) * scale.reshape(-1, 1)).to(dtype)


def dequant_int8_per_row(q2d: torch.Tensor, scale: torch.Tensor,
                         dtype: torch.dtype) -> torch.Tensor:
    """int8 per-row dequant of a 2D payload: q2d (rows, cols) int8 + one fp32
    scale per row -> (rows, cols) in dtype, on the caller's stream.

    Dispatch: native npu_anti_quant (when probed usable), then the fused
    Triton kernel, then the torch composition — all bitwise identical.
    """
    anti_quant = _anti_quant_dequant(q2d.device)
    if anti_quant is not None:
        return anti_quant(q2d, scale, dst_dtype=dtype)
    resolved = _triton_dequant()
    if resolved is not None and _kernel_applicable(q2d, scale):
        rows, cols = q2d.shape
        kernels = resolved[1]
        rows_per_prog, block, exact = _kernel_geometry(rows, cols)
        # Fresh whole contiguous tensor, on the caller's (consume) stream.
        out = torch.empty(q2d.shape, dtype=dtype, device=q2d.device)
        # Geometry proven launchable above; a launch error would surface
        # asynchronously, so there is nothing useful to catch here.
        grid = (-(-rows // rows_per_prog), -(-cols // block))
        if exact:
            kernels['exact'][grid](q2d, out, scale, cols,
                                   ROWS_PER_PROG=rows_per_prog, BLOCK=block)
        else:
            kernels['masked'][grid](q2d, out, scale, rows, cols,
                                    ROWS_PER_PROG=rows_per_prog, BLOCK=block)
        return out
    return _dequantize_torch(q2d, scale, dtype)
