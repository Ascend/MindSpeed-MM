# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
"""Newton-Schulz orthogonalization for Muon.

Two variants share one entry point, ``_newton_schulz``:

- Standard Newton-Schulz runs in BF16. Each iteration applies
  ``X <- (a I + b A + c A^2) X`` with ``A = X X^T``, where X is the wide
  orientation (rows <= columns), so A is the small square side.
- Gram Newton-Schulz runs in FP16. For rectangular matrices it iterates on
  the small Gram matrix ``A = X X^T`` only: the per-iteration polynomials
  are multiplied into one square factor Q, and X is touched only at a few
  "reset" iterations, where ``X <- Q X`` and A is recomputed from X to stop
  FP16 rounding error from accumulating. This saves most of the products
  with the long side of X.

The result is returned in the compute dtype (BF16 or FP16) unless the caller
asks for another dtype: upcasting it would add no precision. Applying it to
an FP32 parameter with ``add_`` computes in FP32 by type promotion.
"""
import torch
import torch.distributed as dist

from .coefficients import resolve_ns_coefficients
from .kernels import _catlass_available, syrk_cost_factor, symmetric_kernels


# Gram Newton-Schulz is used only for these step counts, because its reset
# schedule was validated only for them. The Gram matrix is rebuilt from X at
# each iteration in GRAM_NS_RESET_ITERATIONS that is below the step count:
# 5 steps run as iterations [0, 2) [2, 5); 10 steps as [0, 2) [2, 5) [5, 7) [7, 10).
GRAM_NS_STEPS = (5, 10)
GRAM_NS_RESET_ITERATIONS = (2, 5, 7)


def uses_gram_newton_schulz(rows, columns, steps):
    """Whether a local matrix uses Gram Newton-Schulz.

    Only for 5 or 10 steps, and only when the long side is more than 1.5x the
    short side, where iterating on the small Gram matrix saves work.
    """
    short_side, long_side = sorted((int(rows), int(columns)))
    return steps in GRAM_NS_STEPS and 2 * long_side > 3 * short_side


def uses_column_sharded_gram_newton_schulz(rows, columns, steps):
    """Whether a column-sharded matrix can be orthogonalized without gathering it.

    Each rank holds complete rows over a slice of the columns. Summing the
    partial Gram matrices ``X_i X_i^T`` across ranks gives the full Gram matrix,
    which requires rows to be the short side (by more than 1.5x) and 5 or 10
    steps.
    """
    return steps in GRAM_NS_STEPS and 2 * int(columns) > 3 * int(rows)


def gram_newton_schulz_segments(steps):
    """Iteration counts between Gram resets, one entry per segment.

    5 steps run as (2, 3); 10 steps as (2, 3, 2, 3).
    """
    boundaries = (0, *(i for i in GRAM_NS_RESET_ITERATIONS if i < steps), steps)
    return tuple(end - begin for begin, end in zip(boundaries, boundaries[1:]))


def newton_schulz_flops(rows, columns, steps, device_type="cpu", batch=1):
    """Estimated FLOPs for one matrix's Newton-Schulz, used for owner load balancing.

    Counts the matrix products each variant issues. Standard NS reduces over
    the long side every iteration; Gram NS does so once per reset segment.
    Symmetric products count at half cost when CATLASS SYRK computes one
    triangle. ``factor @ X`` always counts in full. Elementwise and O(m²)
    work is omitted.
    """
    short_side, long_side = sorted((int(rows), int(columns)))
    steps = int(steps)
    # syrk(X) reduces over the long side; every Gram-matrix product is square.
    gram_build = (2 * short_side * short_side * long_side
                  * syrk_cost_factor(short_side, long_side, batch, device_type))
    square = (2 * short_side ** 3
              * syrk_cost_factor(short_side, short_side, batch, device_type))
    polynomial = (2 * short_side ** 3
                  * syrk_cost_factor(short_side, short_side, batch, device_type,
                                     polynomial=True))
    apply_to_x = 2 * short_side * short_side * long_side

    if not uses_gram_newton_schulz(short_side, long_side, steps):
        # Per iteration: syrk(X), syrk(gram, inner), then factor @ X. Standard
        # Newton-Schulz evaluates its polynomial inline, not with the fused
        # CATLASS polynomial kernel.
        return steps * (gram_build + square + apply_to_x)

    total = 0.0
    for length in gram_newton_schulz_segments(steps):
        # One Gram build and one application to X per segment. Per iteration:
        # the polynomial's Gram product, the accumulated factor (free on the
        # first iteration of a segment) and, except on the segment's last
        # iteration, two products to propagate the Gram matrix.
        total += gram_build + apply_to_x
        total += length * polynomial
        total += (length - 1) * 3 * square
    return total


# The host cost of issuing one Newton-Schulz call, in the same
# FLOP-equivalents units that newton_schulz_flops returns.
# An owner stacks same-shaped matrices into a single call, so this
# cost is paid once per call, and callers divide it by the batch size.
NEWTON_SCHULZ_CALL_FLOP_EQUIVALENT = {
    "catlass": 42.0e12,
    "torch": 57.1e12,
}


def newton_schulz_call_overhead(device_type="cpu"):
    """FLOP-equivalent host overhead of one Newton-Schulz call on this backend."""
    backend = "catlass" if _catlass_available(device_type) else "torch"
    return NEWTON_SCHULZ_CALL_FLOP_EQUIVALENT[backend]


def newton_schulz_cost(rows, columns, steps, device_type="cpu", batch=1):
    """Arithmetic plus this matrix's share of the per-call host overhead."""
    overhead = newton_schulz_call_overhead(device_type) / max(1, int(batch))
    return newton_schulz_flops(rows, columns, steps, device_type, batch) + overhead


def newton_schulz_dtype(rows, columns, steps):
    """Compute dtype, and default result dtype, of ``_newton_schulz`` for one local shape.

    FP16 for Gram Newton-Schulz, BF16 for standard Newton-Schulz. Every rank
    gets the same answer for the same shape, so exchange buffers can be
    allocated in this dtype even on ranks that compute nothing.
    """
    return torch.float16 if uses_gram_newton_schulz(rows, columns, steps) else torch.bfloat16


def _coefficients_at(schedule, iteration):
    """Repeat the final coefficient triple after the supplied schedule ends."""
    return schedule[min(iteration, len(schedule) - 1)]


def _normalize_for_newton_schulz(X, dtype=torch.bfloat16, process_group=None):
    """Normalize each matrix by its FP32 Frobenius norm, across shards if needed."""
    norm = torch.linalg.vector_norm(X, dim=(-2, -1), keepdim=True, dtype=torch.float32)
    if process_group is not None:
        norm.square_()
        dist.all_reduce(norm, op=dist.ReduceOp.SUM, group=process_group)
        norm.sqrt_()
    return X.mul(norm.add_(1e-7).reciprocal_()).to(dtype)


def _all_reduced_gram(X, process_group):
    """Sum the partial Gram matrices X_i X_i^T of all shards and return the result in the dtype of X.

    Each rank computes its partial product with an FP32 output, and the sum runs
    in FP32: summing FP16 partial Gram matrices loses more precision than the
    iteration tolerates. Upcasting an FP16 GEMM result would gain nothing, so
    the GEMM itself must produce FP32. The Gram matrix is only short side x
    short side, so the FP32 all-reduce is cheap.
    """
    gram = _gram_fp32(X)
    dist.all_reduce(gram, op=dist.ReduceOp.SUM, group=process_group)
    return gram.to(X.dtype)


def _gram_fp32(X):
    """Return X X^T in FP32 for a 2-D or batched 3-D X.

    On NPU, baddbmm with an FP32 accumulator runs the FP16/BF16 GEMM and writes
    an FP32 result, without FP32 copies of X. The CPU does not accept mixed
    dtypes there, so other devices upcast X first. Once the CATLASS SYRK kernel
    supports an FP32 output, it can compute this product instead.
    """
    if X.device.type != "npu":
        return torch.matmul(X.float(), X.float().mT)
    batch = X.reshape(-1, *X.shape[-2:])
    rows = batch.size(-2)
    gram = torch.baddbmm(batch.new_zeros(batch.size(0), rows, rows, dtype=torch.float32), batch, batch.mT)
    return gram.reshape(*X.shape[:-1], rows)


def _gram_iterations(X, steps, schedule, gram_product, symmetric_product,
                     polynomial, process_group=None):
    """Run Gram Newton-Schulz on a normalized FP16 X.

    Between resets, iterate on the Gram matrix only and multiply the
    per-iteration polynomials into one factor. At each reset, apply that
    factor to X and recompute the Gram matrix from X. With ``process_group``,
    X is this rank's column shard and the partial Gram matrices are summed.
    """
    if steps not in GRAM_NS_STEPS:
        raise ValueError("Gram Newton-Schulz supports only 5 or 10 steps")
    boundaries = (0, *(i for i in GRAM_NS_RESET_ITERATIONS if i < steps), steps)
    for begin, end in zip(boundaries, boundaries[1:]):
        gram = (gram_product(X) if process_group is None
                else _all_reduced_gram(X, process_group))
        factor = None
        for iteration in range(begin, end):
            a, b, c = _coefficients_at(schedule, iteration)
            update = polynomial(gram, a, b, c)
            factor = update if factor is None else symmetric_product(factor, update)
            if iteration + 1 < end:
                # Propagate the Gram matrix: new_gram = update @ gram @ update.
                # Uses symmetric_product (no transpose); see kernels.py.
                right_product = symmetric_product(gram, update)
                gram = symmetric_product(update, right_product)
        X = factor @ X
    return X


def _newton_schulz(G, steps, coefficients=None, out_dtype=None, *, process_group=None):
    """Orthogonalize one matrix or a batch of matrices ``[..., rows, columns]``.

    If ``process_group`` is set, G is this rank's column shard: the norm and
    the Gram matrix are summed across the group (Gram Newton-Schulz only).
    Returns the compute dtype (FP16 for Gram, BF16 for standard Newton-Schulz,
    see ``newton_schulz_dtype``) unless ``out_dtype`` is given.
    """
    _validate_steps(steps)
    distributed = process_group is not None
    if distributed and steps not in GRAM_NS_STEPS:
        raise ValueError("distributed Gram Newton-Schulz supports only 5 or 10 steps")
    use_gram = distributed or uses_gram_newton_schulz(*G.shape[-2:], steps)
    transposed = not distributed and G.size(-2) > G.size(-1)
    X = G.transpose(-2, -1) if transposed else G
    X = _normalize_for_newton_schulz(
        X, torch.float16 if use_gram else torch.bfloat16, process_group
    )
    schedule = resolve_ns_coefficients(coefficients)
    gram_product, symmetric_product, polynomial = symmetric_kernels(G.device.type)
    if use_gram:
        X = _gram_iterations(X, steps, schedule, gram_product, symmetric_product,
                             polynomial, process_group)
    else:
        for iteration in range(steps):
            a, b, c = _coefficients_at(schedule, iteration)
            gram = gram_product(X)
            inner = gram.mul(c)
            inner.diagonal(dim1=-2, dim2=-1).add_(b)
            factor = symmetric_product(gram, inner)
            factor.diagonal(dim1=-2, dim2=-1).add_(a)
            X = factor @ X
    X = X.transpose(-2, -1) if transposed else X
    return X if out_dtype is None else X.to(out_dtype)


def zeropower_via_newtonschulz5(G, steps, coefficients=None, out_dtype=None):
    """Orthogonalize a 2-D matrix with the quintic Newton-Schulz polynomial.

    Uses Gram Newton-Schulz in FP16 for 5 or 10 steps when the long side is
    more than 1.5x the short side, and standard Newton-Schulz in BF16
    otherwise, and returns that compute dtype unless ``out_dtype`` is given.
    The normalizing norm is computed in FP32. A schedule shorter than
    ``steps`` repeats its last coefficient triple.
    """
    if G.ndim != 2:
        raise ValueError(
            f"zeropower_via_newtonschulz5 expects a 2-D tensor, got shape {G.shape}"
        )
    return _newton_schulz(G, steps, coefficients, out_dtype)


def zeropower_via_newtonschulz5_per_head(
    G, num_heads=None, steps=5, coefficients=None, out_dtype=None
):
    """Orthogonalize each matrix in ``[num_heads, rows, columns]``."""
    _validate_block_layout(G, num_heads, "num_heads", steps, "per_head")
    return _newton_schulz(G, steps, coefficients, out_dtype)


def zeropower_via_newtonschulz5_per_expert(
    G, num_experts=None, steps=5, coefficients=None, out_dtype=None
):
    """Orthogonalize each matrix in ``[num_experts, rows, columns]``."""
    _validate_block_layout(G, num_experts, "num_experts", steps, "per_expert")
    return _newton_schulz(G, steps, coefficients, out_dtype)


def _validate_steps(steps):
    if not isinstance(steps, int) or steps < 0:
        raise ValueError(f"steps must be a non-negative integer, got {steps!r}")


def _validate_block_layout(G, count, count_name, steps, mode):
    """Require a 3-D tensor whose leading dimension is ``count`` blocks."""
    if G.ndim != 3:
        raise ValueError(
            f"zeropower_via_newtonschulz5_{mode} expects a 3-D tensor, "
            f"got shape {G.shape}"
        )
    _validate_steps(steps)
    actual = G.size(0)
    if count is None:
        count = actual
    if not isinstance(count, int) or count <= 0:
        raise ValueError(f"{count_name} must be a positive integer, got {count!r}")
    if count != actual:
        raise ValueError(
            f"{count_name} must match the first dimension ({actual}), got {count}"
        )
