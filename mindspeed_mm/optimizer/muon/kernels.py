# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Select CATLASS or PyTorch kernels for the Newton-Schulz matrix products.

On Ascend NPUs with ``torch_catlass`` installed, the symmetric products use
CATLASS SYRK kernels for Ascend 950PR and 950DT series products. For other devices or shapes where CATLASS
is slower, standard ``torch.matmul`` is used instead.
"""

from functools import lru_cache
import logging


logger = logging.getLogger(__name__)

_REQUIRED_KERNELS = ("ascend950_basic_syrk", "ascend950_syrk_polynomial")

# Shape thresholds for using CATLASS instead of torch.matmul. A cost model
# indicates where computing one triangle should win; these values were
# calibrated against one specific CATLASS build, so another build may need
# other values. "rows" is the output side (the short side of the Newton-Schulz
# input), "batch" the number of stacked matrices, "columns" the reduced side.
# - Fewer than CATLASS_SYRK_FLOOR_ROWS rows: always PyTorch.
# - At least CATLASS_SYRK_MIN_ROWS rows: always CATLASS.
# - At least CATLASS_SYRK_TILE_ROWS rows: CATLASS for large batches or for
#   at most CATLASS_SYRK_NARROW_COLUMNS columns.
# - Otherwise: CATLASS only for large batches of wide matrices.
CATLASS_SYRK_FLOOR_ROWS = 32
CATLASS_SYRK_MIN_ROWS = 512
CATLASS_SYRK_TILE_ROWS = 256
CATLASS_SYRK_MIN_BATCH = 32
CATLASS_SYRK_WIDE_COLUMNS = 16384
CATLASS_SYRK_NARROW_COLUMNS = 8192


def _use_catlass_syrk_shape(rows, columns, batch):
    """Whether this shape and batch size favor the CATLASS SYRK kernel."""
    if rows < CATLASS_SYRK_FLOOR_ROWS or columns == 0:
        return False
    if rows >= CATLASS_SYRK_MIN_ROWS:
        return True
    if rows >= CATLASS_SYRK_TILE_ROWS:
        return batch >= CATLASS_SYRK_MIN_BATCH or columns <= CATLASS_SYRK_NARROW_COLUMNS
    return batch >= CATLASS_SYRK_MIN_BATCH and columns >= CATLASS_SYRK_WIDE_COLUMNS


def _use_catlass_syrk(tensor):
    """Whether the matrix shape and batch size favor the CATLASS SYRK kernel."""
    rows, columns = tensor.shape[-2:]
    batch = tensor.size(0) if tensor.ndim == 3 else 1
    return _use_catlass_syrk_shape(rows, columns, batch)


@lru_cache(maxsize=None)
def _catlass_available(device_type):
    """Whether CATLASS SYRK is available on this device type."""
    if device_type != "npu":
        return False
    try:
        import torch_catlass
    except (ImportError, OSError):
        return False
    return all(
        callable(getattr(torch_catlass, name, None)) for name in _REQUIRED_KERNELS
    )


def syrk_cost_factor(rows, columns, batch, device_type, polynomial=False):
    """1.0 for full GEMM cost, 0.5 when CATLASS SYRK computes one triangle."""
    if not _catlass_available(device_type):
        return 1.0
    if polynomial and rows % 32:
        return 1.0
    return 0.5 if _use_catlass_syrk_shape(rows, columns, batch) else 1.0


def _torch_gram_product(left):
    """``left @ left.T``: the transpose is inherent to the Gram construction."""
    return left @ left.transpose(-2, -1)


def _torch_symmetric_product(left, right):
    """``left @ right`` for square operands that are symmetric in exact arithmetic.

    Does not transpose ``right``. ``left @ right.T`` is algebraically equivalent
    but amplifies the antisymmetric rounding drift across iterations.
    """
    return left @ right


def _torch_polynomial(gram, a, b, c):
    inner = gram.mul(c)
    inner.diagonal(dim1=-2, dim2=-1).add_(b)
    result = gram @ inner
    result.diagonal(dim1=-2, dim2=-1).add_(a)
    return result


@lru_cache(maxsize=None)
def symmetric_kernels(device_type):
    """Return ``(gram_product, symmetric_product, polynomial)``, cached.

    ``gram_product(X)`` computes ``X @ X.T`` for rectangular ``X``.
    ``symmetric_product(left, right)`` computes ``left @ right`` (no transpose)
    for square operands that are symmetric in exact arithmetic. CATLASS SYRK
    uses its own ``left @ right.T`` form, which is safe because it mirrors
    the lower triangle and keeps the operands exactly symmetric.
    ``polynomial(gram, a, b, c)`` returns ``a I + b G + c G²``.
    All three preserve the input dtype.
    """
    torch_kernels = (_torch_gram_product, _torch_symmetric_product, _torch_polynomial)
    if device_type != "npu":
        return torch_kernels
    try:
        import torch_catlass
    except (ImportError, OSError):
        logger.warning("Muon: torch_catlass is unavailable; using Torch kernels")
        return torch_kernels

    if not _catlass_available(device_type):
        logger.warning(
            "Muon: torch_catlass does not support the symmetric SYRK kernels; "
            "using Torch kernels"
        )
        return torch_kernels
    syrk_kernel = getattr(torch_catlass, "ascend950_basic_syrk", None)
    polynomial_kernel = getattr(torch_catlass, "ascend950_syrk_polynomial", None)
    gram_product, symmetric_product = _torch_gram_product, _torch_symmetric_product
    polynomial = _torch_polynomial
    if callable(syrk_kernel):
        def gram_product(left):
            if not _use_catlass_syrk(left):
                return _torch_gram_product(left)
            return syrk_kernel(left, None, left.dtype)

        def symmetric_product(left, right):
            if not _use_catlass_syrk(left):
                return _torch_symmetric_product(left, right)
            return syrk_kernel(left, right, left.dtype)
    if callable(polynomial_kernel):
        def polynomial(gram, a, b, c):
            if (gram.size(-2) % 32 or not _use_catlass_syrk(gram)
                    or not gram.is_contiguous()):
                return _torch_polynomial(gram, a, b, c)
            return polynomial_kernel(gram, gram, gram, c, b, a, gram.dtype)
    return gram_product, symmetric_product, polynomial
