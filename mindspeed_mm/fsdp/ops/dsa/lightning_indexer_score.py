# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import triton
from mindspeed_mm.fsdp.ops.dsa.triton.lightning_indexer_score import (
    _lightning_indexer_score_kernel,
)
from mindspeed_mm.fsdp.ops.gdn.triton.utils import is_arch35

__all__ = ["LightningIndexerScoreFunction", "lightning_indexer_score"]

# tl.dot needs at least one cube tile on both the M and N axes.
_MIN_TILE = 16
_MAX_BLOCK_S = 64
# BP = 512 faults the device (npuSynchronizeDevice error), not just a UB overflow.
_MAX_BLOCK_P = 256
# Largest BS * BP that still compiles on arch32 (UB = 192 KiB).
_MAX_TILE_ELEMS = 16384
# Ascend910_9382 has 48 vector cores; keep them busy on short sequences.
_MIN_PROGRAMS = 48


def _pick_tiles(seq_len: int, num_pools: int, batch_size: int = 1) -> tuple:
    """Pick (BS, BP) for one program.

    Measured on Ascend910_9382 (UB 192 KiB -- the compiler reports the budget as
    ``1572864`` *bits*): ``BS * BP <= 16384`` is the point where the BiShengIR
    pipeline still fits the live tiles on chip; the next step up, (128, 256),
    fails in ConvertLinalgRToBinary. ``BP = 512`` faults the device outright, so
    it is capped at ``_MAX_BLOCK_P``.

    Within that budget a larger BP is always better -- q is re-read once per
    pool block, so widening BP directly cuts global memory traffic. A tile sweep
    at S=5304, P=1326 gave 7.0 / 2.6 / 1.9 / 1.6 / 1.26 ms for
    (64,64) / (64,128) / (32,256) / (128,128) / (64,256), i.e. BP dominates and
    (64, 256) wins. That last one needs the 16384 budget: the earlier 8192 cap
    forced (32, 256) and cost ~1.5x.

    Neither axis goes below ``_MIN_TILE``: they are the M and N axes of
    ``tl.dot``, which degenerates below one cube tile. A ragged sequence tail is
    left to ``boundary_check``; a ragged pool tail is padded away by the caller
    (see the note in ``forward``).
    """
    block_p = _MAX_BLOCK_P
    while block_p > _MIN_TILE and block_p >= num_pools * 2:
        block_p //= 2
    block_s = _MAX_BLOCK_S
    while block_s > _MIN_TILE and block_s >= seq_len * 2:
        block_s //= 2
    while block_s > _MIN_TILE and block_s * block_p > _MAX_TILE_ELEMS:
        block_s //= 2
    # Short sequences otherwise launch fewer programs than there are vector
    # cores and leave most of the device idle.
    n_p = -(-num_pools // block_p)
    while block_s > _MIN_TILE and batch_size * (-(-seq_len // block_s)) * n_p < _MIN_PROGRAMS:
        block_s //= 2
    return block_s, block_p


class LightningIndexerScoreFunction(torch.autograd.Function):
    """Fused scoring stage of the DeepSeek sparse-attention (DSA) indexer.

    forward args:
    q:                [B, S, NH, D]   indexer queries (after wq_b + rope)
    pool_keys:        [B, P, D]       compressed k-pool keys
    weights:          [B, S, NH]      per-head weights, already scaled by NH**-0.5
    valid_candidates: [B, S, P] bool  optional; masked-out pools become ``neg_inf``
    scale:            float           softmax_scale, usually D**-0.5

    Returns:
    index_scores:     [B, S, P]  float32

    The indexer runs under ``torch.no_grad`` in the model, so no backward is
    provided -- calling backward raises.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        pool_keys: torch.Tensor,
        weights: torch.Tensor,
        valid_candidates: torch.Tensor = None,
        scale: float = None,
    ):
        if q.dim() != 4:
            raise ValueError(f"q must be [B, S, NH, D], got {tuple(q.shape)}")
        if pool_keys.dim() != 3:
            raise ValueError(f"pool_keys must be [B, P, D], got {tuple(pool_keys.shape)}")
        if weights.dim() != 3:
            raise ValueError(f"weights must be [B, S, NH], got {tuple(weights.shape)}")

        batch_size, seq_len, num_heads, head_dim = q.shape
        num_pools = pool_keys.shape[1]
        if pool_keys.shape[0] != batch_size or pool_keys.shape[2] != head_dim:
            raise ValueError(
                f"pool_keys {tuple(pool_keys.shape)} does not match q {tuple(q.shape)}"
            )
        if weights.shape[:2] != (batch_size, seq_len) or weights.shape[2] != num_heads:
            raise ValueError(
                f"weights {tuple(weights.shape)} does not match q {tuple(q.shape)}"
            )
        if head_dim & (head_dim - 1):
            raise ValueError(f"head_dim must be a power of two, got {head_dim}")

        if is_arch35():
            raise NotImplementedError("this op is not supported in this platform")

        if scale is None:
            scale = head_dim ** -0.5

        # tl.dot wants both operands in the same dtype. It is enough to make them
        # agree -- the instruction multiplies in the input dtype but accumulates
        # the D-wide reduction in fp32 regardless (see the note in the kernel), so
        # promoting further would only cost bandwidth.
        common_dtype = torch.promote_types(q.dtype, pool_keys.dtype)
        q = q.to(common_dtype).contiguous()
        pool_keys = pool_keys.to(common_dtype).contiguous()
        weights = weights.contiguous()

        has_valid = valid_candidates is not None
        if has_valid:
            if tuple(valid_candidates.shape) != (batch_size, seq_len, num_pools):
                raise ValueError(
                    f"valid_candidates must be [B, S, P], got {tuple(valid_candidates.shape)}"
                )
            valid_candidates = valid_candidates.contiguous()
        else:
            valid_candidates = q  # unused placeholder, kernel guards on HAS_VALID

        if batch_size * seq_len * num_pools == 0:
            return torch.empty(
                (batch_size, seq_len, num_pools), dtype=torch.float32, device=q.device
            )

        block_s, block_p = _pick_tiles(seq_len, num_pools, batch_size)

        # `boundary_check` does not reliably clamp the MTE read on the pool axis:
        # some ragged tails (e.g. P=642 with BP=256) fault with an aicore MTE
        # error while others of the same shape class do not, i.e. it depends on
        # where the over-read lands. Pad the pool axis instead so every tile is
        # whole. Padded keys are zero, padded candidates invalid, and the extra
        # columns are sliced off before returning.
        pad = (-num_pools) % block_p
        pools_padded = num_pools + pad
        if pad:
            pool_keys = torch.nn.functional.pad(pool_keys, (0, 0, 0, pad))
            if has_valid:
                valid_candidates = torch.nn.functional.pad(valid_candidates, (0, pad))

        out = torch.empty(
            (batch_size, seq_len, pools_padded), dtype=torch.float32, device=q.device
        )
        grid = (triton.cdiv(seq_len, block_s), pools_padded // block_p, batch_size)
        _lightning_indexer_score_kernel[grid](
            q,
            pool_keys,
            weights,
            valid_candidates,
            out,
            seq_len,
            pools_padded,
            float(scale),
            float(torch.finfo(torch.float32).min),
            num_heads,
            head_dim,
            block_s,
            block_p,
            has_valid,
        )
        return out[:, :, :num_pools] if pad else out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        raise RuntimeError(
            "lightning_indexer_score is forward-only; the DSA indexer runs under torch.no_grad()."
        )


def lightning_indexer_score(
    q: torch.Tensor,
    pool_keys: torch.Tensor,
    weights: torch.Tensor,
    valid_candidates: torch.Tensor = None,
    scale: float = None,
) -> torch.Tensor:
    """Functional entry. See :class:`LightningIndexerScoreFunction`."""
    return LightningIndexerScoreFunction.apply(q, pool_keys, weights, valid_candidates, scale)
