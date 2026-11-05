"""Row fetch for the host-resident vocab embedding."""
import logging

import torch

from . import vocab_stream


logger = logging.getLogger(__name__)


def fetch_rows_from_host(
    weight: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    async_copy: bool = False,
):
    """Gather ``indices`` rows of the host-resident ``weight`` and return them on ``device``.

    Returns ``(rows, event)``. ``event`` is ``None`` for the blocking path; with ``async_copy`` the
    copy is issued on the side stream and ``event`` has to be waited on before ``rows`` is read (see
    ``vocab_stream.wait``). The caller must then consume the rows far enough after this call for the
    overlap to be worth anything -- arranging that is the job of the split dispatcher.
    """
    if weight.device.type != "cpu":
        raise RuntimeError(
            f"[VP] fetch_rows_from_host expects the vocab embedding in host memory, got {weight.device}. "
            f"The embedding must stay host-resident (see _no_placement_params / to_host_if_needed): a "
            f"device-resident embedding puts the engram layer back into accelerator memory."
        )

    host_indices = indices.to("cpu")
    rows = weight.shape[0]
    if host_indices.numel():
        lowest, highest = int(host_indices.min()), int(host_indices.max())
        if lowest < 0 or highest >= rows:
            raise IndexError(
                f"[VP] row id out of range for the vocab embedding held by this rank: [{lowest}, "
                f"{highest}] is not in [0, {rows}). Either the hashed ids and the embedding size "
                f"disagree (check ngram_vocab_size_base / heads on every rank), or a routed id "
                f"fell outside this rank's shard (check the vp window arithmetic)."
            )

    # ``index_select`` returns pageable memory, whose host-to-device copy is ~25x slower than from
    # pinned memory, so the rows are gathered into pinned staging (host-to-host, then one DMA).
    try:
        staging = torch.empty(
            (host_indices.numel(), *weight.shape[1:]), dtype=weight.dtype, pin_memory=True
        )
    except Exception as exc:
        logger.warning(
            "[VP] pinned staging allocation failed (%s); falling back to a pageable row gather", exc
        )
        return weight.index_select(0, host_indices).to(device), None
    torch.index_select(weight, 0, host_indices, out=staging)
    if not async_copy:
        # Blocking on purpose: the caller casts these rows and hands them to the embedding all-to-all
        # in the same call, so there is no later consumer that could wait for this copy.
        return staging.to(device), None
    # The copy runs on the side stream while the layers ahead of the consumer progress; the pinned
    # staging buffer is released by the consumer's wait, or by the host wait at the end of the step.
    with vocab_stream.issue():
        out = staging.to(device, non_blocking=vocab_stream.enabled())
    vocab_stream.keep(staging)
    return out, vocab_stream.mark()
