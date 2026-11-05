"""Sparse optimizer of the host-resident vocab embeddings."""
import logging

import torch
from torch.distributed.tensor import DTensor

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.log import print_rank

from .vocab_sparse_grad import get_device_vocab_embeddings, get_vocab_embeddings


logger = logging.getLogger(__name__)


def build_vocab_optimizer(model, lr: float = 1e-3, betas=(0.9, 0.95), eps: float = 1e-8, backend: str = "cpu"):
    """SparseAdam over the host-resident embeddings of ``model``, or ``None`` when it holds none."""
    if backend != "cpu":
        raise NotImplementedError(f"vocab optimizer backend {backend} is not implemented now.")
    weights = [embedding.weight for _, embedding in get_vocab_embeddings(model)]
    if not weights:
        return None
    elements = sum(weight.numel() for weight in weights)
    print_rank(
        logger.info,
        f"[Vocabulary Parallel] vocab optimizer: SparseAdam over {len(weights)} embedding(s), "
        f"host fp32 moments up to {elements * 8 / 1024 ** 3:.1f} GiB (created on the first step)",
    )
    return torch.optim.SparseAdam(weights, lr=lr, betas=betas, eps=eps)


def finalize_vocab_gradients(model) -> None:
    """Materialize every embedding's step gradient: merge its rows and sum the row-window replicas.

    Call this once per step on every rank before the norm is taken. A rank whose tokens touched no row
    of its own window has to enter the replica collective all the same, which is what reading here
    guarantees. No-op without an embedding; the replica sum is skipped when vdp == 1.
    """
    for _, embedding in get_vocab_embeddings(model):
        embedding._vp_row_grad.pending_sparse_grad()
    for _, embedding in get_device_vocab_embeddings(model):
        sum_replica_gradients(embedding)


def sum_replica_gradients(embedding) -> None:
    """Sum a device-resident embedding's gradient over the ranks that hold the same row window.

    When the embedding lives on the accelerator it is an ordinary parameter, so once
    ``vocab_data_parallel_size > 1`` the same window is replicated over vdp and every replica only sees
    its own share of the batch. The
    host path sums its replicas inside ``VocabRowGrad``; here the gradient is dense, so one all-reduce
    over the replica group is enough. Nothing is divided here: the return all-to-all already divides by
    the number of ranks the loss is averaged over (see ``comm.differentiable_all_to_all``). Summing
    before the clip keeps the reported norm consistent with the other parameters as well.
    """
    grad = embedding.weight.grad
    if grad is None:
        return
    # The row window is replicated over the 'vdp' mesh dim, so the group comes from the parallel state
    # (same accessor clip_grad_norm's has_vocab_replicas uses) instead of a cached module attribute.
    group = get_parallel_state().get_vdp_group()
    if group is None or torch.distributed.get_world_size(group) == 1:
        return
    local = grad.to_local() if isinstance(grad, DTensor) else grad
    torch.distributed.all_reduce(local, group=group)


def prepare_sparse_optimizer_step(model) -> None:
    """Attach each embedding's pending sparse gradient to ``weight.grad``."""
    for _, embedding in get_vocab_embeddings(model):
        embedding.weight.grad = embedding._vp_row_grad.pending_sparse_grad()


def clear_sparse_gradient(model) -> None:
    """Drop the pending gradients once the optimizer has consumed them."""
    for _, embedding in get_vocab_embeddings(model):
        embedding.weight.grad = None
        embedding._vp_row_grad.clear()
