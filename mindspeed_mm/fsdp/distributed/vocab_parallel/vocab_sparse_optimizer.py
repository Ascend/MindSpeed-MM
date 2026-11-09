"""Sparse optimizer of the cpu-resident vocab embeddings: the members lifecycle and the gradient phases."""
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.tensor import DTensor

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state

from .vocab_checkpoint import add_shard_suffixes, get_host_vocab_shard_keys, remove_shard_suffixes
from .vocab_sparse_grad import get_device_vocab_embeddings, get_vocab_embeddings


def finalize_vocab_gradients(model) -> None:
    """Materialize every embedding's step gradient: merge its rows and sum the row-window replicas.

    ``clip_grad_norm`` calls this from ``_vocab_stat``, its first reader, so it happens once per step on
    every rank. A rank whose tokens touched no row of its own window has to enter the replica collective
    all the same, which is what reading here guarantees. No-op without an embedding; the replica sum is
    skipped when vdp == 1.
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


class CpuSparseAdam(torch.optim.SparseAdam):
    """SparseAdam over the cpu-resident embeddings, owning the row-ledger lifecycle.

    The embeddings' gradient is assembled by the backward into a row ledger instead of ``.grad``. The
    member attaches it before it steps and drops it together with ``.grad``, so the container that holds
    it only steps its members and knows nothing about the ledger.
    """

    def __init__(self, params, model, *, lr, betas, eps):
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self.model = model

    def step(self, closure=None):
        # Idempotent: clipping may already have materialized the gradient for the norm.
        prepare_sparse_optimizer_step(self.model)
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        # The same boundary as ``.grad``: the ledger is only read between a step and its zero_grad.
        clear_sparse_gradient(self.model)


    def state_dict_for(self, model) -> dict:
        """The member's checkpoint entries: every row's moments, keyed with its row window.

        A row that never received a gradient has no state yet, and a freshly built optimizer could not
        load a partial one; the window suffix tells a resumed run which rows the entries belong to.
        """
        _materialize_missing_state(self)
        state = get_optimizer_state_dict(
            model, self, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )
        add_shard_suffixes(state, get_host_vocab_shard_keys(model))
        return state

    def load_state_dict_for(self, model, state) -> None:
        """Take the member's entries back: drop the window suffix first, then restore what the load drops."""
        remove_shard_suffixes(state, get_host_vocab_shard_keys(model))
        set_optimizer_state_dict(
            model,
            self,
            optim_state_dict=state,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        _restore_group_defaults(self)


def prepare_sparse_optimizer_step(model) -> None:
    """Attach each embedding's pending sparse gradient to ``weight.grad``."""
    for _, embedding in get_vocab_embeddings(model):
        embedding.weight.grad = embedding._vp_row_grad.pending_sparse_grad()


def clear_sparse_gradient(model) -> None:
    """Drop the pending gradients once the optimizer has consumed them."""
    for _, embedding in get_vocab_embeddings(model):
        embedding.weight.grad = None
        embedding._vp_row_grad.clear()


def _materialize_missing_state(optimizer) -> None:
    """Give the optimizer the zero moments it would create on a row's first update.

    ``torch.optim`` builds state lazily, so an embedding that never received a gradient would be written as a
    partial state that a freshly built optimizer cannot load.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            state = optimizer.state[param]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)


def _restore_group_defaults(optimizer) -> None:
    """Put back the hyperparameters that the DCP optimizer load drops.

    ``set_optimizer_state_dict`` rebuilds ``param_groups`` from the checkpoint and keeps the tensors and
    ``lr`` only, so a resumed ``SparseAdam`` fails on its first ``step()`` with ``KeyError: 'betas'``.
    """
    for group in optimizer.param_groups:
        for key, value in optimizer.defaults.items():
            group.setdefault(key, value)
