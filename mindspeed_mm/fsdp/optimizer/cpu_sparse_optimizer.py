"""SparseAdam for parameters whose gradient arrives as a row ledger instead of ``.grad``.

Such a table lives outside the dense parameter set, so the member owns the two step boundaries itself:
it attaches the step's gradient before ``step`` and drops it after ``zero_grad``. Both boundaries and
the optional checkpoint adaptation are injected by the caller, which keeps this module independent of
the subsystem that produces the ledger.
"""
import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)


# Name of the cpu table optimizer inside MultiOptimizer: its moments are stored under this entry next to
# the dense ones (see optimizer.py).
CPU_SPARSE_KEY = "cpu_sparse"

logger = logging.getLogger(__name__)


@dataclass
class CpuSparseHooks:
    """What the table owner injects: the two ledger boundaries and the optional checkpoint adaptation."""
    before_step: Optional[Callable[[], None]] = None
    after_zero_grad: Optional[Callable[[], None]] = None
    checkpoint_adapter: object = None


class CpuSparseAdam(torch.optim.SparseAdam):
    """SparseAdam over a cpu-resident table, driven by a row ledger owned by the caller."""

    def __init__(self, params, *, lr, betas, eps, before_step=None, after_zero_grad=None,
                 checkpoint_adapter=None):
        """
        before_step: attaches this step's gradient; must be idempotent, clipping may have called it.
        after_zero_grad: drops the gradient and the ledger, at the same boundary as ``.grad``.
        checkpoint_adapter: optional object with ``adapt_saved(model, state)`` / ``adapt_loaded(model,
            state)``, for entries whose keys depend on how the table is sharded.
        """
        super().__init__(params, lr=lr, betas=betas, eps=eps)
        self._before_step = before_step
        self._after_zero_grad = after_zero_grad
        self._checkpoint_adapter = checkpoint_adapter

    def step(self, closure=None):
        if self._before_step is not None:
            self._before_step()
        return super().step(closure)

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none)
        if self._after_zero_grad is not None:
            self._after_zero_grad()

    # ``MultiOptimizer`` looks for these two hooks (see its state_dict/load_state_dict).

    def state_dict_for(self, model) -> dict:
        """The member's entries, with the moments torch would create lazily filled in.

        ``torch.optim`` builds state on the first update, so a parameter that never received a gradient
        would otherwise be written as a partial state that a freshly built optimizer cannot load.
        """
        _materialize_missing_state(self)
        state = get_optimizer_state_dict(
            model, self, options=StateDictOptions(flatten_optimizer_state_dict=True)
        )
        if self._checkpoint_adapter is not None:
            self._checkpoint_adapter.adapt_saved(model, state)
        return state

    def load_state_dict_for(self, model, state: dict) -> None:
        """Take the member's entries back and restore what the DCP load drops."""
        if self._checkpoint_adapter is not None:
            self._checkpoint_adapter.adapt_loaded(model, state)
        set_optimizer_state_dict(
            model,
            self,
            optim_state_dict=state,
            options=StateDictOptions(flatten_optimizer_state_dict=True),
        )
        _restore_group_defaults(self)


def _materialize_missing_state(optimizer) -> None:
    """Give the optimizer the zero moments it would create on a parameter's first update."""
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


def build_cpu_sparse_member(params, *, lr, betas, eps, hooks: CpuSparseHooks) -> CpuSparseAdam:
    """``CpuSparseAdam`` over ``params``, with the hooks of the table they belong to.

    ``params`` is the caller's host-resident list and is expected to be non-empty: the caller only
    asks for the member when the model actually keeps a table out of the dense parameter set.
    """
    elements = sum(param.numel() for param in params)
    logger.info(
        "cpu sparse optimizer: SparseAdam over %d embedding(s), fp32 moments up to %.1f GiB "
        "(created on the first step)", len(params), elements * 8 / 1024 ** 3,
    )
    return CpuSparseAdam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        before_step=hooks.before_step,
        after_zero_grad=hooks.after_zero_grad,
        checkpoint_adapter=hooks.checkpoint_adapter,
    )
