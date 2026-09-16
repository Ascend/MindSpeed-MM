"""Unified metrics entry point with pluggable handler backends.

This module lets the training loop report metrics through a single
``metrics.record(...)`` call without knowing whether the data ends up as a text
log line or as TensorBoard curves (or, in the future, wandb etc.). It mirrors
the logging.Logger / logging.Handler split: ``record`` is the unified entry and
each ``MetricsHandler`` is a pluggable backend.

It is parallel to the text-logging module (RFC #664 / PR #3092): the
``TensorBoardHandler`` here is one backend; a ``TextLogHandler`` can be added
later once the logging module is merged, sharing the same dispatcher.

All four kinds are routed by ``TensorBoardHandler`` today:

- ``scalar``: overall training scalars (loss / learning rate / grad norm /
  iteration time / consumed samples / val loss) written under the caller-given
  ``prefix`` ("train" / "val"); the "tokens" prefix additionally honors the
  ``token_stats`` switch.
- ``per_layer``: per-layer gradient norms (computed before clipping by
  ``mindspeed_mm.fsdp.optimizer.clip_grad_norm.compute_per_layer_grad_norm``),
  written under the "grad_norm" prefix, gated by ``grad_norm_per_layer``.
- ``per_rank``: per-rank local values (token counts), gathered to the main rank
  and aggregated into min/max/ave/std curves, gated by ``token_stats``.
- ``per_rank_detail``: one curve per rank, gated by ``token_stats_per_rank``.

Feature switches are read from the config inside the handler, so the training
loop only calls ``metrics.record(...)`` without knowing which backend consumes
the payload.
"""

import logging
from abc import ABC, abstractmethod

from mindspeed_mm.fsdp.log.tensorboard import tb_writer
from mindspeed_mm.fsdp.utils.decorators import Singleton

logger = logging.getLogger(__name__)

# Metric kinds produced by the training loop.
KIND_SCALAR = "scalar"
KIND_PER_LAYER = "per_layer"
KIND_PER_RANK = "per_rank"
KIND_PER_RANK_DETAIL = "per_rank_detail"


class MetricsHandler(ABC):
    """A pluggable metrics backend."""

    @abstractmethod
    def setup(self) -> None:
        """Initialize the backend (e.g. create the SummaryWriter)."""

    @abstractmethod
    def record(self, iteration: int, metrics: dict, kind: str, prefix: str) -> None:
        """Handle one metrics payload. ``kind`` selects the semantics, ``prefix``
        namespaces the output (e.g. "train" / "val")."""

    @abstractmethod
    def close(self) -> None:
        """Flush and release the backend."""


class TensorBoardHandler(MetricsHandler):
    """Write metrics to TensorBoard via the existing ``tb_writer`` singleton.

    Takes two configs: ``config`` is the TensorBoard *backend* section
    (``args.tools.tensorboard``: enable/dir), and ``metrics_config`` is the
    backend-agnostic *feature-switch* section (``args.training.metrics``:
    grad_norm_per_layer / token_stats / ...). Keeping them separate lets other
    backends reuse the same feature switches. When ``metrics_config`` is None,
    all feature switches default to off (only train/val scalars are written).
    """

    def __init__(self, config, metrics_config=None):
        self.config = config
        self.metrics_config = metrics_config

    def _switch(self, name: str) -> bool:
        return bool(getattr(self.metrics_config, name, False)) if self.metrics_config is not None else False

    def setup(self) -> None:
        tb_writer.reset(self.config)

    def record(self, iteration: int, metrics: dict, kind: str, prefix: str) -> None:
        # Feature switches live here (read from metrics_config), so the training
        # loop does not need to check them; it just calls metrics.record(...).
        if kind == KIND_SCALAR:
            # Token-count scalars (prefix "tokens") are gated by token_stats;
            # other scalar prefixes (train/val) are always written when enabled.
            if prefix == "tokens" and not self._switch("token_stats"):
                return
            tb_writer.write_scalars(iteration, metrics, prefix=prefix)
        elif kind == KIND_PER_LAYER:
            if self._switch("grad_norm_per_layer"):
                tb_writer.write_scalars(iteration, metrics, prefix="grad_norm")
        elif kind == KIND_PER_RANK:
            if self._switch("token_stats"):
                tb_writer.write_rank_scalars(iteration, metrics, prefix="rank")
        elif kind == KIND_PER_RANK_DETAIL:
            if self._switch("token_stats_per_rank"):
                tb_writer.write_per_rank_scalars(
                    iteration,
                    metrics,
                    prefix="rank_tokens",
                    ranks=getattr(self.metrics_config, "token_stats_per_rank_list", None),
                )

    def close(self) -> None:
        tb_writer.close()


class MetricsDispatcher(metaclass=Singleton):
    """Forward each metrics payload to every registered handler."""

    def __init__(self):
        self._handlers = []

    def add_handler(self, handler: MetricsHandler) -> None:
        self._handlers.append(handler)

    def reset(self) -> None:
        self.close()
        self._handlers.clear()

    def setup(self) -> None:
        for handler in self._handlers:
            handler.setup()

    def record(self, iteration: int, metrics: dict, kind: str = KIND_SCALAR, prefix: str = "train") -> None:
        for handler in self._handlers:
            handler.record(iteration, metrics, kind, prefix)

    def close(self) -> None:
        for handler in self._handlers:
            handler.close()


metrics = MetricsDispatcher()
