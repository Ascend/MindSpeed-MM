"""TensorBoard scalar recording backend for MindSpeed-MM FSDP2 training.

This module is only responsible for *writing* scalars to TensorBoard; it is the
backend used by the metrics handlers in ``mindspeed_mm.fsdp.log.metrics``. The
business computations that produce those scalars live elsewhere:
- per-layer gradient norms: ``mindspeed_mm.fsdp.optimizer.clip_grad_norm``
  (``compute_per_layer_grad_norm``),
- per-step token counts: ``mindspeed_mm.fsdp.utils.token_counter`` (``TokenCounter``).

It is parallel to the unified text-logging module (RFC #664): it lives in the
same ``mindspeed_mm.fsdp.log`` package but is fully independent and can be
enabled separately.

Design:
- A single process, the "main rank" (global rank ``TENSORBOARD_MAIN_RANK``,
  default 0), owns the ``SummaryWriter`` and writes to disk. Each run writes to
  a timestamped subdirectory of ``config.dir`` so runs do not mix.
- Overall metrics are already reduced across the data-parallel group by the
  training loop, so the main rank writes them directly via :meth:`write_scalars`.
- Per-rank metrics are gathered to the main rank as float32 (HCCL does not
  support float64 collectives) and aggregated there in Python (float64)
  precision into min/max/ave/std curves by :meth:`write_rank_scalars`;
  :meth:`write_per_rank_scalars` writes one curve per rank.
- When disabled or on non-main ranks, all write calls are no-ops so there is no
  overhead and no behavior change.
"""

import logging
import os
import time

import torch

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.utils.decorators import Singleton
from mindspeed_mm.fsdp.utils.device import get_device_type

logger = logging.getLogger(__name__)


def _get_global_rank() -> int:
    """Resolve the global rank, tolerating an uninitialized process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return envs.get("RANK")


def _get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return envs.get("WORLD_SIZE")


class TensorBoardWriter(metaclass=Singleton):
    """Process-wide TensorBoard writer gated to a single main rank."""

    def __init__(self):
        self.enable = False
        self.writer = None
        self.main_rank = 0

    def reset(self, config) -> None:
        """(Re)initialize from a ``Tensorboard`` config section.

        Only creates a ``SummaryWriter`` when ``config.enable`` is set and the
        current process is the main rank. Safe to call on every rank. Write
        frequency is controlled by the caller (training.log_interval), not here.
        """
        self.enable = bool(getattr(config, "enable", False))
        self.main_rank = envs.get("TENSORBOARD_MAIN_RANK")
        self.writer = None
        if not self.enable:
            return
        if not self.is_main_rank():
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise ImportError(
                "TensorBoard recording is enabled (tools.tensorboard.enable=true) "
                "but the `tensorboard` package is not installed. "
                "Please install it with `pip install tensorboard`."
            ) from exc
        base_dir = getattr(config, "dir", "./tensorboard")
        run_dir = os.path.join(base_dir, time.strftime("%Y%m%d_%H%M%S"))
        self.writer = SummaryWriter(log_dir=run_dir)
        logger.info("TensorBoard writing enabled on rank %d, log dir: %s", self.main_rank, run_dir)

    def is_main_rank(self) -> bool:
        return _get_global_rank() == self.main_rank

    def should_write(self) -> bool:
        return self.enable and self.writer is not None

    def write_scalars(self, iteration: int, scalars: dict, prefix: str = "train") -> None:
        """Write overall (already-reduced) scalars. No-op off the main rank."""
        if not self.should_write():
            return
        for name, value in scalars.items():
            if value is None:
                continue
            tag = name.replace(" ", "_")
            self.writer.add_scalar(f"{prefix}/{tag}", value, iteration)

    def write_rank_scalars(self, iteration: int, local_scalars: dict, prefix: str = "rank", stats=None) -> None:
        """Aggregate per-rank scalars across ranks and write distribution curves.

        Every rank must call this with its own local values. The values are
        gathered to the main rank as float32 (HCCL does not support float64
        collectives), and the statistics are then computed locally on the main
        rank in Python (float64) precision. A single rank's raw value fits
        comfortably in float32, so the gather loses no meaningful precision,
        while the precision-sensitive sums / square-sums / std are done off-device
        where float64 is available. With distributed uninitialized it degrades to
        writing only the local value (std = 0).

        ``stats`` selects which statistics to write, as a list of names among
        ``"min"``, ``"max"``, ``"ave"``, ``"std"``. ``None`` (default) writes all
        four. Unknown names raise ``ValueError``.
        """
        if not self.enable:
            return
        all_stats = ("min", "max", "ave", "std")
        if stats is None:
            stats = list(all_stats)
        else:
            invalid = [s for s in stats if s not in all_stats]
            if invalid:
                raise ValueError(f"write_rank_scalars got invalid stats {invalid}, must be among {all_stats}.")
        names = [name for name, value in local_scalars.items() if value is not None]
        if not names:
            return
        device = get_device_type()
        # Pack all scalars into one float32 tensor for a single gather (HCCL-safe).
        local_tensor = torch.tensor(
            [float(local_scalars[name]) for name in names], dtype=torch.float32, device=device
        )

        if torch.distributed.is_initialized():
            world_size = _get_world_size()
            if self.is_main_rank():
                gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            else:
                gather_list = None
            torch.distributed.gather(local_tensor, gather_list, dst=self.main_rank)
            if not self.is_main_rank() or self.writer is None:
                return
            packed = [t.tolist() for t in gather_list]
        else:
            world_size = 1
            packed = [local_tensor.tolist()]

        # Compute statistics on the main rank in Python (float64) precision.
        for pos, name in enumerate(names):
            values = [packed[rank_idx][pos] for rank_idx in range(world_size)]
            ave_v = sum(values) / world_size
            mean_sq = sum(v * v for v in values) / world_size
            stat_values = {
                "min": min(values),
                "max": max(values),
                "ave": ave_v,
                "std": max(0.0, mean_sq - ave_v ** 2) ** 0.5,
            }
            tag = name.replace(" ", "_")
            for stat_name in stats:
                self.writer.add_scalar(f"{prefix}/{tag}/{stat_name}", stat_values[stat_name], iteration)

    def write_per_rank_scalars(self, iteration: int, local_scalars: dict, prefix: str = "rank_detail", ranks=None) -> None:
        """Write one curve per rank for each scalar (tag: ``{prefix}/{name}/rank{i}``).

        Every rank's own values are gathered to the main rank with a single
        tensor ``gather`` (all values packed into one tensor to minimize
        collectives), then the main rank writes each rank's curve. Only ranks in
        ``ranks`` (None = all ranks) are written. This is a collective, so every
        rank must call it; non-main ranks only participate in the gather. With
        distributed uninitialized it degrades to writing only the local rank.
        """
        if not self.enable:
            return
        names = [name for name, value in local_scalars.items() if value is not None]
        if not names:
            return
        device = get_device_type()
        local_tensor = torch.tensor(
            [float(local_scalars[name]) for name in names], dtype=torch.float32, device=device
        )

        if torch.distributed.is_initialized():
            world_size = _get_world_size()
            if self.is_main_rank():
                gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
            else:
                gather_list = None
            torch.distributed.gather(local_tensor, gather_list, dst=self.main_rank)
            if not self.is_main_rank() or self.writer is None:
                return
            packed = [t.tolist() for t in gather_list]
        else:
            world_size = 1
            packed = [local_tensor.tolist()]

        for rank_idx in range(world_size):
            if ranks is not None and rank_idx not in ranks:
                continue
            for pos, name in enumerate(names):
                tag = name.replace(" ", "_")
                self.writer.add_scalar(f"{prefix}/{tag}/rank{rank_idx}", packed[rank_idx][pos], iteration)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


tb_writer = TensorBoardWriter()
