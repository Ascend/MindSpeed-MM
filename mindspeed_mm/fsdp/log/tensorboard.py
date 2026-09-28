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
  :meth:`write_rank_detail` renders the same gathered row four ways: one curve
  per rank, a per-step histogram over every rank, a rank x step heatmap (labelled
  when matplotlib is available, plain RGB otherwise) and the max/min/avg
  imbalance scalars.
- When disabled or on non-main ranks, all write calls are no-ops so there is no
  overhead and no behavior change.
"""

import logging
import os
import time

import numpy as np
import torch

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.utils.decorators import Singleton
from mindspeed_mm.fsdp.utils.device import get_device_type

logger = logging.getLogger(__name__)

# Set once when labelling a heatmap fails (matplotlib missing or broken), so the
# fallback is reported without spamming the log on every render.
_warned_unlabelled_heatmap = False


def _get_global_rank() -> int:
    """Resolve the global rank, tolerating an uninitialized process group."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return envs.get("RANK")


def _get_world_size() -> int:
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return envs.get("WORLD_SIZE")


def _normalize_step_rows(rows) -> "np.ndarray":
    """Normalize a ``(steps x ranks)`` matrix by each step's own maximum.

    The absolute token count differs from step to step, and what this picture is
    meant to show is which ranks sit below the others, so each row is scaled by
    that step's best rank (1.0 = the rank that processed the most tokens).
    """
    matrix = np.stack(rows)
    peak = matrix.max(axis=1, keepdims=True)
    return np.divide(matrix, peak, out=np.zeros_like(matrix), where=peak > 0)


def _heatmap_rgb(norm) -> "np.ndarray":
    """Colorize a 0..1 matrix as an RGB image (HWC, uint8).

    Used when matplotlib is unavailable, in which case the picture carries no axis
    ticks and no colorbar.
    """
    low = np.array([255.0, 255.0, 204.0])  # pale yellow
    high = np.array([128.0, 0.0, 38.0])  # dark red
    return (low + (high - low) * norm[..., None]).astype(np.uint8)


def _labelled_heatmap(norm, steps, tag, rank_labels=None) -> "np.ndarray":
    """Render the heatmap with matplotlib: rank ticks, step ticks and a colorbar.

    ``rank_labels`` names every column (the rank it was collected from) and is used for
    the x ticks, so the picture stays readable when the collected ranks are not a
    contiguous ``0..N-1`` range (for example a narrowed ``token_stats_per_rank_list``
    or a data-parallel subset of the world).

    At most ~20 rank ticks and ~15 step ticks are drawn, and the canvas is capped,
    so a few thousand ranks cannot turn one heatmap into a multi-megabyte image.
    """
    import io

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    step_count, rank_count = norm.shape
    # Columns are labelled with the rank they were collected from when known, so the
    # x axis agrees with the `rank{i}` curves even for a non-contiguous rank set.
    column_labels = list(rank_labels) if rank_labels is not None else list(range(rank_count))
    fig, ax = plt.subplots(
        figsize=(min(60.0, max(8.0, rank_count * 0.08)), min(40.0, max(4.0, step_count * 0.12)))
    )
    # An Agg canvas of this size can reach tens of megabytes, and pyplot keeps every
    # figure alive in its global registry until it is closed, so drawing *and*
    # encoding are both wrapped: a failure anywhere (colorbar, layout, savefig) must
    # not leak the figure, or a persistent failure would exhaust the main rank.
    try:
        im = ax.imshow(norm, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0, origin="upper", interpolation="nearest")
        rank_ticks = list(range(0, rank_count, max(1, rank_count // 20)))
        ax.set_xticks(rank_ticks)
        ax.set_xticklabels([str(column_labels[i]) for i in rank_ticks], fontsize=7)
        row_ticks = list(range(0, step_count, max(1, step_count // 15)))
        ax.set_yticks(row_ticks)
        ax.set_yticklabels([str(steps[i]) for i in row_ticks], fontsize=7)
        ax.set_xlabel("rank")
        ax.set_ylabel("step")
        ax.set_title(f"{tag.rsplit('/', 1)[-1]} across ranks (steps {steps[0]}-{steps[-1]})", fontsize=9)
        fig.colorbar(im, ax=ax, label="tokens / step max")
        fig.tight_layout()

        buffer = io.BytesIO()
        try:
            fig.savefig(buffer, format="png", dpi=100)
            buffer.seek(0)
            return np.asarray(Image.open(buffer).convert("RGB"))
        finally:
            buffer.close()
    finally:
        plt.close(fig)


def _heatmap_image(norm, steps, tag, rank_labels=None) -> "np.ndarray":
    """Labelled heatmap when matplotlib is available, plain RGB otherwise.

    matplotlib is an optional dependency here, and a metrics view must never break
    a training run, so any rendering failure falls back to the unlabelled image and
    is reported once.
    """
    global _warned_unlabelled_heatmap
    try:
        return _labelled_heatmap(norm, steps, tag, rank_labels)
    except Exception as exc:  # noqa: BLE001 - a metrics view must never break training
        if not _warned_unlabelled_heatmap:
            logger.warning(
                "Heatmaps are written without axis labels or colorbar (%s). "
                "Install matplotlib to get labelled heatmaps.",
                exc,
            )
            _warned_unlabelled_heatmap = True
        return _heatmap_rgb(norm)


class TensorBoardWriter(metaclass=Singleton):
    """Process-wide TensorBoard writer gated to a single main rank."""

    # A heatmap keeps the most recent logged steps; redrawing it rewrites the whole
    # history, so it is throttled relative to the per-step histograms.
    _HEATMAP_MAX_STEPS = 200
    _HEATMAP_RENDER_EVERY = 10

    def __init__(self):
        self.enable = False
        self.writer = None
        self.main_rank = 0
        self._heatmap_rows = {}
        self._hist_calls = 0

    def reset(self, config) -> None:
        """(Re)initialize from a ``Tensorboard`` config section.

        Only creates a ``SummaryWriter`` when ``config.enable`` is set and the
        current process is the main rank. Safe to call on every rank. Write
        frequency is controlled by the caller (training.log_interval), not here.
        """
        self.enable = bool(getattr(config, "enable", False))
        self.main_rank = envs.get("TENSORBOARD_MAIN_RANK")
        self.writer = None
        self._heatmap_rows = {}
        self._hist_calls = 0
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

    def _gather_rank_values(self, local_values: list):
        """Gather one row of values from every rank to the main rank.

        Every rank must call this: it is a collective. Returns ``packed`` with
        ``packed[rank][pos]`` on the main rank and ``None`` on the others (which
        have nothing left to write). The values travel as float32 because HCCL does
        not support float64 collectives; a single rank's raw value fits comfortably
        in float32, so the precision-sensitive sums are still done on the host, in
        Python floats, by the callers.
        """
        device = get_device_type()
        # Pack all values into one tensor for a single gather (HCCL-safe).
        local_tensor = torch.tensor(local_values, dtype=torch.float32, device=device)
        if not torch.distributed.is_initialized():
            return [local_tensor.tolist()]
        world_size = _get_world_size()
        if self.is_main_rank():
            gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        else:
            gather_list = None
        torch.distributed.gather(local_tensor, gather_list, dst=self.main_rank)
        if not self.is_main_rank() or self.writer is None:
            return None
        return [t.tolist() for t in gather_list]

    def write_rank_scalars(self, iteration: int, local_scalars: dict, prefix: str = "rank", stats=None) -> None:
        """Aggregate per-rank scalars across ranks and write distribution curves.

        Every rank must call this with its own local values; they are gathered to
        the main rank (see :meth:`_gather_rank_values`) and the statistics are then
        computed there in Python (float64) precision. With distributed
        uninitialized it degrades to writing only the local value (std = 0).

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
        packed = self._gather_rank_values([float(local_scalars[name]) for name in names])
        if packed is None:
            return
        rank_count = len(packed)

        # Compute statistics on the main rank in Python (float64) precision.
        for pos, name in enumerate(names):
            values = [packed[rank_idx][pos] for rank_idx in range(rank_count)]
            ave_v = sum(values) / rank_count
            mean_sq = sum(v * v for v in values) / rank_count
            stat_values = {
                "min": min(values),
                "max": max(values),
                "ave": ave_v,
                "std": max(0.0, mean_sq - ave_v ** 2) ** 0.5,
            }
            tag = name.replace(" ", "_")
            for stat_name in stats:
                self.writer.add_scalar(f"{prefix}/{tag}/{stat_name}", stat_values[stat_name], iteration)

    def write_rank_detail(
        self, iteration: int, local_scalars: dict, prefix: str = "rank_detail",
        histogram_prefix: str = "tokens/hist", heatmap_prefix: str = "tokens/heatmap",
        imbalance_prefix: str = "tokens/vio", ranks=None,
    ) -> None:
        """Render the per-rank payload four ways from a single gather.

        - one curve per rank, ``{prefix}/{name}/rank{i}``, for the ranks in
          ``ranks`` (None = all);
        - one histogram per step, ``{histogram_prefix}/{name}``, always over every
          gathered rank: it is the picture of the distribution the min/max/ave/std
          curves summarize, so narrowing it to a few ranks would make the two views
          disagree;
        - one ``(steps x ranks)`` heatmap, ``{heatmap_prefix}/{name}``, restricted to
          ``ranks`` (None = all), because a single image can afford to show the whole
          history at once and that is where an out-of-line rank stands out;
        - the imbalance of each metric under ``{imbalance_prefix}/{name}``:
          ``max_vio`` / ``min_vio`` / ``avg_vio`` are relative to the mean of the
          gathered ranks, so all three are 0.0 when every rank processed the same
          amount (the analogue of a load-balance monitor's violation triple).

        Every rank must call this: it gathers. Non-main ranks only take part in the
        gather, and with distributed uninitialized it degrades to the local rank.
        """
        if not self.enable:
            return
        names = [name for name, value in local_scalars.items() if value is not None]
        if not names:
            return
        packed = self._gather_rank_values([float(local_scalars[name]) for name in names])
        if packed is None:
            return
        rank_count = len(packed)
        # `ranks` narrows the per-rank curves and the heatmap, never the histogram.
        keep_ranks = list(range(rank_count)) if ranks is None else [r for r in range(rank_count) if r in set(ranks)]

        for pos, name in enumerate(names):
            tag = name.replace(" ", "_")
            per_rank = np.asarray([packed[rank_idx][pos] for rank_idx in range(rank_count)], dtype=np.float32)
            if histogram_prefix:
                self.writer.add_histogram(f"{histogram_prefix}/{tag}", per_rank, iteration)
            for rank_idx in keep_ranks:
                self.writer.add_scalar(f"{prefix}/{tag}/rank{rank_idx}", float(per_rank[rank_idx]), iteration)
            if heatmap_prefix and keep_ranks:
                self._append_heatmap_row(
                    f"{heatmap_prefix}/{tag}", per_rank[keep_ranks], iteration, keep_ranks
                )
            self._write_imbalance(imbalance_prefix, tag, per_rank, iteration)

        self._render_heatmaps(iteration)

    def _write_imbalance(self, prefix: str, tag: str, per_rank, iteration: int) -> None:
        """Write the rank-to-rank imbalance of one metric.

        The deviations are relative to the mean of the gathered ranks, so all three
        scalars are 0.0 when every rank processed the same amount. A rank that
        processed twice the average reports ``max_vio == 1.0``; one that processed
        half reports ``min_vio == -0.5``.
        """
        if not prefix:
            return
        mean = float(per_rank.mean())
        if mean <= 0:
            return
        deviation = per_rank / mean - 1.0
        self.writer.add_scalar(f"{prefix}/{tag}/max_vio", float(deviation.max()), iteration)
        self.writer.add_scalar(f"{prefix}/{tag}/min_vio", float(deviation.min()), iteration)
        self.writer.add_scalar(f"{prefix}/{tag}/avg_vio", float(np.abs(deviation).mean()), iteration)

    def _append_heatmap_row(self, tag: str, step_values, iteration: int, rank_labels=None) -> None:
        """Append one step to a ``{tag}`` heatmap, keeping only recent steps."""
        state = self._heatmap_rows.setdefault(tag, {"steps": [], "rows": [], "rank_labels": None})
        state["rows"].append(np.asarray(step_values, dtype=np.float32))
        state["steps"].append(iteration)
        if rank_labels is not None:
            state["rank_labels"] = list(rank_labels)
        if len(state["rows"]) > self._HEATMAP_MAX_STEPS:
            state["rows"].pop(0)
            state["steps"].pop(0)

    def _render_heatmaps(self, iteration: int) -> None:
        """Redraw the heatmaps every ``_HEATMAP_RENDER_EVERY`` calls.

        Each image rewrites the whole history, so it is throttled relative to the
        per-step histograms and per-rank curves.
        """
        self._hist_calls += 1
        if self._hist_calls % self._HEATMAP_RENDER_EVERY:
            return
        for tag, state in self._heatmap_rows.items():
            if state["rows"] and len(state["rows"][0]) > 0:
                norm = _normalize_step_rows(state["rows"])
                self.writer.add_image(
                    tag,
                    _heatmap_image(norm, state["steps"], tag, state.get("rank_labels")),
                    iteration,
                    dataformats="HWC",
                )

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


tb_writer = TensorBoardWriter()
