"""Token counting utility for training-step statistics.

This is a data/training-statistics helper, independent of any metrics backend
(TensorBoard, text log, etc.). The training loop enables it for the steps it will
log, feeds it each micro-batch and reads the counters once per step.
"""

import torch
import torch.distributed as dist

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state, is_parallel_state_initialized
from mindspeed_mm.fsdp.utils.device import get_device_type


def _as_int(value) -> int:
    """Materialise an accumulator as a Python int (one device sync per read)."""
    return int(value.item()) if torch.is_tensor(value) else int(value)


class TokenCounter:
    """Accumulate per-step token counts across a gradient-accumulation window.

    Keeps token-counting logic out of the training loop: the caller only invokes
    :meth:`update` per micro-batch and reads :meth:`step_totals` once per step.
    When ``enabled`` is False, ``update`` is a no-op.

    ``valid`` and ``padding`` are accumulated on the device and are only copied to
    the host when read, so a counted micro-batch costs no device synchronisation:
    the training loop reads the counters after the optimizer step, never inside the
    accumulation loop where a sync would stall the input prefetch. Callers that log
    every N steps therefore enable the counter only on those steps.

    The counters themselves are per rank; :meth:`step_totals` turns them into what
    one step processed in total, while the per-rank values stay available for the
    rank-to-rank distribution.

    Tokens are split into three non-overlapping kinds (total = valid + prompt +
    padding):
    - ``valid``: positions whose label is not IGNORE_INDEX (-100) — the answer
      tokens the model actually learns, i.e. the count the loss is normalized by.
    - ``padding``: positions where ``attention_mask == 0`` — real pad placeholders.
      This works for both 0/1 masks and packed segment-id masks (segment ids start
      at 1, so packed data has no zeros and thus padding = 0).
    - ``prompt``: real data tokens that are not learned (attention_mask != 0 but
      label == -100), i.e. ``total - padding - valid``.
    """

    def __init__(self):
        self.enabled = False
        self._valid = 0
        self._padding = 0
        self.total = 0

    def reset(self, enabled: bool) -> None:
        self.enabled = enabled
        self._valid = 0
        self._padding = 0
        self.total = 0

    def update(self, batch_data: dict) -> None:
        if not self.enabled:
            return
        labels = batch_data.get("labels")
        attention_mask = batch_data.get("attention_mask")
        if labels is not None:
            self._valid = self._valid + (labels != -100).sum()
        if attention_mask is not None:
            self.total += attention_mask.numel()
            self._padding = self._padding + (attention_mask == 0).sum()

    @property
    def valid(self) -> int:
        return _as_int(self._valid)

    @property
    def padding(self) -> int:
        return _as_int(self._padding)

    @property
    def prompt(self) -> int:
        return self.total - self.padding - self.valid

    def step_totals(self) -> dict:
        """Return this step's token counts, summed over the data-parallel group.

        The per-rank counts are summed over the DP group because its ranks hold
        distinct samples: context- and tensor-parallel siblings hold copies of the
        same samples and would multiply the counts. All three kinds are summed with
        the same group so that ``total == padding + prompt + valid`` keeps holding
        globally and ``valid_tokens`` stays the count this step's loss is normalized
        by. This is a collective, so every rank of the group must call it; without an
        initialized distributed and parallel state the local counts are returned.
        """
        valid, padding, total = self.valid, self.padding, self.total
        if dist.is_initialized() and is_parallel_state_initialized():
            counts = torch.tensor([valid, padding, total], dtype=torch.int64, device=get_device_type())
            dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=get_parallel_state().get_dp_group())
            valid, padding, total = counts.tolist()
        return {"valid_tokens": valid, "padding_tokens": padding, "total_tokens": total}
