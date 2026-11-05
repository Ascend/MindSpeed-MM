"""Row-sparse gradient accumulation for the host-resident vocab embedding."""
import logging

import torch
import torch.distributed as dist

from .vocab_host_store import fetch_rows_from_host
from . import vocab_stream


logger = logging.getLogger(__name__)


def _to_pinned(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Copy a device tensor into pinned host memory, keeping its shape.

    A pageable device-to-host copy runs at ~1 GB/s against ~25 GB/s into pinned memory, and the ledger
    is read back on the host by ``_coalesce`` and the optimizer, so the copy lands in pinned memory.
    The copy runs on the side stream: the ledger is only read at the end of the step (see
    ``pending_sparse_grad``), so the transfer overlaps the rest of the backward instead of blocking it.
    """
    source = tensor.detach()
    try:
        host = torch.empty(source.shape, dtype=dtype, pin_memory=True)
    except Exception as exc:
        logger.warning(
            "[VP] pinned gradient staging failed (%s); falling back to pageable host memory", exc
        )
        return source.to("cpu", dtype=dtype)
    with vocab_stream.issue():
        host.copy_(source, non_blocking=vocab_stream.enabled())
    vocab_stream.record()   # the host reads the buffer later, so a stream dependency is not enough
    vocab_stream.keep(source)
    return host


class VocabRowGrad:
    """Pending sparse gradient of one host-resident vocab embedding.

    ``group`` holds the replicas of this rank's row window (the ``vdp`` dimension); they are summed
    once per step, before anything reads the gradient.
    """

    def __init__(self, shape, group=None, device=None, device_sort: bool = False,
                 device_merge: bool = False):
        self.shape = tuple(shape)
        self.device_sort = device_sort
        self.device_merge = device_merge
        self.group = group
        self.device = device
        self._rows = []
        self._values = []
        self._dev_rows = []      # device-side parts, kept for ``device_merge`` (see ``add``)
        self._dev_vals = []
        self._pending = None
        self._reduced = False

    def add(self, rows: torch.Tensor, grads: torch.Tensor) -> None:
        rows = rows.reshape(-1)
        grads = grads.reshape(-1, grads.shape[-1])
        if self.device_merge:
            # The rows are already on the device, so the sort and the duplicate sum run there; the
            # merge completes in ``pending_sparse_grad``, where the host waits anyway (see ``_device_merge``).
            self._dev_rows.append(rows)
            self._dev_vals.append(grads)
            self._reduced = False
            return
        if self.device_sort and rows.numel() > 1:
            # Sorting here is nearly free: the ids are still on the device, the sort is over int32 ids
            # alone and the gather runs on the device at full bandwidth, while the same work on the
            # host at step end costs a CPU sort plus a pageable-side gather over the values.
            order = torch.argsort(rows)
            rows = rows[order]
            grads = grads.index_select(0, order)
        # int32 is enough for a row window (2.5M rows here) and halves what the host sort has to move.
        self._rows.append(_to_pinned(rows, torch.int32))
        self._values.append(_to_pinned(grads, grads.dtype))
        self._reduced = False   # a gradient read before these rows arrived is stale; clear() drops it

    def pending_sparse_grad(self):
        """Coalesced gradient of this step, summed over the replicas of the row window."""
        if not self._reduced:
            if self.device_merge and self._dev_rows:
                local = self._device_merge()
            else:
                # The rows were copied on the side stream during the backward; this reads them on the
                # host, so block here for the remainder of the transfer. Nothing queued is a no-op.
                vocab_stream.host_wait()
                local = self._coalesce(self._rows, self._values) if self._rows else None
            self._pending = self._sum_replicas(local) if self._has_replicas() else local
            self._reduced = True
        return self._pending

    def scale(self, factor: float) -> None:
        """Scale the pending gradient in place (grad clipping)."""
        if factor == 1.0:
            return
        grad = self.pending_sparse_grad()
        if grad is not None:
            grad._values().mul_(factor)

    def clear(self) -> None:
        self._rows, self._values = [], []
        self._dev_rows, self._dev_vals = [], []
        self._pending, self._reduced = None, False

    def _has_replicas(self) -> bool:
        return self.group is not None and dist.get_world_size(self.group) > 1

    def _coalesce(self, rows_parts: list, values_parts: list):
        """Merge the per-call ``(rows, values)`` parts into one coalesced COO gradient on the host.

        ``torch.sparse_coo_tensor(...).coalesce()`` sorts int64 indices and then builds an arange and
        a gather to reorder the values. Both steps are avoidable: the ids fit in int32, and when the
        rows already arrive ordered (``device_sort``) or come out of our own sort as strictly
        increasing, the COO can be flagged as coalesced instead of being coalesced again.
        """
        rows = torch.cat(rows_parts)
        if rows.numel() == 0:
            return None
        values = torch.cat(values_parts).to(torch.float32)
        indices = rows.to(torch.int64).unsqueeze(0)
        if self._is_sorted_unique(rows):
            # Already sorted and unique, so the COO is coalesced by construction and neither the sort
            # nor the duplicate sum of ``coalesce()`` has anything left to do.
            grad = torch.sparse_coo_tensor(
                indices, values, size=self.shape, device="cpu",
                is_coalesced=True, check_invariants=False,
            )
        else:
            # Duplicates present: hand the merge to torch. Its coalesce sorts and sums in C++ and
            # measured faster than the argsort + index_add_ pair we used to run here, so there is
            # nothing to gain from reimplementing it on the host.
            grad = torch.sparse_coo_tensor(
                indices, values, size=self.shape, device="cpu",
            ).coalesce()
        return grad

    @staticmethod
    def _is_sorted_unique(rows: torch.Tensor) -> bool:
        """True when ``rows`` is strictly increasing, i.e. the COO is already coalesced."""
        if rows.numel() <= 1:
            return True
        return bool((rows[1:] > rows[:-1]).all())

    def _device_merge(self):
        """Sort and deduplicate the row gradients on the device, then assemble the COO on the host.

        Called from ``pending_sparse_grad`` at step end. The host is going to block for this result in
        any case, so the one scalar read the compaction needs (how many distinct rows there are) is
        free here -- which is exactly what made the same merge mid-backward a net loss.
        """
        rows = torch.cat(self._dev_rows) if len(self._dev_rows) > 1 else self._dev_rows[0]
        values = torch.cat(self._dev_vals) if len(self._dev_vals) > 1 else self._dev_vals[0]
        rows = rows.reshape(-1).to(torch.int32)
        values = values.reshape(-1, values.shape[-1]).to(torch.float32)
        if rows.numel() == 0:
            self._dev_rows, self._dev_vals = [], []
            return None

        order = torch.argsort(rows)
        rows = rows[order]
        values = values.index_select(0, order)
        keep = torch.ones_like(rows, dtype=torch.bool)
        keep[1:] = rows[1:] != rows[:-1]
        if bool(keep.all()):
            merged_rows, merged_values = rows, values
        else:
            run = torch.cumsum(keep.to(torch.int64), 0) - 1
            merged_values = torch.zeros(int(run[-1]) + 1, values.shape[1], dtype=torch.float32,
                                        device=values.device)
            merged_values.index_add_(0, run, values)
            merged_rows = rows[keep]
        self._dev_rows, self._dev_vals = [], []

        host_rows = _to_pinned(merged_rows, torch.int32)
        host_values = _to_pinned(merged_values, torch.float32)
        vocab_stream.host_wait()
        grad = torch.sparse_coo_tensor(
            host_rows.to(torch.int64).unsqueeze(0), host_values.to(torch.float32),
            size=self.shape, device="cpu", is_coalesced=True, check_invariants=False,
        )
        return grad

    def _sum_replicas(self, local):
        """Sum the local ``(rows, values)`` of every replica of this window."""
        size = dist.get_world_size(self.group)
        if local is None:
            local_rows = torch.empty(0, dtype=torch.int64)
            local_values = torch.empty(0, self.shape[1], dtype=torch.float32)
        else:
            local_rows, local_values = local.indices()[0], local._values()

        counts = torch.zeros(size, dtype=torch.int64, device=self.device)
        counts[dist.get_rank(self.group)] = local_rows.numel()
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=self.group)
        max_rows = int(counts.max())
        if max_rows == 0:  # no replica hit a row this step
            return None

        # Row counts differ per replica, so the gather is padded to the largest one.
        padded_rows = torch.zeros(max_rows, dtype=torch.int64, device=self.device)
        padded_rows[: local_rows.numel()] = local_rows.to(self.device)
        padded_values = torch.zeros(max_rows, self.shape[1], dtype=torch.float32, device=self.device)
        padded_values[: local_values.shape[0]] = local_values.to(self.device)
        gathered_rows = [torch.empty_like(padded_rows) for _ in range(size)]
        gathered_values = [torch.empty_like(padded_values) for _ in range(size)]
        dist.all_gather(gathered_rows, padded_rows, group=self.group)
        dist.all_gather(gathered_values, padded_values, group=self.group)

        rows_parts, values_parts = [], []
        for replica, count in enumerate(counts.tolist()):
            if count:
                rows_parts.append(gathered_rows[replica][:count].to("cpu", copy=True))
                values_parts.append(gathered_values[replica][:count].to("cpu", copy=True))
        return self._coalesce(rows_parts, values_parts)


def _named_modules(model):
    """``named_modules()`` of a real model; nothing for a stand-in (a unit-test stub, for instance)."""
    named_modules = getattr(model, "named_modules", None)
    return named_modules() if callable(named_modules) else ()


def get_vocab_embeddings(model) -> list:
    """``(name, module)`` of every host-resident embedding: the ones whose gradient lands in a row ledger."""
    return [
        (name, module)
        for name, module in _named_modules(model)
        if getattr(module, "_vp_row_grad", None) is not None
    ]


def get_device_vocab_embeddings(model) -> list:
    """``(name, module)`` of every embedding on the accelerator, updated by the model optimizer directly."""
    return [
        (name, module)
        for name, module in _named_modules(model)
        if getattr(module, "_vp_dense_embedding", False)
    ]


class _HostRowLookup(torch.autograd.Function):
    """Host lookup whose backward accumulates row gradients instead of a dense embedding gradient.

    ``async_copy`` issues the host-to-device copy on the side stream; the event it needs is appended
    to ``ready`` (a one-element list supplied by the caller), because a copy that has not landed yet
    must not be read and the caller is the only one that knows when the rows are consumed.
    """

    @staticmethod
    def forward(ctx, weight, indices, accumulator, device, async_copy, ready):
        ctx.accumulator = accumulator
        ctx.save_for_backward(indices)
        rows, event = fetch_rows_from_host(weight, indices, device, async_copy=async_copy)
        if ready is not None:
            ready.append(event)
        return rows

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        ctx.accumulator.add(indices, grad_output)
        return None, None, None, None, None, None


def host_row_lookup(
    weight: torch.Tensor,
    indices: torch.Tensor,
    accumulator: VocabRowGrad,
    device,
    async_copy: bool = False,
    ready: list | None = None,
):
    """Look up ``indices`` rows of the host-resident ``weight``; gradients go to ``accumulator``."""
    return _HostRowLookup.apply(weight, indices, accumulator, device, async_copy, ready)
