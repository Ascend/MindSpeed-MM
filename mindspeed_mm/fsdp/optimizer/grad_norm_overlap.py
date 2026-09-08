"""Overlap gradient-norm computation with backward (FSDP2).

When enabled (`features.enable_grad_norm_overlap`), per-tensor gradient norms
are computed right after each FSDP2 param group's `foreach_reduce`, instead of
serially reading all gradients after backward. Two execution paths share one
registry:

- CPU worker thread: for CPU-offloaded gradients, once the non-blocking D2H
  lands them on pinned host memory (the job's `d2h_event`).
- Device side stream: for device-resident gradients, on a dedicated stream
  gated by the group's `post_reduce_event`, overlapping with the remaining
  backward of later groups.

The manager is FSDP-agnostic: the fully_shard patch layer extracts plain
`(param, grad_local, event)` jobs from FSDPParam internals (see
`_collect_grad_norm_jobs` in fully_shard.py) and arms the hook when it patches
`post_backward` (`mark_hook_planted`). `configure()` therefore needs no torch
version table of its own — torch versions without a patched post_backward
simply never get the hook, and configure() keeps the feature off.

Bitwise-identity design: both paths only compute *per-tensor* p-norms in fp32
(the dtype=torch.float32 arg hits the same kernel as fp32 inputs, so
uniform-fp32 groups are unaffected); the consumer reassembles the full norm
list in the original `model.parameters()` order, groups by the grads'
(device, dtype) exactly like the serial `_local_pth_sum`, and reduces with
the same kernels/values/order, so the result is bitwise identical by
construction for uniform-dtype models. Mixed-dtype models (more than one
grad dtype on the same device) are NOT reassembled: the serial path
materializes every grad to fp32 first (single group per device), which
changes summation order, so consume() falls back to the serial path there
instead of silently diverging. Intermediate invocations of a param group may
read torn accumulations — their results are always overwritten by the group's
last invocation before consumption (single FIFO worker / single side stream
guarantee order), and the consumed last invocation races no writer
(structural order: consume runs after backward, before
optimizer.step/zero_grad).

Stream-lifetime argument for the device path (no record_stream needed):
norm outputs are allocated on the side stream and consumed on the current
stream after `wait_event(last_norm_event)`; any later reuse of the side
stream is gated on a later `post_reduce_event`, which `foreach_reduce`
gates on `reduce_scatter_stream.wait_stream(current_stream)` — i.e. on the
current stream having already passed the consumption point. Fallback paths
close the same edge via `_wait_device_norms()` before returning None, so
in-flight side-stream reads never race the serial path's clip writes.
"""

import logging
import queue
import threading
from typing import List, Optional

import torch

from mindspeed_mm.fsdp.utils.device import (
    create_event,
    create_stream,
    get_current_stream,
    switch_to_specified_stream,
)

logger = logging.getLogger(__name__)


class _GradNormOverlapManager:
    # The in-repo training pipeline always clips with 2-norm (there is no
    # norm_type config key). Per-tensor norms are computed at record time, so
    # p must be known up front; a consume-time mismatch disables the feature.
    _SUPPORTED_P = 2.0

    def __init__(self):
        self.enabled = False
        self._hook_planted = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        # id(param) -> ((grad_device, grad_dtype), 0-dim fp32 norm tensor)
        self._norms = {}
        self._error: Optional[BaseException] = None
        self._warned = set()
        self._norm_stream = None  # lazily created side stream (device path)
        self._last_event = None  # last event recorded on _norm_stream

    # ---- wiring ----
    def mark_hook_planted(self) -> None:
        """Called by the fully_shard patch layer when it installs the record
        hook into a vendored post_backward. Without the mark there is no
        producer for this torch version, and configure() keeps the feature off."""
        self._hook_planted = True

    def configure(self, enabled: bool) -> None:
        """Enable/disable the overlap path. Call once per process after args are parsed."""
        if not enabled:
            return
        if not self._hook_planted:
            logger.warning("enable_grad_norm_overlap is set, but no patched post_backward "
                           "exists for this torch version; keeping the serial grad-norm path.")
            return
        self.enabled = True
        logger.info("grad norm overlap enabled (CPU worker path for offloaded grads, "
                    "device side-stream path for device-resident grads).")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(msg)

    # ---- producer: fully_shard hook, tail of patched post_backward ----
    def record(self, cpu_jobs, dev_jobs, post_reduce_event=None) -> None:
        """Take over this param group's reduced grads for norm computation.

        cpu_jobs: [(param, grad_local, d2h_event_or_None)] for CPU-offloaded grads.
        dev_jobs: [(param, grad_local)] for device-resident grads.
        """
        if not self.enabled:
            return
        if self._error is not None:
            return  # consume will raise; stop enqueueing
        if cpu_jobs:
            self._ensure_worker()
            self._queue.put(cpu_jobs)
        if dev_jobs:
            self._record_device(dev_jobs, post_reduce_event)

    def _record_device(self, dev_jobs, post_reduce_event) -> None:
        """Enqueue per-tensor norms on the side stream (device-resident grads)."""
        if post_reduce_event is None:
            self._warn_once("no_event", "grad norm overlap: post_reduce_event unavailable for "
                                        "device-resident grads; skipping device path "
                                        "(serial fallback at consume).")
            return
        if self._norm_stream is None:
            self._norm_stream = create_stream()
        stream = self._norm_stream
        with switch_to_specified_stream(stream):
            stream.wait_event(post_reduce_event)  # this group's RS/accumulate writes
            grads = [j[1] for j in dev_jobs]  # grads stay alive via sharded_param refs
            with torch.no_grad():
                # Always compute in fp32: for fp32 inputs the dtype arg selects the
                # same kernel bitwise (probe-verified); for anything else it matches
                # the serial path, which computes every non-fp32 group in fp32.
                norms = torch._foreach_norm(grads, self._SUPPORTED_P, dtype=torch.float32)
            for (param, local), norm in zip(dev_jobs, norms):
                key = (local.device, local.dtype)
                self._norms[id(param)] = (key, norm)  # overwrite: last invocation wins
            self._last_event = create_event()
            self._last_event.record(stream)

    # ---- worker thread (CPU ops only) ----
    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._queue = queue.Queue()
            self._worker = threading.Thread(
                target=self._work, name="grad-norm-overlap", daemon=True)
            self._worker.start()

    def _work(self) -> None:
        while True:
            jobs = self._queue.get()
            try:
                for _, _, ev in jobs:
                    if ev is not None:
                        ev.synchronize()  # host-wait the non-blocking D2H (new-assignment path)
                grads = [j[1] for j in jobs]  # refs held by the job until read is done
                if grads:
                    with torch.no_grad():
                        # Same fp32 rule as the device path (see _record_device).
                        norms = torch._foreach_norm(grads, self._SUPPORTED_P, dtype=torch.float32)
                    for (param, local, _), n in zip(jobs, norms):
                        key = (local.device, local.dtype)
                        self._norms[id(param)] = (key, n)  # overwrite: last invocation wins
            except Exception as exc:  # noqa: BLE001 - surfaced loudly at consume
                self._error = exc
                logger.exception("grad norm overlap worker failed")
            finally:
                self._queue.task_done()

    # ---- consumer: main thread, fast path of _fsdp2_reduce_group ----
    def consume(self, params: List[torch.nn.Parameter], p: float) -> Optional[torch.Tensor]:
        """Return the local sum of p-th powers if every param is covered, else None (fallback)."""
        if not self.enabled:
            return None
        if self._worker is None and not self._norms:
            return None
        if self._worker is not None:
            self._queue.join()
        if self._error is not None:
            raise RuntimeError("grad norm overlap worker error") from self._error
        if p != self._SUPPORTED_P:
            self._warn_once("p_mismatch", f"grad norm overlap supports p={self._SUPPORTED_P} only "
                                          f"(clip requested p={p}); disabling the feature.")
            self.enabled = False
            self._wait_device_norms()
            return None
        covered = [param for param in params if param.grad is not None]  # same filter as serial
        if not covered:
            return None  # nothing to reduce; the serial path handles this trivially
        entries = []
        for param in covered:
            entry = self._norms.get(id(param))
            if entry is None:
                self._warn_once("uncovered", "grad norm overlap: param not covered by any group "
                                             "invocation (never-fired or non-FSDP param); falling "
                                             "back to serial path.")
                self._wait_device_norms()
                return None
            entries.append(entry)
        # The serial path materializes mixed-dtype grads to fp32 first (one group
        # per device); reassembling by original (device, dtype) would change the
        # summation order, so fall back instead of silently diverging.
        dtypes_per_device = {}
        for key, _ in entries:
            dtypes_per_device.setdefault(key[0], set()).add(key[1])
        if any(len(dtypes) > 1 for dtypes in dtypes_per_device.values()):
            self._warn_once("mixed_dtype", "grad norm overlap: mixed grad dtypes on one device "
                                           "are unsupported; falling back to serial path.")
            self._wait_device_norms()
            return None
        for param in covered:
            self._norms.pop(id(param), None)
        # Reduce exactly like _local_pth_sum: group by the grads' (device, dtype) in
        # first-occurrence order; per group foreach_pow_ -> stack -> sum -> res += .
        self._wait_device_norms()
        default_device = entries[0][0][0]
        res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
        groups = {}
        for key, norm in entries:
            groups.setdefault(key, []).append(norm)
        for norms in groups.values():
            out = torch._foreach_pow_(norms, p)
            res += torch.sum(torch.stack(out)).to(default_device)
        return res

    def _wait_device_norms(self) -> None:
        """Close the side-stream edge: in-flight device norm reads finish before
        the current stream proceeds (consumed on the fast path; merely awaited on
        fallback paths so serial clip writes cannot race them)."""
        if self._last_event is not None:
            get_current_stream().wait_event(self._last_event)

    def clear(self) -> None:
        """Drop leftover entries (e.g. fired but never consumed). Wired from
        the feature layer at the step boundary so staleness cannot accumulate."""
        if not self.enabled:
            return
        self._norms.clear()


manager = _GradNormOverlapManager()
