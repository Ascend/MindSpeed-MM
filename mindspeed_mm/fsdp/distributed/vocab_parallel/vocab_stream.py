"""Side stream for the vocab row-gradient transfers, so a host<->device copy overlaps the compute stream.

``vocab_parallelize_modules`` calls :func:`configure` once, for a process that applied vocab
parallelism, with ``VocabPlanConfig.async_offload``.
It is deliberately independent of ``VocabPlanConfig.prefetch``: the two cover different transfers (the
prefetch hoists the forward lookup, this stream hides the backward copies), so gating one behind the
other would make an A/B of either measure both. The stream is created on first use, and the wait
returns immediately while nothing has been issued.

Two paths use it. The backward row gradients go to a pinned buffer and the host reads that buffer
later, in ``_coalesce``, so the copy only has to be done before the end of the step. The forward row
transfer blocks by default, because the dtype cast and the embedding all-to-all consume the rows
inside the issuing call; when the dispatcher is split (see ``VocabPlanConfig.prefetch_split``) it is
issued here instead and the consuming half waits for :func:`mark`'s event.

Both directions need an explicit dependency: the side stream waits for the compute stream that
produced the rows being copied (:func:`issue`), and the host waits for the copy before reading the
pinned ledger (:func:`host_wait`). A stream dependency is not enough for the second one -- it does
not keep the host from reading a buffer whose copy is still in flight.
"""
import contextlib

from mindspeed_mm.fsdp.utils.device import get_torch_device

_ENABLED = False        # off until configure() says vocab parallelism is applied
_STREAM = None
_EVENT = None
# Buffers a copy is still reading from. The copies are issued outside the allocator's stream
# bookkeeping, so the sources stay referenced until the matching wait (a dropped reference could
# let the caching allocator hand the memory to a new tensor while the copy is in flight).
_KEEPALIVE = []


def configure(enabled: bool = True) -> None:
    """Enable or disable the side stream for this process (see VocabPlanConfig.async_offload)."""
    global _ENABLED
    _ENABLED = bool(enabled)


def enabled() -> bool:
    return _ENABLED


def issue():
    """Context manager to issue a copy on: the side stream, or a no-op when the transfer is disabled.

    The side stream first waits for the work already queued on the compute stream: everything copied
    here was produced there (the step's row gradients), and two streams have no ordering of their own.
    """
    if not _ENABLED:
        return contextlib.nullcontext()
    global _STREAM
    if _STREAM is None:
        _STREAM = get_torch_device().Stream()
    _STREAM.wait_stream(get_torch_device().current_stream())
    return get_torch_device().stream(_STREAM)


def keep(tensor) -> None:
    """Hold ``tensor`` alive until the next wait."""
    if _ENABLED:
        _KEEPALIVE.append(tensor)


def record() -> None:
    """Record an event on the side stream, for consumers that read the buffer on the host."""
    if _STREAM is None:
        return
    global _EVENT
    if _EVENT is None:
        _EVENT = get_torch_device().Event()
    _EVENT.record(_STREAM)


def mark():
    """Record a fresh event on the side stream and return it, for a consumer on the compute stream.

    One event per copy: the forward lookup and the backward row-gradient transfer both use the side
    stream, and a shared event would let one wait pretend the other's copy had landed.
    """
    if not _ENABLED or _STREAM is None:
        return None
    event = get_torch_device().Event()
    event.record(_STREAM)
    return event


def wait(event) -> None:
    """Make the compute stream wait for a copy recorded by :func:`mark`.

    Only the consumer stalls, and only until that copy lands; work queued before it keeps running.
    """
    if event is None or not _ENABLED:
        return
    get_torch_device().current_stream().wait_event(event)


def host_wait() -> None:
    """Block the host until the copies recorded on the side stream are done."""
    if _EVENT is None:
        return
    _EVENT.synchronize()
    _KEEPALIVE.clear()
