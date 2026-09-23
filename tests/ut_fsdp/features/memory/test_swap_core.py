"""White-box tests for SwapCore's stream/event anchoring.

D2H is ordered after the handle's put event (recorded at put, the contract
point where the tensor's content becomes final), never after the
eviction-time ambient stream. Release ordering anchors the put stream (the
HBM block returns to its allocation pool, which the allocator binds to the
allocation stream's pool), never the release-time ambient; swap_in lands in
the put stream's pool with the H2D waiting the put stream; _wait_h2d settles
on both the consume stream and the put stream, so a zero-cost
REPLICATED eviction needs no wait of its own. These tests spy the
stream/event calls; they do not depend on race probabilities and need no
device.
"""
import contextlib

import torch

from mindspeed_mm.fsdp.features.memory import swap_core as swap_core_mod
from mindspeed_mm.fsdp.features.memory.swap_core import SwapHandle, SwapState


class _SpyEvent:
    def __init__(self):
        self.recorded_on = None
        self.wait_calls = []

    def record(self, stream):
        self.recorded_on = stream

    def wait(self, stream):
        self.wait_calls.append(stream)


class _SpyStream:
    def __init__(self, name):
        self.name = name
        self.wait_stream_calls = []
        self.wait_event_calls = []

    def wait_stream(self, stream):
        self.wait_stream_calls.append(stream)

    def wait_event(self, event):
        self.wait_event_calls.append(event)


class _FakeBuf:
    def __init__(self, state=None):
        self._state = state
        self.landing_allocated_on = None

    def store_from(self, tensor, non_blocking=True):
        pass

    def prepare_landing(self):
        if self._state is not None:
            self.landing_allocated_on = self._state['cur']
        return torch.zeros(16)

    def load_into(self, landing, non_blocking=True):
        return landing


class _FakeArena:
    def __init__(self, state=None):
        self._state = state
        self.buf = None

    def payload_bytes(self, tensor):
        return tensor.nbytes

    def register(self, tensor):
        self.buf = _FakeBuf(self._state)
        return self.buf

    def free(self, buf):
        pass


def _patch_stream_layer(monkeypatch):
    """Replace the device stream/event layer with spies; returns a mutable
    holder whose 'cur' is the ambient stream. switch_to_specified_stream
    really switches the ambient (and restores on exit)."""
    state = {'cur': _SpyStream('ambient-at-put')}
    monkeypatch.setattr(swap_core_mod, 'create_event', _SpyEvent)
    monkeypatch.setattr(swap_core_mod, 'get_current_stream',
                        lambda: state['cur'])

    @contextlib.contextmanager
    def _switch(stream):
        prev = state['cur']
        state['cur'] = stream
        try:
            yield
        finally:
            state['cur'] = prev

    monkeypatch.setattr(swap_core_mod, 'switch_to_specified_stream', _switch)
    return state


def test_put_anchor_captured_on_put_time_ambient(monkeypatch):
    state = _patch_stream_layer(monkeypatch)
    handle = SwapHandle(torch.zeros(16), h2d_stream=_SpyStream('swap'),
                        d2h_stream=_SpyStream('swap'), cpu_arena=_FakeArena())
    assert handle._put_stream is state['cur']
    assert handle._put_event.recorded_on is state['cur']


def test_swap_out_waits_put_event_not_eviction_ambient(monkeypatch):
    state = _patch_stream_layer(monkeypatch)
    d2h = _SpyStream('d2h')
    handle = SwapHandle(torch.zeros(16), h2d_stream=d2h, d2h_stream=d2h,
                        cpu_arena=_FakeArena())

    # Eviction fires later, under an unrelated ambient context (e.g. another
    # put inside a model stream region forked off the compute stream).
    state['cur'] = _SpyStream('ambient-at-eviction')
    handle.swap_out()

    assert d2h.wait_event_calls == [handle._put_event]
    assert d2h.wait_stream_calls == []


def test_wait_d2h_anchors_put_stream_under_foreign_ambient(monkeypatch):
    # Releasing (trim/clear) under a foreign ambient must still order
    # the put stream after the D2H read — the freed block returns to the
    # allocation stream's pool regardless of the free-time ambient.
    state = _patch_stream_layer(monkeypatch)
    put_stream = state['cur']
    swap = _SpyStream('swap')
    handle = SwapHandle(torch.zeros(16), h2d_stream=swap, d2h_stream=swap,
                        cpu_arena=_FakeArena())
    handle.swap_out()

    foreign = _SpyStream('ambient-at-release')
    state['cur'] = foreign
    handle.wait()

    assert handle._d2h_event.wait_calls == [foreign, put_stream]
    assert handle.state() is SwapState.DDR_ONLY


def test_wait_d2h_single_stream_settles_once(monkeypatch):
    state = _patch_stream_layer(monkeypatch)
    put_stream = state['cur']
    swap = _SpyStream('swap')
    handle = SwapHandle(torch.zeros(16), h2d_stream=swap, d2h_stream=swap,
                        cpu_arena=_FakeArena())
    handle.swap_out()

    handle.wait()  # same ambient as put: the put-stream anchor degenerates

    assert handle._d2h_event.wait_calls == [put_stream]


def _offloaded_handle(monkeypatch):
    """A handle whose tensor has reached DDR_ONLY (D2H settled at put)."""
    state = _patch_stream_layer(monkeypatch)
    swap = _SpyStream('swap')
    arena = _FakeArena(state)
    handle = SwapHandle(torch.zeros(16), h2d_stream=swap, d2h_stream=swap,
                        cpu_arena=arena)
    handle.swap_out()
    handle.wait()
    assert handle.state() is SwapState.DDR_ONLY
    return state, swap, arena, handle


def test_swap_in_lands_in_put_stream_pool_and_waits_put_stream(monkeypatch):
    # The landing is allocated under the put stream's context (its pool),
    # and the H2D waits the put stream — never the consume-time ambient.
    state, swap, arena, handle = _offloaded_handle(monkeypatch)
    put_stream = state['cur']

    state['cur'] = _SpyStream('ambient-at-consume')
    handle.swap_in()

    assert arena.buf.landing_allocated_on is put_stream
    assert swap.wait_stream_calls == [put_stream]
    assert handle.state() is SwapState.H2D_IN_PROGRESS


def test_wait_h2d_dual_settle_covers_zero_cost_eviction(monkeypatch):
    # _wait_h2d settles on both the consume stream and the put
    # stream, so the later zero-cost REPLICATED eviction can drop the tensor
    # with no wait of its own.
    state, swap, arena, handle = _offloaded_handle(monkeypatch)
    put_stream = state['cur']

    consume = _SpyStream('ambient-at-consume')
    state['cur'] = consume
    handle.swap_in()
    handle.wait()

    assert handle._h2d_event.wait_calls == [consume, put_stream]
    assert handle.state() is SwapState.REPLICATED

    # Zero-cost eviction: no D2H issued, no further ordering edges.
    handle.swap_out()
    assert handle.state() is SwapState.DDR_ONLY
    assert len(handle._d2h_event.wait_calls) == 1  # only the settle above
    assert swap.wait_event_calls == [handle._put_event]


def test_consume_d2h_in_progress_settles_both_and_accounts(monkeypatch):
    # consume() from D2H_IN_PROGRESS: the tensor is captured before clear and
    # handed out with BOTH orderings settled (consume stream for the
    # consumer, put stream for the pool the block returns to); accounting
    # releases both sides and the handle ends EMPTY.
    state = _patch_stream_layer(monkeypatch)
    put_stream = state['cur']
    swap = _SpyStream('swap')
    handle = SwapHandle(torch.zeros(16), h2d_stream=swap, d2h_stream=swap,
                        cpu_arena=_FakeArena())
    handle.swap_out()

    consume = _SpyStream('ambient-at-consume')
    state['cur'] = consume
    tensor, delta = handle.consume()

    assert tensor is not None
    assert handle._d2h_event.wait_calls == [consume, put_stream]
    assert handle.state() is SwapState.EMPTY
    assert (delta.hbm, delta.ddr) == (-64, -64)  # float32 zeros(16) = 64 B
