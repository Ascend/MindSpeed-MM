"""Differentiable chunk cache and layer-level CPU offload."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import torch

from mindspeed_mm.fsdp.utils.decorators import Singleton
from mindspeed_mm.fsdp.utils.device import (
    create_stream,
    create_event,
    get_current_stream,
    switch_to_specified_stream,
)
from mindspeed_mm.fsdp.features.memory.slab_arena import SlabArena
from mindspeed_mm.fsdp.features.memory.swap_core import (
    SwapCache,
    SwapHandle as TensorSwapHandle,
)


CachePair = tuple[torch.Tensor, torch.Tensor]
_FA_COMPONENTS = {"k": 0, "v": 1}


@dataclass(frozen=True)
class CacheSpec:
    """Location of one differentiable cache tensor in a layer cache."""

    cache_key: str
    component: str
    chain_key: str
    chunk_idx: int
    reused_tokens: int = 0


class CacheHandle:
    """One differentiable cache tensor and its location in layer storage."""

    def __init__(self, tensor: torch.Tensor, spec: CacheSpec):
        self.tensor = tensor
        self.spec = spec
        self.copy_event = None
        self.source_released = False
        self.bound_to_cache = False

    @property
    def record_key(self) -> tuple[str, str, int]:
        return self.spec.component, self.spec.chain_key, self.spec.chunk_idx

def _launch_d2d(source: torch.Tensor, target: torch.Tensor, transfer_stream):
    producer_event = create_event()
    producer_event.record()
    copy_event = create_event()
    source.record_stream(transfer_stream)
    with torch.no_grad(), switch_to_specified_stream(transfer_stream):
        transfer_stream.wait_event(producer_event)
        target.copy_(source, non_blocking=True)
        copy_event.record()
    return copy_event


class LayerCache(ABC):
    """Common compute and storage contract for one decoder-layer cache."""

    def __init__(self, plan, layer_idx: int):
        self.key = plan.cache_key
        self.forward_id = plan.forward_id
        self.layer_idx = layer_idx
        self.layout = plan.layout
        self.batch_size = plan.batch_size
        self.num_chunks = plan.num_chunks
        self.layer_capacity = plan.sequence_length
        self.chunk_infos = plan.chunk_infos
        self.device = None
        self.state = "staging"
        self.offload = False

        self.handles: list[CacheHandle] = []
        self.handles_by_chunk: dict[int, list[CacheHandle]] = defaultdict(list)
        self.copy_events: dict[tuple[str, str, int], object] = {}
        self.released_storage_ids = set()
        self.device_tensors: dict[str, torch.Tensor] = {}
        self.swap_handles: dict[str, TensorSwapHandle] = {}

    @abstractmethod
    def prepare_cache(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> tuple[CachePair, dict]:
        """Combine the current chunk cache with its differentiable history."""

    @abstractmethod
    def append(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> CachePair:
        """Select and describe the cache carried to the next chunk."""

    @abstractmethod
    def stage(self, handle: CacheHandle, transfer_stream) -> None:
        """Copy one saved tensor into this layer's compact NPU cache."""

    @abstractmethod
    def view_for(self, handle: CacheHandle) -> torch.Tensor:
        """Return the restored NPU view for a saved tensor."""

    def swap_out(self, swap_cache: SwapCache) -> None:
        """Hand the completed compact cache to SwapCore."""
        if self.state != "staging":
            raise RuntimeError(f"layer cache is not staging: {self.key}")
        if not self.device_tensors:
            raise RuntimeError(f"layer cache has no device tensors: {self.key}")
        if self.swap_handles:
            raise RuntimeError(f"layer cache already owns swap handles: {self.key}")

        self._validate_before_swap_out()
        self.swap_handles = {
            name: swap_cache.put(tensor)
            for name, tensor in self.device_tensors.items()
        }
        # SwapCore owns the physical HBM/DDR lifecycle from this point.  Do not
        # resize the compact bank's storage: capacity eviction drops SwapCore's
        # HBM reference after the ordered D2H completes.
        self.device_tensors.clear()
        self.state = "managed"

    def swap_in(self, swap_cache: SwapCache) -> None:
        """Materialize the compact cache on device and bind its logical views."""
        if self.state == "device":
            return
        if self.state != "managed":
            raise RuntimeError(f"layer cache is not managed by SwapCore: {self.key}")
        if self.device_tensors:
            raise RuntimeError(f"managed cache still owns device tensors: {self.key}")

        restored = {}
        for name, swap_handle in self.swap_handles.items():
            tensor = swap_cache.pop(swap_handle)
            if tensor is None:
                raise RuntimeError(
                    f"SwapCore handle is missing for layer cache {self.key}/{name}"
                )
            restored[name] = tensor
        self.device_tensors = restored
        self.swap_handles.clear()

        for handle in self.handles:
            if handle.bound_to_cache:
                continue
            view = self.view_for(handle)
            with torch.no_grad():
                handle.tensor.set_(
                    view.untyped_storage(),
                    view.storage_offset(),
                    view.size(),
                    view.stride(),
                )
            handle.bound_to_cache = True
        self.state = "device"

    def release_materialized(self) -> None:
        """Drop layer-owned references after this layer's backward completes."""
        self.device_tensors.clear()
        self.swap_handles.clear()
        self.handles.clear()
        self.handles_by_chunk.clear()
        self.copy_events.clear()
        self.released_storage_ids.clear()
        self.state = "released"

    def _validate_before_swap_out(self) -> None:
        """Validate subclass-specific compact-cache completeness."""

    def add_handle(self, handle: CacheHandle, transfer_stream) -> None:
        if handle.spec.cache_key != self.key:
            raise RuntimeError(f"cache key mismatch for {self.key}")
        if not 0 <= handle.spec.chunk_idx < self.num_chunks:
            raise RuntimeError(f"invalid cache chunk index for {self.key}")
        if self.state != "staging":
            raise RuntimeError(f"cannot add tensors after cache finalization: {self.key}")
        if self.device is None:
            self.device = handle.tensor.device
        elif handle.tensor.device != self.device:
            raise RuntimeError(f"cache device changed within layer {self.key}")
        self.stage(handle, transfer_stream)
        self.handles.append(handle)
        self.handles_by_chunk[handle.spec.chunk_idx].append(handle)

    def complete_chunk(self, consumer_chunk_idx: int) -> None:
        """Release the preceding carry once this chunk has consumed it."""
        if self.state != "staging":
            return None
        current_stream = get_current_stream()
        for handle in self.handles_by_chunk.pop(consumer_chunk_idx - 1, ()):
            if handle.copy_event is not None:
                current_stream.wait_event(handle.copy_event)
            storage = handle.tensor.untyped_storage()
            storage_id = storage._cdata
            if storage_id not in self.released_storage_ids:
                storage.resize_(0)
                self.released_storage_ids.add(storage_id)
            handle.source_released = True

    def _carry_tensor(
        self,
        tensor: torch.Tensor,
        spec: CacheSpec | None,
        *,
        always_clone: bool,
    ) -> torch.Tensor:
        if always_clone or tensor._base is not None:
            tensor = tensor.clone()
        if spec is not None and self.offload:
            if self.state != "staging":
                return
            handle = CacheHandle(tensor, spec)
            self.add_handle(handle, CacheOffloadManager().swap_stream)
        return tensor


class FACache(LayerCache):
    """Full-attention K/V cache with incremental token storage."""

    def __init__(self, plan, layer_idx: int):
        super().__init__(plan, layer_idx)
        chain_lengths: dict[str, int] = {}
        if self.layout == "bsnd":
            if self.num_chunks > 1:
                chain_lengths["batch"] = self.chunk_infos[-1]["start"]
        else:
            for chunk_idx, chunk_info in enumerate(self.chunk_infos):
                if not chunk_info["carries_last_doc"]:
                    continue
                doc_id = chunk_info["doc_ids"][-1]
                doc_start = chunk_info["doc_ranges"][doc_id][0]
                carried_tokens = chunk_info["end"] - doc_start
                chain_key = f"doc{doc_id}"
                chain_lengths[chain_key] = max(
                    chain_lengths.get(chain_key, 0), carried_tokens
                )

        self.chain_slots: dict[str, tuple[int, int]] = {}
        offset = 0
        for chain_key, length in chain_lengths.items():
            self.chain_slots[chain_key] = (offset, length)
            offset += length
        self.storage_tokens = offset

    def prepare_cache(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> tuple[CachePair, dict]:
        key, value = cache
        chunk_info = self.chunk_infos[chunk_idx]
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError("full-attention K/V must use [B, N, T, D] layout")
        if key.shape[:3] != value.shape[:3]:
            raise ValueError("full-attention K/V must agree in B/N/T dimensions")
        if key.shape[0] != self.batch_size:
            raise ValueError("full-attention K/V batch size does not match chunk plan")
        history_k, history_v = history_cache
        if (history_k is None) != (history_v is None):
            raise ValueError("full-attention K/V history must be provided as a pair")

        if self.layout == "bsnd":
            if history_k is not None and history_k.numel() != 0:
                key = torch.cat((history_k, key), dim=2)
                value = torch.cat((history_v, value), dim=2)
            return (key, value), {"max_length_k": key.shape[2]}
        elif self.layout == "tnd":
            # Reuse the owned single-document concat as the next carry; slicing it
            # back to a view would clone the growing history in _carry_tensor().
            if len(chunk_info["doc_ids"]) == 1:
                if history_k is not None and history_k.numel() != 0:
                    key = torch.cat((history_k, key), dim=2)
                    value = torch.cat((history_v, value), dim=2)
                cu_seqlens_k = torch.tensor(
                    [0, key.shape[2]], dtype=torch.int64, device=key.device
                )
                return (key, value), {
                    "cu_seq_lens_k": cu_seqlens_k,
                    "max_length_k": key.shape[2],
                }

            key_parts = []
            value_parts = []
            lengths = []
            local_cu_seqlens = chunk_info["local_cu_seqlens"]
            for local_doc_idx in range(len(chunk_info["doc_ids"])):
                start = int(local_cu_seqlens[local_doc_idx])
                end = int(local_cu_seqlens[local_doc_idx + 1])
                doc_k = key[:, :, start:end, :]
                doc_v = value[:, :, start:end, :]
                if (
                    local_doc_idx == 0
                    and history_k is not None
                    and history_k.numel() != 0
                ):
                    doc_k = torch.cat((history_k, doc_k), dim=2)
                    doc_v = torch.cat((history_v, doc_v), dim=2)
                key_parts.append(doc_k)
                value_parts.append(doc_v)
                lengths.append(doc_k.shape[2])

            key = torch.cat(key_parts, dim=2)
            value = torch.cat(value_parts, dim=2)
            cu_seqlens_k = torch.tensor(
                [0, *lengths], dtype=torch.int64, device=key.device
            ).cumsum(0)
            return (key, value), {
                "cu_seq_lens_k": cu_seqlens_k,
                "max_length_k": max(lengths),
            }

    def append(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> CachePair:
        key, value = cache
        chunk_info = self.chunk_infos[chunk_idx]
        if not chunk_info["carries_last_doc"]:
            empty_k = key[:, :, :0, :]
            empty_v = value[:, :, :0, :]
            return (
                self._carry_tensor(empty_k, None, always_clone=False),
                self._carry_tensor(empty_v, None, always_clone=False),
            )

        history_k, _ = history_cache
        if self.layout == "bsnd":
            next_k, next_v = key, value
            reused = 0 if history_k is None else history_k.shape[2]
            chain_key = "batch"
        elif self.layout == 'tnd':
            if len(chunk_info["doc_ids"]) == 1:
                next_k, next_v = key, value
                reused = (
                    history_k.shape[2]
                    if history_k is not None and history_k.numel() != 0
                    else 0
                )
                doc_id = chunk_info["doc_ids"][0]
                chain_key = f"doc{doc_id}"
            else:
                local_cu_seqlens = chunk_info["local_cu_seqlens"]
                lengths = [
                    int(local_cu_seqlens[idx + 1] - local_cu_seqlens[idx])
                    for idx in range(len(chunk_info["doc_ids"]))
                ]
                if history_k is not None and history_k.numel() != 0:
                    lengths[0] += history_k.shape[2]
                last_length = lengths[-1]
                last_start = sum(lengths[:-1])
                next_k = key[:, :, last_start:last_start + last_length, :]
                next_v = value[:, :, last_start:last_start + last_length, :]
                reused = (
                    history_k.shape[2]
                    if len(chunk_info["doc_ids"]) == 1
                    and history_k is not None
                    and history_k.numel() != 0
                    else 0
                )
                doc_id = chunk_info["doc_ids"][-1]
                chain_key = f"doc{doc_id}"

        common = {
            "cache_key": self.key,
            "chain_key": chain_key,
            "chunk_idx": chunk_idx,
            "reused_tokens": reused,
        }
        return (
            self._carry_tensor(
                next_k, CacheSpec(component="k", **common), always_clone=False
            ),
            self._carry_tensor(
                next_v, CacheSpec(component="v", **common), always_clone=False
            ),
        )

    def stage(self, handle: CacheHandle, transfer_stream) -> None:
        tensor, spec = handle.tensor, handle.spec
        if tensor.ndim != 4 or spec.component not in _FA_COMPONENTS:
            raise ValueError("FA cache must be K or V in [B, N, T, D] format")
        device_cache = self.device_tensors.get("fa")
        if device_cache is None:
            batch_size, num_heads, _, head_dim = tensor.shape
            device_cache = torch.empty(
                (2, batch_size, num_heads, self.storage_tokens, head_dim),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            self.device_tensors["fa"] = device_cache
        expected = (
            device_cache.shape[1],
            device_cache.shape[2],
            device_cache.shape[4],
        )
        if (tensor.shape[0], tensor.shape[1], tensor.shape[3]) != expected:
            raise RuntimeError("FA cache B/N/D dimensions changed within one layer")
        if tensor.dtype != device_cache.dtype:
            raise RuntimeError("FA cache dtype changed within one layer")

        slot = self.chain_slots.get(spec.chain_key)
        if slot is None:
            raise RuntimeError(
                f"FA chain is absent from the chunk plan: {spec.chain_key}"
            )
        chain_offset, chain_length = slot

        if not 0 <= spec.reused_tokens <= tensor.shape[2]:
            raise RuntimeError("invalid reused-token count")
        # Copy only tokens not already stored by the previous carry.
        new_tokens = tensor.shape[2] - spec.reused_tokens
        if tensor.shape[2] > chain_length:
            raise RuntimeError(f"FA KV exceeds chain capacity for {spec.chain_key}")

        event = self.copy_events.get(handle.record_key)
        if event is None:
            offset = chain_offset + spec.reused_tokens
            # The new-token slice is a view of the growing concatenated K/V
            # storage.  Recording that view on the transfer stream would keep
            # the complete history allocation alive until the asynchronous
            # copy finishes.  Stage an owned compact source so cross-stream
            # lifetime tracking retains only the tokens that are copied.
            with torch.no_grad():
                compact_source = (
                    tensor[:, :, spec.reused_tokens:, :].detach().clone()
                )
            event = _launch_d2d(
                compact_source,
                device_cache[
                    _FA_COMPONENTS[spec.component],
                    :,
                    :,
                    offset:offset + new_tokens,
                    :,
                ],
                transfer_stream,
            )
            self.copy_events[handle.record_key] = event
        handle.copy_event = event

    def view_for(self, handle: CacheHandle) -> torch.Tensor:
        offset, max_length = self.chain_slots[handle.spec.chain_key]
        length = handle.tensor.shape[2]
        if length > max_length:
            raise RuntimeError("saved FA KV exceeds its compact chain")
        return self.device_tensors["fa"][
            _FA_COMPONENTS[handle.spec.component],
            :,
            :,
            offset:offset + length,
            :,
        ]


class GDNCache(LayerCache):
    """GDN convolution/recurrent state cache."""

    def __init__(self, plan, layer_idx: int):
        super().__init__(plan, layer_idx)
        carry_chunks = [
            chunk_idx
            for chunk_idx, chunk_info in enumerate(self.chunk_infos)
            if chunk_info["carries_last_doc"]
        ]
        self.chunk_slots = {
            chunk_idx: compact_idx
            for compact_idx, chunk_idx in enumerate(carry_chunks)
        }
        self.records: dict[str, dict[int, tuple[str, str, int]]] = defaultdict(dict)

    def prepare_cache(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> tuple[CachePair, dict]:
        conv, delta = cache
        history_conv, history_delta = history_cache
        if (history_conv is None) != (history_delta is None):
            raise ValueError("GDN history must be provided as a pair")
        expected = self.chunk_infos[chunk_idx]["state_count"]
        if conv.shape[0] not in (1, expected) or delta.shape[0] not in (1, expected):
            raise ValueError("GDN empty cache must contain one row or all local rows")
        if (
            history_conv is None
            or history_conv.numel() == 0
            or history_delta.numel() == 0
        ):
            if conv.shape[0] == expected and delta.shape[0] == expected:
                prepared = conv, delta
            else:
                prepared = (
                    conv.expand(expected, *conv.shape[1:]).contiguous(),
                    delta.expand(expected, *delta.shape[1:]).contiguous(),
                )
            return prepared, {}
        if history_conv.ndim == conv.ndim - 1:
            history_conv = history_conv.unsqueeze(0)
        if history_delta.ndim == delta.ndim - 1:
            history_delta = history_delta.unsqueeze(0)
        if self.layout == "bsnd":
            expected_conv = (expected, *conv.shape[1:])
            expected_delta = (expected, *delta.shape[1:])
            if history_conv.shape != expected_conv or history_delta.shape != expected_delta:
                raise ValueError("GDN BSND history shape changed between chunks")
            return (history_conv, history_delta), {}
        if expected == 1:
            return (history_conv, history_delta), {}
        return (
            (
                torch.cat(
                    (history_conv, conv[:1].expand(expected - 1, *conv.shape[1:])),
                    dim=0,
                ),
                torch.cat(
                    (history_delta, delta[:1].expand(expected - 1, *delta.shape[1:])),
                    dim=0,
                ),
            ),
            {},
        )

    def append(
        self,
        cache: CachePair,
        history_cache: tuple[torch.Tensor | None, torch.Tensor | None],
        chunk_idx: int,
    ) -> CachePair:
        del history_cache
        conv, delta = cache
        chunk_info = self.chunk_infos[chunk_idx]
        expected = chunk_info["state_count"]
        if conv.shape[0] != expected or delta.shape[0] != expected:
            raise ValueError(
                f"GDN final-state count must be {expected}, got {conv.shape[0]} and {delta.shape[0]}"
            )
        if not chunk_info["carries_last_doc"]:
            return (
                self._carry_tensor(conv[:0], None, always_clone=True),
                self._carry_tensor(delta[:0], None, always_clone=True),
            )
        if self.layout == "bsnd":
            next_conv, next_delta = conv, delta
            chain_key = "batch"
        else:
            next_conv, next_delta = conv[-1:], delta[-1:]
            chain_key = f"doc{chunk_info['doc_ids'][-1]}"
        common = {
            "cache_key": self.key,
            "chain_key": chain_key,
            "chunk_idx": chunk_idx,
        }
        return (
            self._carry_tensor(
                next_conv,
                CacheSpec(component="conv", **common),
                always_clone=True,
            ),
            self._carry_tensor(
                next_delta,
                CacheSpec(component="delta", **common),
                always_clone=True,
            ),
        )

    def stage(self, handle: CacheHandle, transfer_stream) -> None:
        tensor, spec = handle.tensor, handle.spec
        compact_idx = self.chunk_slots.get(spec.chunk_idx)
        if compact_idx is None:
            raise RuntimeError("GDN cache chunk is absent from the chunk plan")
        device_cache = self.device_tensors.get(spec.component)
        if device_cache is None:
            device_cache = torch.empty(
                (len(self.chunk_slots), *tensor.shape),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            self.device_tensors[spec.component] = device_cache
        elif (
            device_cache.shape[1:] != tensor.shape
            or device_cache.dtype != tensor.dtype
        ):
            raise RuntimeError(
                f"GDN {spec.component} shape or dtype changed within one layer"
            )
        previous = self.records[spec.component].get(spec.chunk_idx)
        if previous is not None and previous != handle.record_key:
            raise RuntimeError(
                f"multiple GDN {spec.component} states occupy chunk {spec.chunk_idx}"
            )
        self.records[spec.component][spec.chunk_idx] = handle.record_key

        event = self.copy_events.get(handle.record_key)
        if event is None:
            event = _launch_d2d(tensor, device_cache[compact_idx], transfer_stream)
            self.copy_events[handle.record_key] = event
        handle.copy_event = event

    def view_for(self, handle: CacheHandle) -> torch.Tensor:
        compact_idx = self.chunk_slots[handle.spec.chunk_idx]
        return self.device_tensors[handle.spec.component][compact_idx]

    def _validate_before_swap_out(self) -> None:
        expected_chunks = set(self.chunk_slots)
        for component in self.device_tensors:
            if set(self.records[component]) != expected_chunks:
                raise RuntimeError(
                    f"GDN {component} cache records do not match the chunk plan"
                )


class CacheOffloadManager(metaclass=Singleton):
    """Schedule layer-cache lifetimes while SwapCore owns physical transfers."""

    def __init__(self):
        self.caches: dict[str, LayerCache] = {}
        self.last_cache_by_forward: dict[int, str] = {}
        self.stats: list[dict] = []
        self._swap_stream = None
        self.tensor_swap_cache = None

    @property
    def swap_stream(self):
        if self._swap_stream is None:
            self._swap_stream = create_stream()
        return self._swap_stream

    def register(self, cache: LayerCache) -> None:
        existing = self.caches.get(cache.key)
        if existing is not None and existing is not cache:
            raise RuntimeError(f"duplicate layer cache key: {cache.key}")
        self.caches[cache.key] = cache

    def get_cache(self, key: str) -> LayerCache | None:
        return self.caches.get(key)

    def finalize(self, key: str) -> tuple[str, str | None] | None:
        cache = self.caches.get(key)
        if cache is None:
            return None
        if cache.state != "staging":
            raise RuntimeError(f"layer cache already finalized: {key}")
        if not cache.handles:
            self.caches.pop(key)
            return None
        if any(not handle.source_released for handle in cache.handles):
            raise RuntimeError("layer cache finalized before all consumers completed")

        cache.swap_out(self.tensor_swap_cache)

        previous_key = self.last_cache_by_forward.get(cache.forward_id)
        if previous_key is not None:
            previous = self.caches.get(previous_key)
            if previous is None or previous.layer_idx >= cache.layer_idx:
                previous_key = None
        self.last_cache_by_forward[cache.forward_id] = key
        self._record_stats(cache)
        return key, previous_key

    def _record_stats(self, cache: LayerCache) -> None:
        actual_bytes = sum(
            swap_handle.bytes()
            for swap_handle in cache.swap_handles.values()
        )
        logical_bytes = sum(
            handle.tensor.numel() * handle.tensor.element_size()
            for handle in cache.handles
        )
        self.stats.append(
            {
                "cache_key": cache.key,
                "layer_idx": cache.layer_idx,
                "layout": cache.layout,
                "components": tuple(
                    sorted({handle.spec.component for handle in cache.handles})
                ),
                "actual_d2h_bytes": actual_bytes,
                "logical_d2h_bytes": logical_bytes,
            }
        )
        if len(self.stats) > 64:
            del self.stats[:-64]

    def get(self, key: str) -> LayerCache:
        cache = self.caches.get(key)
        if cache is None:
            raise RuntimeError(f"layer cache does not exist: {key}")
        cache.swap_in(self.tensor_swap_cache)
        return cache

    def release(self, key: str) -> None:
        cache = self.caches.pop(key, None)
        if cache is None:
            return
        cache.release_materialized()
        if self.last_cache_by_forward.get(cache.forward_id) == key:
            self.last_cache_by_forward.pop(cache.forward_id)

    def clear(self) -> None:
        for cache in self.caches.values():
            cache.release_materialized()
        self.caches.clear()
        self.last_cache_by_forward.clear()
        self.stats.clear()


class _CacheBackwardEnter(torch.autograd.Function):
    """Restore this layer cache before its backward starts."""

    @staticmethod
    def forward(ctx, hidden_states, current_key, previous_key):
        ctx.current_key = current_key
        ctx.previous_key = previous_key
        return hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        manager = CacheOffloadManager()
        manager.get(ctx.current_key)
        return grad_hidden_states, None, None


class _CacheBackwardExit(torch.autograd.Function):
    """Release this layer cache after its input gradient is produced."""

    @staticmethod
    def forward(ctx, hidden_states, current_key):
        ctx.current_key = current_key
        return hidden_states.view_as(hidden_states)

    @staticmethod
    def backward(ctx, grad_hidden_states):
        CacheOffloadManager().release(ctx.current_key)
        return grad_hidden_states, None


def mark_cache_backward_exit(hidden_states: torch.Tensor, key: str) -> torch.Tensor:
    return _CacheBackwardExit.apply(hidden_states, key)


def mark_cache_backward_enter(
    hidden_states: torch.Tensor,
    current_key: str,
    previous_key: str | None,
) -> torch.Tensor:
    return _CacheBackwardEnter.apply(hidden_states, current_key, previous_key)
