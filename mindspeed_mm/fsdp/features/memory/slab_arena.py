import torch
from dataclasses import dataclass
from typing import Optional, List


# torch_npu DMA copies require the start addresses of both ends to be aligned
# (measured: >=128B recovers, 512B safest; odd storage_offset halves bandwidth).
_DMA_ALIGN = 512


class SlabBuffer:
    """Fixed-size byte-contiguous buffer; allocates linearly forward and never reclaims interior holes"""

    def __init__(self, capacity_bytes: int, device: str = 'cpu', pin_memory: bool = True,
                 oversized: bool = False):
        # capacity is aligned to 8 bytes so views of any dtype (fp64 / int64, etc.) fit
        self.capacity = (capacity_bytes + 7) // 8 * 8
        self.buf = torch.empty(self.capacity, dtype=torch.uint8, device=device, pin_memory=pin_memory)
        self.used = 0          # bytes allocated so far
        self.ref_count = 0     # reference count from the business layer
        # oversized: a dedicated slab for a single tensor whose payload exceeds
        # slab_size. One tensor per slab, never in the general packing pool
        # (a squatting small tensor would keep ref_count from reaching zero,
        # breaking pool_policy="standard"'s release-on-empty).
        self.oversized = oversized

    def alloc(self, num_bytes: int, alignment: int) -> Optional[int]:
        """Try to allocate num_bytes bytes and return the byte offset; return None if space is insufficient"""
        aligned = (self.used + alignment - 1) // alignment * alignment
        if aligned + num_bytes > self.capacity:
            return None
        offset = aligned
        self.used = aligned + num_bytes
        self.ref_count += 1
        return offset

    def dec_ref(self) -> bool:
        """Drop one reference; return True when the slab just became empty."""
        if self.ref_count <= 0:
            raise RuntimeError("SlabBuffer.dec_ref: double free detected")
        self.ref_count -= 1
        return self.ref_count == 0

    def reset(self):
        if self.ref_count != 0:
            raise RuntimeError(
                f"SlabBuffer.reset: ref_count={self.ref_count} != 0, "
                "slab still has live allocations")
        self.used = 0
        self.ref_count = 0


@dataclass(frozen=True)
class TensorBuffer:
    """
    Memento of one slab allocation plus the layout of the registered tensor.
    Immutable after construction; the single source of truth for "what these
    bytes mean".

    - slab_slice: 1D view (in the original dtype) onto the CPU slab; D2H
                 destination / H2D source.
    - restored: CPU-side restore view with the original shape/stride (also
                the layout carrier for device-side rebuilds).
    - slab: the owning SlabBuffer, used for freeing.
    - byte_start/byte_end: byte offset range.
    - origin_device: the device the tensor was registered from (H2D target).

    Transfer contract (all methods are ambient-stream and ordering-agnostic):
    these methods only allocate, issue copy_ on the CURRENT stream, and build
    views; they know nothing about streams, events, or state machines. The
    caller owns all timing: it must order the current stream after the
    tensor's last write (e.g. wait_stream) BEFORE calling, keep the source
    alive until the copy completes, and record/wait events afterwards as its
    state machine requires. Allocator note: device blocks take their pool
    from the ambient stream, so call prepare_landing()/load() on the stream
    whose pool the landing buffer should belong to.
    """
    slab_slice: torch.Tensor
    restored: torch.Tensor
    slab: SlabBuffer
    byte_start: int
    byte_end: int
    origin_device: torch.device

    def store_from(self, tensor: torch.Tensor, non_blocking: bool = False) -> None:
        """D2H: copy the tensor's copy window (occupied storage range, plus
        the alignment head when needed) into the slab slot as a flat 1D
        transfer.

        Precondition: the tensor is the one this buffer was registered from
        (same layout); the caller keeps it alive until the copy completes.
        """
        self.slab_slice.copy_(SlabArena.storage_range_1d(tensor), non_blocking=non_blocking)

    def prepare_landing(self) -> torch.Tensor:
        """Allocate the device-side H2D landing buffer (flat 1D, contiguous).
        The allocation pool is decided by the ambient stream at the call site."""
        return torch.empty(self.slab_slice.numel(), dtype=self.slab_slice.dtype,
                           device=self.origin_device)

    def load_into(self, landing: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
        """H2D: copy the slab slot into `landing` and return the rebuilt view
        with the original shape/stride on it. Data becomes valid once the
        copies queued on the current stream complete."""
        misaligned_dma_dst = (self.origin_device.type != 'cpu'
                              and landing.data_ptr() % _DMA_ALIGN != 0)
        if (landing.numel() != self.slab_slice.numel()
                or landing.dtype != self.slab_slice.dtype
                or landing.device != self.origin_device
                or misaligned_dma_dst):
            raise RuntimeError(
                "TensorBuffer.load_into: landing does not match the registered "
                "buffer (expected numel/dtype/device/alignment of prepare_landing())")
        landing.copy_(self.slab_slice, non_blocking=non_blocking)
        # the original first element sits at +shift inside the copy window
        rebuild_off = self.restored.storage_offset() - self.slab_slice.storage_offset()
        return torch.as_strided(landing, self.restored.shape, self.restored.stride(),
                                storage_offset=rebuild_off)

    def load(self, non_blocking: bool = False) -> torch.Tensor:
        """One-call form of prepare_landing() + load_into(); the allocation
        pool is the ambient stream at the call site."""
        return self.load_into(self.prepare_landing(), non_blocking=non_blocking)


class SlabArena:
    """
    Slab allocator for CPU tensor offloading.

    - uint8 buffer underneath, byte-granularity allocation, mixed dtypes
    - fixed-size slabs, extended on demand; a slab returns to the free list
      once all its tensors are freed
    - oversized slabs (single-tensor payload > slab_size) are dedicated,
      one tensor per slab

    Layout fidelity is unconditional: register captures the tensor's copy
    window (occupied storage range, holes included, plus an alignment head of
    <_DMA_ALIGN bytes when the first-element address needs it — odd
    storage_offset halves torch_npu DMA bandwidth), transfers move the whole
    window as a flat 1D image, and load restores the original shape/stride
    via as_strided (skipping the head).

    Memory release strategy (pool_policy):
    - "all": keep every empty slab's physical memory for reuse; only an
      explicit release_empty_slabs() returns it to the torch host cache
      (measured: it does not go back to the OS directly).
    - "standard": pool standard slabs; physically release oversized slabs as
      soon as they empty out.
    - "none" (on-demand): physically release every empty slab immediately;
      exact per-tensor allocation at the cost of pinned alloc/free each time.
    """

    _POOL_POLICIES = ("all", "standard", "none")

    def __init__(self, slab_size_bytes: int = 50_000_000, device: str = 'cpu',
                 pin_memory: bool = True, pool_policy: str = "all"):
        if pool_policy not in self._POOL_POLICIES:
            raise ValueError(
                f"SlabArena: invalid pool_policy '{pool_policy}', "
                f"expected one of {self._POOL_POLICIES}")
        self.slab_size = max(1, slab_size_bytes)
        self.device = device
        self.pin_memory = pin_memory
        self.pool_policy = pool_policy
        # All owned slabs are exactly the union of these two lists (disjoint):
        # no separate registry to keep in sync.
        self._active: List[SlabBuffer] = []     # slabs with live allocations
        self._free: List[SlabBuffer] = []       # completely idle slabs (reusable)
        self._oversized_total = 0               # cumulative count of oversized slabs created

    @classmethod
    def on_demand(cls, pin_memory: bool = True) -> "SlabArena":
        """On-demand arena: minimal slab size -> exact per-tensor slab,
        physically released as soon as it empties (pool_policy="none").
        Convenience factory for exact per-tensor allocation (no slab
        pooling), used by tests/probes and by the orchestration layer
        when no pooled arena is wanted."""
        return cls(slab_size_bytes=1, device='cpu', pin_memory=pin_memory,
                   pool_policy="none")

    @staticmethod
    def compute_storage_range(tensor: torch.Tensor):
        """Compute the element-index range [start, end) the tensor actually occupies in its underlying storage"""
        if any(s < 0 for s in tensor.stride()):
            raise RuntimeError(
                "SlabArena: negative stride is not supported "
                "(e.g. from torch.flip); call .contiguous() first")
        if tensor.numel() == 0:
            return tensor.storage_offset(), tensor.storage_offset()
        offset = tensor.storage_offset()
        max_idx = offset
        for dim_size, stride in zip(tensor.shape, tensor.stride()):
            max_idx += (dim_size - 1) * stride
        return offset, max_idx + 1

    @staticmethod
    def _aligned_window(tensor: torch.Tensor):
        """Copy window [win_start, end) plus shift: the single entry for sizing,
        accounting, D2H source views and layout matching.

        When the first-element address is not _DMA_ALIGN-aligned (odd
        storage_offset), the window start is pulled down by shift elements so
        the D2H source address lands on an alignment boundary (odd offsets
        halve DMA bandwidth; the probe-validated fix). The extra prefix
        (<_DMA_ALIGN bytes, holes within the same storage) rides along in the
        slab image and is skipped by the +shift offset of restored/rebuilt
        views. Unfixable cases (residue not divisible by element size, or
        insufficient prefix — external memory only) keep the original window
        and take the slow path."""
        start, end = SlabArena.compute_storage_range(tensor)
        esize = tensor.element_size()
        misalign = tensor.data_ptr() % _DMA_ALIGN
        if tensor.numel() == 0 or misalign == 0 or misalign % esize != 0:
            return start, end, 0
        shift = misalign // esize
        if shift > start:
            return start, end, 0
        return start - shift, end, shift

    @staticmethod
    def storage_range_1d(tensor: torch.Tensor) -> torch.Tensor:
        """1D view over the tensor's copy window (occupied storage range, plus
        the alignment head when the source address needs it) — the flat D2H
        source form. Pure metadata; ownership passes to the caller."""
        start, end, _ = SlabArena._aligned_window(tensor)
        return torch.as_strided(tensor, (end - start,), (1,), storage_offset=start)

    def payload_bytes(self, tensor: torch.Tensor) -> int:
        """Swap payload bytes: the copy window (storage range plus alignment
        head). Single source of truth for accounting and allocation sizing."""
        start, end, _ = self._aligned_window(tensor)
        return (end - start) * tensor.element_size()

    def _activate_and_alloc(self, slab: SlabBuffer, num_bytes: int, alignment: int):
        """Atomic unit: activate an idle slab into the active pool and allocate
        from it. The caller guarantees the slab is empty and has enough
        capacity (fresh slabs are trivially empty)."""
        slab.reset()
        self._active.append(slab)
        offset = slab.alloc(num_bytes, alignment)
        if offset is None:
            raise RuntimeError(
                "SlabArena: allocation failed on a slab with enough capacity")
        return slab, offset

    def _find_or_create_slab(self, num_bytes: int, alignment: int):
        """Find or create a slab for num_bytes; returns (slab, byte_offset).

        Three steps in order: first-fit among active standard slabs; reuse an
        idle slab from the free list (matching channel); create a new slab
        (exact-size dedicated slab for oversized payloads, slab_size for
        standard ones)."""
        oversized = num_bytes > self.slab_size

        # 1) first-fit among active standard slabs (oversized slabs are
        # one-tensor dedicated and never join the general packing pool)
        if not oversized:
            for slab in self._active:
                if slab.oversized:
                    continue
                offset = slab.alloc(num_bytes, alignment)
                if offset is not None:
                    return slab, offset

        # 2) reuse an idle slab from the free list (same channel, enough capacity)
        for i, slab in enumerate(self._free):
            if slab.oversized == oversized and slab.capacity >= num_bytes:
                return self._activate_and_alloc(self._free.pop(i), num_bytes, alignment)

        # 3) create a new slab
        if oversized:
            # dedicated channel: one tensor per slab, never squatted by small
            # tensors (a squatter would keep ref_count from reaching zero,
            # breaking pool_policy="standard"'s release-on-empty)
            self._oversized_total += 1
        slab = SlabBuffer(num_bytes if oversized else self.slab_size,
                          self.device, self.pin_memory, oversized=oversized)
        return self._activate_and_alloc(slab, num_bytes, alignment)

    def register(self, tensor: torch.Tensor) -> TensorBuffer:
        """
        Register a tensor and pre-allocate its slab slot, capturing its layout
        (metadata only: shape/stride/dtype/storage offset are read, no
        device-side view is retained). Returns the TensorBuffer memento; all
        subsequent transfer/free operations go through it.
        """
        win_start, storage_end, shift = self._aligned_window(tensor)
        range_numel = storage_end - win_start

        num_bytes = range_numel * tensor.element_size()
        # torch_npu DMA copies require the host-side address to be 512B-aligned;
        # element-size alignment would let a slice fall into the slow copy path
        # (measured: D2H -35% at a 2000B offset). 512B costs <=511B of slab
        # internal fragmentation per tensor, negligible against MB payloads.
        alignment = max(tensor.element_size(), _DMA_ALIGN)

        slab, byte_start = self._find_or_create_slab(num_bytes, alignment)
        byte_end = byte_start + num_bytes

        # Interpret the CPU slab in the original dtype, for dst / restored
        buf_typed = slab.buf.view(tensor.dtype)
        element_start = byte_start // tensor.element_size()
        slab_slice = buf_typed[element_start:element_start + range_numel]

        # the restored view preserves the original shape / stride mapping;
        # +shift skips the alignment head at the front of the copy window
        restored = torch.as_strided(
            buf_typed,
            tensor.shape,
            tensor.stride(),
            storage_offset=element_start + shift,
        )

        return TensorBuffer(
            slab_slice=slab_slice,
            restored=restored,
            slab=slab,
            byte_start=byte_start,
            byte_end=byte_end,
            origin_device=tensor.device,
        )

    def free(self, buf: TensorBuffer) -> bool:
        """
        Free a TensorBuffer. When the owning slab's ref_count drops to zero:
        "all" sends it to the free list; "standard" frees oversized slabs
        physically and pools standard ones; "none" frees it physically.
        """
        slab = buf.slab
        if not slab.dec_ref():
            return False
        self._active.remove(slab)
        retain = (self.pool_policy == "all"
                  or (self.pool_policy == "standard" and not slab.oversized))
        if retain:
            self._free.append(slab)
        else:
            slab.buf = None  # physically released; the shell is dropped by GC
        return True

    def release_empty_slabs(self):
        """Actually release the memory of idle slabs (drop tensor references)"""
        for slab in self._free:
            slab.buf = None
        self._free.clear()

    def stats(self):
        owned = self._active + self._free
        total_cap = sum(s.capacity for s in owned)
        active_used = sum(s.used for s in self._active)
        oversized = [s for s in owned if s.oversized]
        return {
            'total_capacity': total_cap,
            'active_used': active_used,
            'active_slabs': len(self._active),
            'free_slabs': len(self._free),
            'total_slabs': len(owned),
            # a cumulative count growing per iteration means oversize is
            # recurring; the right fix is a larger slab_size
            'oversized_slabs': len(oversized),
            'oversized_total': self._oversized_total,
        }
