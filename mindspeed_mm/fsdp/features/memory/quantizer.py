"""Low-precision payload quantizers for the swap cache (quantize/dequantize).

A quantizer is deliberately outside the swap core: the cache, its accounting
(_bytes/hbm_bytes/ddr_bytes), TensorBuffer, the prefetch budget and the
REPLICATED state machine only ever see the payload a quantizer returns.
Quantizing is put-side preprocessing, dequantizing is pop-side postprocessing:

- quantize() runs inside SwapCache.put on the calling (put) stream under
  torch.no_grad(). A quantized payload is a freshly allocated, whole,
  contiguous tensor; an unsupported input passes through unchanged with
  meta=None (the quantizer warns once per reject reason), so the cache never
  learns
  whether a payload was quantized. The handle is constructed only after
  quantizing, so the put event that anchors the D2H covers the quantized
  payload.
- dequantize() runs inside SwapCache.pop on the calling (consume) stream under
  torch.no_grad(), and must restore dtype and shape exactly; meta=None (a
  passed-through payload) returns the payload untouched. pop() of a quantized
  handle therefore returns a NEW tensor rather than the object that went into
  put (the unquantized path keeps returning the same object).

Quantizer contract:
- no RNG consumption: op replay stores an RNG snapshot right after the cached
  op ran and, on replay, rewinds the stream to it. Quantizing/dequantizing must
  not draw from any RNG (CPU or device), otherwise the recompute RNG stream
  desynchronizes from forward.
- no in-place mutation of the input tensor of quantize(): the swap cache
  forbids in-place writes to a put tensor while its handle is alive (get()-peek
  plus a write would stale the DDR slot image), with no runtime fallback.
- pass-through: quantize() never refuses an input — an unsupported tensor
  returns unchanged with meta=None, and dequantize() with meta=None returns
  the payload untouched.

Registering a new quantizer: add one instance to QUANTIZERS under its name;
tenants then hand the instance to SwapCache.put(..., quantizer=...) (config
files reference it by name, resolved at scope construction). Nothing else in
the swap path needs to change. (Extension point: an e4m3/fp8 quantizer would
need a hand-written quantize kernel — this stack has no native fp8 cast —
intentionally not implemented.)

This module is the tenant-facing layer: the contract, the meta record, the
registry. The hardware-facing dequant implementation (native op probe, Triton
kernels, per-arch geometry) lives in mindspeed_mm/fsdp/ops/int8_dequant.py.
"""
import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import torch

from mindspeed_mm.fsdp.log import print_rank

from ...ops.int8_dequant import dequant_int8_per_row


logger = logging.getLogger(__name__)


class PayloadQuantizer(Protocol):
    """Payload transformation contract (see the module docstring).

    quantize/dequantize are called only under torch.no_grad() on the caller's
    stream (put stream / consume stream). quantize() never refuses an input:
    an unsupported tensor passes through unchanged with meta=None.
    """

    name: str

    def quantize(self, t: torch.Tensor) -> Tuple[torch.Tensor, Optional[object]]: ...

    def dequantize(self, payload: torch.Tensor, meta) -> torch.Tensor: ...


@dataclass(frozen=True)
class Int8PerTokenMeta:
    """Opaque int8-per-token payload descriptor.

    The swap core never interprets it (SwapHandle just carries the reference
    and drops it in clear(), which releases the scale tensor); pop() hands it
    back to the quantizer that produced it.

    orig_dtype/orig_shape: the tensor put into the cache.
    reduced_shape: the >=2D shape the fused op worked on — for a 1D input the
      [1, numel] shim, i.e. the shape the payload must be restored to before
      dequantization.
    scale: fp32 per-token scales of shape reduced_shape[:-1].
    """

    orig_dtype: torch.dtype
    orig_shape: torch.Size
    reduced_shape: torch.Size
    scale: torch.Tensor


class Int8PerTokenQuantizer:
    """int8 per-token symmetric quantization via the fused NPU op.

    quantize: torch.ops.npu.npu_dynamic_quant(x2d) -> (int8 payload, fp32
    per-token scale), one fused pass (measured: 1.4 ms for a 2D 1 GiB bf16
    payload). The op requires >=2D input; a 1D tensor is shimmed to [1, numel],
    which is ~40x slower and degenerates to a single scale for the whole
    tensor — whether to quantize 1D payloads is the scope's call.
    dequantize: handed to fsdp.ops.int8_dequant.dequant_int8_per_row, which
    dispatches over three bitwise-identical paths (fused native op, Triton
    kernel, torch composition — see that module).

    Net effect on the cache: the payload is half the original bytes, so both
    capacity accounting and the D2H/H2D traffic halve, at the cost of one
    quantize pass at put and one dequantize pass at pop.
    """

    name = "int8_pertoken"

    def __init__(self):
        # reject reasons already warned about: one warning per reason instead
        # of one per skipped tensor.
        self._skip_warned = set()

    @staticmethod
    def _reject_reason(t: torch.Tensor) -> Optional[str]:
        # NPU-only: quantize() rides torch.ops.npu.npu_dynamic_quant, which has
        # no CUDA counterpart in this stack.
        if t.dtype not in (torch.bfloat16, torch.float16):
            return f"dtype={t.dtype}"
        if not t.is_contiguous():
            return "non-contiguous"
        if t.numel() == 0:
            return "empty tensor"
        if t.device.type != 'npu':
            return f"device={t.device.type}"
        return None

    def quantize(self, t: torch.Tensor) -> Tuple[torch.Tensor, Optional[object]]:
        reason = self._reject_reason(t)
        if reason is not None:
            # Pass the tensor through unchanged, meta=None. Warn once per
            # reject reason — a per-put warning would flood training logs —
            # and only on the logging ranks (rank 0).
            if reason not in self._skip_warned:
                self._skip_warned.add(reason)
                print_rank(
                    logger.warning,
                    f"payload quantizer {self.name!r} does not support this "
                    f"tensor ({reason}, shape={tuple(t.shape)}); passing it "
                    f"through unchanged. Further skips of this reason are "
                    f"not warned again.",
                    ranks=0)
            return t, None
        # The fused op takes a 2D input; 1D falls back to a single-token shim
        # ([1, numel]) and the shim shape is recorded for dequantization.
        x2d = t.reshape(-1, t.shape[-1]) if t.dim() >= 2 else t.reshape(1, -1)
        reduced_shape = x2d.shape
        q, scale = torch.ops.npu.npu_dynamic_quant(x2d)
        # payload: a fresh contiguous block (the op allocates q; reshape is a
        # view of it), int8 with the original shape.
        payload = q.reshape(t.shape)
        meta = Int8PerTokenMeta(
            orig_dtype=t.dtype,
            orig_shape=t.shape,
            reduced_shape=reduced_shape,
            scale=scale,
        )
        return payload, meta

    def dequantize(self, payload: torch.Tensor,
                   meta: Optional[Int8PerTokenMeta]) -> torch.Tensor:
        if meta is None:  # pass-through payload: never quantized
            return payload
        rows, cols = int(meta.reduced_shape[0]), int(meta.reduced_shape[-1])
        out = dequant_int8_per_row(payload.reshape(rows, cols), meta.scale,
                                   meta.orig_dtype)
        return out.reshape(meta.orig_shape)


# Name -> quantizer instance. Config layers reference a quantizer by name and
# resolve it here (OpReplayScope construction); the swap core itself never
# imports this module.
QUANTIZERS = {
    Int8PerTokenQuantizer.name: Int8PerTokenQuantizer(),
}
