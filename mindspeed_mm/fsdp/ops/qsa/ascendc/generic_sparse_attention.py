"""Qwen QSA AscendC backend: CANN TND forward and native QSAG backward.

The adapter supports causal prefill with equal physical sequence lengths.
Right padding is allowed when padded query outputs do not contribute to loss.
Forward indices include the incomplete tail; QSAG adds that tail implicitly.
"""

import importlib
import logging
import math
from functools import lru_cache

import torch
from torch.autograd.function import once_differentiable


logger = logging.getLogger(__name__)


def validate_prefill_mask(attention_mask, position_ids, batch, sequence):
    """Check sequence boundaries once per model call, before building 4D masks."""
    valid = None
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != (batch, sequence):
            raise NotImplementedError("AscendC QSA accepts only a 2D padding mask or None; custom/packed masks are unsupported")
        valid = attention_mask != 0
        invalid = ((attention_mask != 0) & ( attention_mask != 1)).any()
        invalid = invalid | (valid[:, 1:] & ~valid[:, :-1]).any()
        if invalid.item():
            raise NotImplementedError("AscendC QSA supports binary right-padding masks only; left padding/packing is unsupported")
    if position_ids is not None:
        if position_ids.shape != (batch, sequence):
            raise ValueError("QSA text position_ids must have shape [B,S]")
        reset = position_ids[:, 1:] <= position_ids[:, :-1]
        if valid is not None:
            reset = reset & valid[:, 1:] & valid[:, :-1]
        if reset.any().item():
            raise NotImplementedError("AscendC QSA does not support packed sequences with reset text positions")


def _qsag_pattern(ids, counts, batch, sequence, block_size):
    """Convert forward [Hkv,B*S,K] to complete-block-only [B,Hkv,S,K]."""
    heads, _, width = ids.shape
    ids = ids.reshape(heads, batch, sequence, width).permute(1, 0, 2, 3)
    counts = counts.reshape(heads, batch, sequence).permute(1, 0, 2)
    slots = torch.arange(width, device=ids.device)
    complete = (torch.arange(sequence, device=ids.device) + 1) // block_size
    valid = (slots < counts[..., None]) & (ids >= 0) & (ids < complete[None, None, :, None])
    sentinel = (sequence + block_size - 1) // block_size
    indices = torch.where(valid, ids, sentinel).float().sort(dim=-1).values.to(torch.int32)
    indices = torch.where(indices < sentinel, indices, 0).to(torch.int32).contiguous()
    return indices, valid.sum(-1).to(torch.int32).contiguous()


@lru_cache(maxsize=1)
def _load_qsag_backward():
    import cann_ops_transformer.ops as ops
    from cann_ops_transformer.ops import generic_block_sparse_attention_grad

    # Some CANN wheels have the QSAG-capable native ABI but retain a Python
    # validator restricted to layout=1/residual=0. Use that wheel's existing
    # binding without changing global registrations or files in site-packages.
    legacy_module = (
        "cann_ops_transformer.ops.attention.generic_block_sparse_attention_grad."
        "generic_block_sparse_attention_grad"
    )
    if isinstance(generic_block_sparse_attention_grad, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)):
        # @impl may export a dispatcher handle whose __module__ is torch._ops.
        if "generic_block_sparse_attention_grad" in getattr(ops, "_ep_overridden_ops", ()):
            return generic_block_sparse_attention_grad, False
        module = importlib.import_module(legacy_module)
    else:
        module = importlib.import_module(generic_block_sparse_attention_grad.__module__)
    if module.__name__ == legacy_module and not hasattr(module, "_check_qsag"):
        native = module.generic_block_sparse_attention_grad_op_builder.load()
        logger.warning(
            "CANN grad Python wrapper predates QSAG; using its native "
            "GenericBlockSparseAttentionGrad binding (layout=0, residual=1). "
            "The loaded custom operator must support QSAG."
        )
        return native.generic_block_sparse_attention_grad, True
    return generic_block_sparse_attention_grad, False


def _qsag_backward(q, k, v, dout, out, lse, ids, counts, cu, block_size, scale):
    op, native_binding = _load_qsag_backward()
    if native_binding:
        # Positional ABI from CANN's csrc/attention/generic_block_sparse_attention_grad.cpp.
        # metadata/mask=None, INT64 cumulative lengths, no seqused, TND Q/KV,
        # layout=0, causal, precision=0, no window, implicit incomplete tail.
        return op(
            q, k, v, dout, out, lse, ids, counts,
            None, None, cu, cu, None, None, [1, block_size],
            "TND", "TND", 0, scale, 1, 0, -1, -1, 1, False,
        )
    return op(
        q, k, v, dout, out, lse, ids, counts, [1, block_size],
        metadata=None, attn_mask=None,
        cu_seqlens_q=cu, cu_seqlens_kv=cu, seqused_q=None, seqused_kv=None,
        layout_q="TND", layout_kv="TND", layout_sparse_pattern=0,
        softmax_scale=scale, mask_mode=1, softmax_precision=0,
        win_left=-1, win_right=-1, residual_block_mode=1,
        is_consistent_topk=False,
    )


class _GenericSparseAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, counts, cu, metadata, block_size, scale, needs_backward):
        from cann_ops_transformer.ops import generic_block_sparse_attention

        out, lse = generic_block_sparse_attention(
            q, k, v, ids, counts, [1, block_size], metadata=metadata,
            attn_mask=None, block_table=None, cu_seqlens_q=cu, cu_seqlens_kv=cu,
            layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
            residual_block_mode=0, is_consistent_topk=False,
            softmax_scale=scale, mask_mode=1, quant_mode=0, softmax_precision=1,
            return_softmax_lse=needs_backward,
        )
        if needs_backward:
            ctx.save_for_backward(q, k, v, out, lse, ids, counts, cu)
            ctx.block_size = block_size
            ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        q, k, v, out, lse, ids, counts, cu = ctx.saved_tensors
        batch = cu.numel() - 1
        sequence = q.shape[0] // batch
        grad_ids, grad_counts = _qsag_pattern(ids, counts, batch, sequence, ctx.block_size)
        dq, dk, dv = _qsag_backward(
            q, k, v, dout.contiguous(), out, lse,
            grad_ids, grad_counts, cu, ctx.block_size, ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def complete_block_ids(compact, sequence, token_budget, compress_ratio):
    """Restore dense-prefix rows omitted by the Lightning indexer."""
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError("QSA token_budget must be positive and divisible by compress_ratio")
    batch, sparse_rows, width = compact.shape
    prefix = min(sequence, token_budget + compress_ratio - 1)
    if sparse_rows != sequence - prefix or width != token_budget // compress_ratio + 1:
        raise ValueError("Compact QSA routing shape does not match sequence/budget/block size")
    rows = torch.arange(prefix, device=compact.device)
    ids = torch.arange(width, dtype=torch.int32, device=compact.device)
    dense = torch.where(ids[None, :] <= rows[:, None] // compress_ratio, ids[None, :], -1)
    return torch.cat((dense[None].expand(batch, -1, -1), compact), dim=1)


def generic_sparse_attention(query, key, value, block_ids, *, block_size, scale, dropout=0.0):
    """BNSD -> contiguous TND -> BSH; single-backward causal prefill only."""
    if dropout != 0.0:
        raise NotImplementedError("CANN QSA does not support attention dropout")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("QSA requires Q/K/V in BNSD layout and matching K/V shapes")
    b, hq, s, d = query.shape
    hkv = key.shape[1]
    if min(b, hq, hkv, s, d) <= 0 or hq % hkv or hq // hkv > 256:
        raise ValueError("QSA requires positive dimensions, Hq divisible by Hkv and GQA<=256")
    if key.shape != (b, hkv, s, d) or not 1 <= d <= 256:
        raise ValueError("QSA requires equal Q/KV sequence lengths and head_dim<=256")
    max_block_size = 128 if hq // hkv <= 128 else 64
    if not isinstance(block_size, int) or not 1 <= block_size <= max_block_size or not math.isfinite(scale):
        raise ValueError("Unsupported QSA block size or softmax scale")
    if query.device.type != "npu" or query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CANN QSA requires FP16/BF16 NPU tensors")
    if any(x.device != query.device or x.dtype != query.dtype for x in (key, value)):
        raise ValueError("QSA Q/K/V must have matching dtype and device")
    if (block_ids.ndim != 3 or block_ids.shape[:2] != (b, s) or block_ids.shape[2] < 1
            or block_ids.dtype != torch.int32 or block_ids.device != query.device):
        raise ValueError("QSA block_ids must be INT32 [B,S,K] on the Q/K/V device")

    from cann_ops_transformer.ops import generic_block_sparse_attention_metadata

    needs_backward = torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value))
    limit = (s + block_size - 1) // block_size
    valid = (block_ids >= 0) & (block_ids < limit)
    ids = torch.where(valid, block_ids, limit).float().sort(dim=-1).values.to(torch.int32)
    count = valid.sum(-1).to(torch.int32)
    ids = torch.where(ids < limit, ids, 0).to(torch.int32)
    ids = ids.reshape(1, b * s, -1).expand(hkv, -1, -1).contiguous()
    count = count.reshape(1, b * s).expand(hkv, -1).contiguous()
    cu = torch.arange(b + 1, dtype=torch.int64, device=query.device) * s
    metadata = generic_block_sparse_attention_metadata(
        ids, count, hq, hkv, d, [1, block_size],
        cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=s, max_seqlen_kv=s,
        layout_q="TND", layout_kv="TND", layout_sparse_pattern=4,
        residual_block_mode=0, is_consistent_topk=False,
        mask_mode=1, quant_mode=0, softmax_precision=1,
    )
    q, k, v = [x.transpose(1, 2).reshape(b * s, x.shape[1], d).contiguous()
               for x in (query, key, value)]
    out = _GenericSparseAttention.apply(
        q, k, v, ids, count, cu, metadata, block_size, scale, needs_backward,
    )
    return out.reshape(b, s, hq * d)
