# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Indexer expansion for SFA."""

import os

import torch
import triton
import triton.language as tl
from triton.runtime import driver


@triton.jit(do_not_specialize=['ROWS', 'SEQ', 'K'])
def _expand_indexer_indices_kernel(
    selected_ptr, valid_ptr, output_ptr,
    ROWS, SEQ, K,
    WIDTH: tl.constexpr, POOL: tl.constexpr, PACK_TAIL: tl.constexpr,
    HAS_K: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_WIDTH: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    slots = tl.arange(0, BLOCK_WIDTH)
    in_row = rows < ROWS
    # Address arithmetic must stay 64-bit even when token values fit int32.
    row_offsets = rows.to(tl.int64)

    if HAS_K:
        pools = tl.arange(0, BLOCK_K)
        valid = tl.load(valid_ptr + row_offsets[:, None] * K + pools[None, :],
                        in_row[:, None] & (pools[None, :] < K), other=0)
        selected = tl.load(selected_ptr + row_offsets[:, None] * K + pools[None, :],
                           in_row[:, None] & (pools[None, :] < K), other=0).to(tl.int32)
        gather_pool = tl.minimum(slots // POOL, K - 1)
        gather_pool = tl.broadcast_to(gather_pool[None, :], (BLOCK_ROWS, BLOCK_WIDTH))
        # This Ascend gather accepts fp32/int8 but not int32. Reinterpret bits
        # for movement only; no floating-point arithmetic or rounding occurs.
        expanded_pool = tl.gather(selected.to(tl.float32, bitcast=True), gather_pool, axis=1)
        expanded = expanded_pool.to(tl.int32, bitcast=True) * POOL + slots[None, :] % POOL
        # 0/1 converts exactly. FP32 also avoids the int8 vgather layout bug
        # on Ascend 950PR&950DT products.
        slot_valid = tl.gather(valid.to(tl.float32), gather_pool, axis=1) != 0
        history = tl.where(slot_valid & (slots[None, :] < K * POOL), expanded, -1)
    else:
        history = tl.full((BLOCK_ROWS, BLOCK_WIDTH), -1, tl.int32)

    if PACK_TAIL:
        visible = rows % SEQ + 1
        tail_count = visible % POOL
        if HAS_K:
            valid_count = tl.sum(valid.to(tl.int32), axis=1)
        else:
            valid_count = tl.full((BLOCK_ROWS,), 0, tl.int32)
        history_length = tl.minimum(valid_count * POOL, WIDTH - tail_count)
        tail_offset = slots[None, :] - history_length[:, None]
        is_tail = (tail_offset >= 0) & (tail_offset < tail_count[:, None])
        tail_value = (visible - tail_count)[:, None] + tail_offset
        history = tl.where(slots[None, :] < history_length[:, None], history, -1)
        result = tl.where(is_tail, tail_value, history)
    else:
        result = history

    tl.store(output_ptr + row_offsets[:, None] * WIDTH + slots[None, :],
             result.to(tl.int64), in_row[:, None] & (slots[None, :] < WIDTH))


def _expand_unpadded_indices(
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    always_select_tail: bool,
    pack_indices_for_sparse_kernel: bool,
):
    """Return int64 [B, S, index_topk], or None for the caller's fallback.

    The existing model hook guarantees no KV cache and an all-valid attention
    mask. Pool p must represent p * index_kpool + arange(index_kpool), and every
    selected index must be in range (invalid entries are clamped by the selector).
    Validity holes are preserved exactly; this function does not compact pools.
    """
    if not pack_indices_for_sparse_kernel:
        return None
    if selected.device.type != "npu" or selected_valid.device != selected.device:
        return None
    if selected.ndim != 3 or selected_valid.shape != selected.shape:
        return None
    if selected.dtype not in (torch.int32, torch.int64) or selected_valid.dtype != torch.bool:
        return None
    if not selected.is_contiguous() or not selected_valid.is_contiguous():
        return None
    if index_kpool <= 0 or index_topk < 0 or index_topk % index_kpool:
        return None

    batch, seq, k = selected.shape
    if k > min(index_topk // index_kpool, seq // index_kpool):
        return None
    # Token arithmetic and per-row coordinates use int32; addresses use int64.
    if max(seq, batch * seq, index_topk, index_kpool) >= 2**31:
        return None
    if batch * seq > 2**31 - 8:
        return None
    # Bound the on-chip tile; unusually wide configurations retain the old path.
    if index_topk > 8192:
        return None
    output = torch.empty((batch, seq, index_topk), dtype=torch.int64, device=selected.device)
    if output.numel() == 0:
        return output

    block_width = triton.next_power_of_2(index_topk)
    block_rows = max(1, min(8, 4096 // block_width))
    _launch_cached(_expand_indexer_indices_kernel, (triton.cdiv(batch * seq, block_rows), 1, 1),
        selected, selected_valid, output,
        batch * seq, seq, k, index_topk, index_kpool,
        always_select_tail and index_kpool > 1,
        k > 0,
        block_rows, triton.next_power_of_2(max(k, 1)), block_width,
    )
    return output


@triton.jit(do_not_specialize=['ROWS', 'SEQ', 'K', 'POOL_COUNT', 'POOL_STRIDE_B', 'MASK_STRIDE_B', 'PREFIX_STRIDE_B'])
def _expand_padded_indices_kernel(
    selected_ptr, valid_ptr, pool_ptr, mask_ptr, prefix_ptr, first_ptr, output_ptr,
    ROWS, SEQ, K, POOL_COUNT,
    WIDTH: tl.constexpr, POOL: tl.constexpr, PACK_TAIL: tl.constexpr,
    HAS_K: tl.constexpr,
    POOL_STRIDE_B, POOL_STRIDE_N: tl.constexpr, POOL_STRIDE_P: tl.constexpr,
    MASK_STRIDE_B, MASK_STRIDE_S: tl.constexpr,
    PREFIX_STRIDE_B, PREFIX_STRIDE_S: tl.constexpr,
    FIRST_STRIDE: tl.constexpr,
    BLOCK_ROWS: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_WIDTH: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    slots = tl.arange(0, BLOCK_WIDTH)
    in_row = rows < ROWS
    row64 = rows.to(tl.int64)
    batch = rows // SEQ
    query = rows % SEQ
    batch64 = batch.to(tl.int64)
    query64 = query.to(tl.int64)
    query_valid = tl.load(mask_ptr + batch64 * MASK_STRIDE_B + query64 * MASK_STRIDE_S,
                          in_row, other=0) != 0

    if HAS_K:
        pools = tl.arange(0, BLOCK_K)
        valid = tl.load(valid_ptr + row64[:, None] * K + pools[None, :],
                        in_row[:, None] & (pools[None, :] < K), other=0)
        selected = tl.load(selected_ptr + row64[:, None] * K + pools[None, :],
                           in_row[:, None] & (pools[None, :] < K), other=0).to(tl.int32)
        gather_pool = tl.minimum(slots // POOL, K - 1)
        gather_pool = tl.broadcast_to(gather_pool[None, :], (BLOCK_ROWS, BLOCK_WIDTH))
        selected_pool = tl.gather(selected.to(tl.float32, bitcast=True), gather_pool, axis=1)
        selected_pool = selected_pool.to(tl.int32, bitcast=True)
        slot_valid = tl.gather(valid.to(tl.float32), gather_pool, axis=1) != 0
        slot_in_range = slots[None, :] < K * POOL
        # Read the model's filtered pool table. Its indices can contain holes,
        # -1 values, and a different first valid token for each batch row.
        table_offset = (batch64[:, None] * POOL_STRIDE_B
                        + selected_pool.to(tl.int64) * POOL_STRIDE_N
                        + (slots[None, :] % POOL).to(tl.int64) * POOL_STRIDE_P)
        if (WIDTH == 2048 and POOL == 4) and (POOL_STRIDE_N == 4 and POOL_STRIDE_P == 1):
            # Uniform runtime branch keeps N/K out of specialization keys.
            if (K == POOL_COUNT) & (POOL_COUNT <= BLOCK_K):
                # Keep each source row a 1-D contiguous masked load. On this
                # backend the non-affine 2-D batch gather can lower to a full
                # WIDTH GM read followed by a slice, exceeding a small table.
                table = tl.full((BLOCK_ROWS, BLOCK_WIDTH), -1, tl.int32)
                for table_row in tl.static_range(0, BLOCK_ROWS):
                    source_row = tl.program_id(0) * BLOCK_ROWS + table_row
                    source_batch = (source_row // SEQ).to(tl.int64)
                    row_table = tl.load(
                        pool_ptr + source_batch * POOL_STRIDE_B + slots.to(tl.int64),
                        (source_row < ROWS) & (slots < POOL_COUNT * POOL), other=-1,
                    ).to(tl.int32)
                    table = tl.where(tl.arange(0, BLOCK_ROWS)[:, None] == table_row,
                                     row_table[None, :], table)
                table_index = selected_pool * POOL + slots[None, :] % POOL
                # fp32 gather moves the int32 bits, including -1, without math.
                value = tl.gather(table.to(tl.float32, bitcast=True), table_index, axis=1)
                value = value.to(tl.int32, bitcast=True)
            else:
                value = tl.load(pool_ptr + table_offset,
                                in_row[:, None] & slot_in_range, other=-1).to(tl.int32)
        else:
            value = tl.load(pool_ptr + table_offset,
                            in_row[:, None] & slot_in_range, other=-1).to(tl.int32)
        history = tl.where(slot_valid & slot_in_range, value, -1)
    else:
        history = tl.full((BLOCK_ROWS, BLOCK_WIDTH), -1, tl.int32)

    if PACK_TAIL:
        visible = tl.load(prefix_ptr + batch64 * PREFIX_STRIDE_B + query64 * PREFIX_STRIDE_S,
                          in_row, other=0).to(tl.int32)
        first = tl.load(first_ptr + batch64 * FIRST_STRIDE, in_row, other=0).to(tl.int32)
        tail_count = visible % POOL
        if HAS_K:
            valid_count = tl.sum(valid.to(tl.int32), axis=1)
        else:
            valid_count = tl.full((BLOCK_ROWS,), 0, tl.int32)
        history_length = tl.minimum(valid_count * POOL, WIDTH - tail_count)
        tail_offset = slots[None, :] - history_length[:, None]
        tail_value = (first + visible - tail_count)[:, None] + tail_offset
        is_tail = ((tail_offset >= 0) & (tail_offset < tail_count[:, None])
                   & (tail_value < SEQ))
        history = tl.where(slots[None, :] < history_length[:, None], history, -1)
        result = tl.where(is_tail, tail_value, history)
    else:
        result = history
    # The reference returns zero, not -1, for every padded query row.
    result = tl.where(query_valid[:, None], result, 0)
    tl.store(output_ptr + row64[:, None] * WIDTH + slots[None, :],
             result.to(tl.int64), in_row[:, None] & (slots[None, :] < WIDTH))


def expand_indexer_indices(
    selected: torch.Tensor,
    selected_valid: torch.Tensor,
    index_topk: int,
    index_kpool: int,
    always_select_tail: bool,
    pack_indices_for_sparse_kernel: bool,
    pool_indices: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    visible_counts: torch.Tensor | None = None,
    first_key: torch.Tensor | None = None,
):
    """Expand pool selections, preserving the original uncached mask semantics.

    Four optional tensors select the padded path. ``visible_counts`` is the
    inclusive mask prefix sum, and ``first_key`` is the first true mask position
    per batch row. For an all-padding row first_key may be zero: its output is
    always overwritten with zeros. Unsupported metadata returns None.
    """
    if pool_indices is None and attention_mask is None and visible_counts is None and first_key is None:
        return _expand_unpadded_indices(selected, selected_valid, index_topk, index_kpool,
                                       always_select_tail, pack_indices_for_sparse_kernel)
    if any(x is None for x in (pool_indices, attention_mask, visible_counts, first_key)):
        return None
    if not pack_indices_for_sparse_kernel or selected.device.type != "npu":
        return None
    if any(x.device != selected.device for x in (selected_valid, pool_indices, attention_mask,
                                                  visible_counts, first_key)):
        return None
    if selected.ndim != 3 or selected_valid.shape != selected.shape:
        return None
    if selected.dtype not in (torch.int32, torch.int64) or selected_valid.dtype != torch.bool:
        return None
    if not selected.is_contiguous() or not selected_valid.is_contiguous():
        return None
    if index_kpool <= 0 or index_topk < 0 or index_topk % index_kpool or index_topk > 8192:
        return None
    batch, seq, k = selected.shape
    if seq == 0 or pool_indices.ndim != 3 or pool_indices.shape[0] != batch:
        return None
    if pool_indices.shape[2] != index_kpool or k > min(index_topk // index_kpool, pool_indices.shape[1]):
        return None
    if pool_indices.dtype not in (torch.int32, torch.int64):
        return None
    if attention_mask.shape != (batch, seq) or attention_mask.dtype != torch.bool:
        return None
    if visible_counts.shape != (batch, seq) or visible_counts.dtype not in (torch.int32, torch.int64):
        return None
    if first_key.shape != (batch,) or first_key.dtype not in (torch.int32, torch.int64):
        return None
    if max(seq, batch * seq, index_topk, index_kpool) >= 2**31 or batch * seq > 2**31 - 8:
        return None
    output = torch.empty((batch, seq, index_topk), dtype=torch.int64, device=selected.device)
    if output.numel() == 0:
        return output
    block_width = triton.next_power_of_2(index_topk)
    block_rows = max(1, min(8, 4096 // block_width))
    # One K specialization for the measured SFA configuration. The guard
    # above guarantees K <= 2048 // 4; masked lanes still contribute zero.
    # Other index widths and pool sizes retain the original tiling policy.
    if k > 0 and index_topk == 2048 and index_kpool == 4:
        block_k = 512
    else:
        block_k = triton.next_power_of_2(max(k, 1))
    _launch_cached(_expand_padded_indices_kernel, (triton.cdiv(batch * seq, block_rows), 1, 1),
        selected, selected_valid, pool_indices, attention_mask, visible_counts, first_key, output,
        batch * seq, seq, k, pool_indices.shape[1], index_topk, index_kpool,
        always_select_tail and index_kpool > 1,
        k > 0,
        *pool_indices.stride(), *attention_mask.stride(), *visible_counts.stride(), first_key.stride(0),
        block_rows, block_k, block_width,
    )
    return output


_GLOBAL_MISSING = object()


def _launch_cached(kernel, grid, *args):
    """Reuse Triton's exact binder/cache key and public compiled launch API.

    Cache hits avoid rebuilding the Ascend backend and querying its target.
    Cache misses and instrumented calls retain the standard JIT path. No tensor,
    stream or alternative specialization cache is retained by this wrapper.
    """
    binder = getattr(kernel, "binder", None)
    cache = getattr(kernel, "cache", None)
    if (binder is None or cache is None or not hasattr(kernel, "used_global_vals")
            or getattr(kernel, "pre_run_hooks", ())):
        return kernel[grid](*args)
    device = driver.active.get_current_device()
    stream = driver.active.get_current_stream(device)
    debug = os.environ.get("TRITON_DEBUG", "0") == "1"
    try:
        bound_args, signature, constants, runtime_args, options = binder(*args, debug=debug)
        key = "".join(signature) + str((constants, options))
    except (TypeError, ValueError, AttributeError):
        # Unsupported Triton binder revisions retain their original behavior.
        return kernel[grid](*args)
    compiled = cache[device].get(key)
    if compiled is None:
        return kernel[grid](*args)
    for (name, _), (value, namespace) in kernel.used_global_vals.items():
        if namespace.get(name, _GLOBAL_MISSING) != value:
            # Preserve the JIT's normal global-mutation error and hook handling.
            return kernel[grid](*args)
    if hasattr(compiled, "result"):
        compiled = compiled.result()
    compiled[grid](*runtime_args, stream=stream)
    return compiled
