# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton LightningIndexer ID-to-mask postprocessing for Ascend.

Sanitize complete-block IDs, scatter into a compact int8 block-membership
map, then fuse causal/tail handling and final mask materialization. The
existing eager path remains in the LightningIndexer adapter. The caller
still combines this selected mask with its causal/padding mask.
"""

import logging
from functools import lru_cache

import torch
import triton
import triton.language as tl

from mindspeed_mm.fsdp.utils.device import get_device_name, get_vector_num, is_arch35

logger = logging.getLogger(__name__)

# TILE 4096 is the configuration validated on Ascend 910C/950, but its live
# 4096-wide uint32 vectors overflow the smaller unified buffer on Atlas A2
# training products and fail
# HIVM PlanMemory during triton compilation. The width is picked from the SoC
# version instead and halved again if even that fails to compile.
_PACKED16_TILE = None       # resolved lazily on the first launch
_PACKED16_DISABLED = False  # flipped off permanently once every tile failed


@lru_cache(maxsize=None)
def _is_ascend_a2() -> bool:
    """True on Atlas A2 training products, whose AIV unified buffer is the smallest."""
    try:
        return "910B" in get_device_name()
    except Exception:
        # Undetectable SoC: keep the validated wide tile and let the launch
        # ladder shrink it if the compiler rejects it.
        return False


def _device_tile() -> int:
    """4096 on validated SoCs, 2048 on Atlas A2 training products (half the
    unified-buffer footprint)."""
    return 2048 if _is_ascend_a2() else 4096


@triton.jit
def _qsa_id2mask_sanitize_block_ids_kernel(IDs, Safe, S: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
             SB: tl.constexpr, SR: tl.constexpr, SK: tl.constexpr,
             TOTAL: tl.constexpr, TILE: tl.constexpr):
    for start in range(tl.program_id(0) * TILE, TOTAL, tl.num_programs(0) * TILE):
        i = start.to(tl.int64) + tl.arange(0, TILE)
        if (SK == 1 and SR == K) and SB == S * K:
            x = tl.load(IDs + i, i < TOTAL, other=-1)
        else:
            row = i // K
            x = tl.load(IDs + row // S * SB + row % S * SR + i % K * SK, i < TOTAL, other=-1)
        tl.store(Safe + i, tl.where((x >= 0) & (x < N), x, N).to(tl.int32), i < TOTAL)


@triton.jit
def _qsa_id2mask_block_table_to_mask_kernel(Table, Out, Q: tl.constexpr, KV: tl.constexpr, PREFIX: tl.constexpr,
                 S: tl.constexpr, R: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr,
                 ROWS: tl.constexpr, BINS: tl.constexpr, TILE: tl.constexpr,
                 MIN: tl.constexpr, BOOL: tl.constexpr):
    keys = tl.arange(0, TILE)
    blocks = tl.arange(0, BINS)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        q = row % Q
        pos = q + KV - Q
        if q < PREFIX:
            flags = tl.full((BINS,), 1, tl.float32)
        else:
            # Ascend950's vector layout solver rejects this gather on int8.
            # Widen exact 0/1 flags locally while keeping the table compact.
            flags = tl.load(Table + ((row.to(tl.int64) // Q) * S + q - PREFIX) * STRIDE + blocks,
                            blocks < N, other=0).to(tl.float32)
        for start in range(tl.cdiv(KV, TILE)):
            key = start * TILE + keys
            selected = tl.gather(flags, tl.minimum(key // R, BINS - 1), 0) != 0
            tail = ((pos + 1) % R != 0) & (key // R == (pos + 1) // R)
            allow = (key <= pos) & (selected | tail)
            if BOOL:
                value = allow
            else:
                value = tl.where(allow, 0, MIN)
            tl.store(Out + row.to(tl.int64) * KV + key, value, key < KV)


@triton.jit
def _qsa_id2mask_block_table_to_mask_packed16_kernel(Table, Out, Q: tl.constexpr, KV: tl.constexpr, PREFIX: tl.constexpr,
                          S: tl.constexpr, R: tl.constexpr, N: tl.constexpr,
                          STRIDE: tl.constexpr, ROWS: tl.constexpr, BINS: tl.constexpr,
                          TILE: tl.constexpr, MASK16: tl.constexpr):
    # Even R places both packed 16-bit elements in the same selected block.
    # The final causal pair may have only its low half visible.
    pairs = tl.arange(0, TILE)
    blocks = tl.arange(0, BINS)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        q = row % Q
        pos = q + KV - Q
        if q < PREFIX:
            flags = tl.full((BINS,), 1, tl.float32)
        else:
            # Match the FP32 gather workaround in the unpacked kernel.
            flags = tl.load(Table + ((row.to(tl.int64) // Q) * S + q - PREFIX) * STRIDE + blocks,
                            blocks < N, other=0).to(tl.float32)
        for start in range(tl.cdiv(KV // 2, TILE)):
            pair = start * TILE + pairs
            key = pair * 2
            if q < PREFIX:
                active = tl.full((TILE,), True, tl.int1)
            else:
                active = tl.gather(flags, tl.minimum(key // R, BINS - 1), 0) != 0
                if (pos + 1) % R != 0:
                    active |= key // R == (pos + 1) // R
            word = tl.where(active & (key <= pos), tl.full((TILE,), 0, tl.uint32),
                            tl.full((TILE,), MASK16 | (MASK16 << 16), tl.uint32))
            if pos % 2 == 0:
                word |= tl.where(key == pos, tl.full((TILE,), MASK16 << 16, tl.uint32),
                                 tl.full((TILE,), 0, tl.uint32))
            tl.store(Out + row.to(tl.int64) * (KV // 2) + pair, word, pair < KV // 2)


@triton.jit
def _qsa_id2mask_ratio4_packed16_kernel(
    Table, Out, Q: tl.constexpr, KV: tl.constexpr, PREFIX: tl.constexpr,
    S: tl.constexpr, N: tl.constexpr, STRIDE: tl.constexpr, ROWS: tl.constexpr,
    BLOCKS_PER_TILE: tl.constexpr, MASK16: tl.constexpr,
):
    # Every 4-token block expands into two packed 16-bit output pairs.
    # Tile blocks so arbitrary sequence lengths do not exceed on-chip memory.
    blocks = tl.arange(0, BLOCKS_PER_TILE)
    pairs = tl.arange(0, 2 * BLOCKS_PER_TILE)
    masked_word = MASK16 | (MASK16 << 16)
    for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
        q = row % Q
        pos = q + KV - Q
        for tile in range(tl.cdiv(KV // 2, 2 * BLOCKS_PER_TILE)):
            pair = tile * 2 * BLOCKS_PER_TILE + pairs
            keys = pair * 2
            if q < PREFIX:
                word = tl.where(keys <= pos, tl.full((2 * BLOCKS_PER_TILE,), 0, tl.uint32),
                                tl.full((2 * BLOCKS_PER_TILE,), masked_word, tl.uint32))
            else:
                table_row = (row.to(tl.int64) // Q) * S + q - PREFIX
                block = tile * BLOCKS_PER_TILE + blocks
                flags = tl.load(Table + table_row * STRIDE + block,
                                block < N, other=0).to(tl.float32)
                selected = tl.interleave(flags, flags) != 0
                tail = ((pos + 1) % 4 != 0) & (keys // 4 == (pos + 1) // 4)
                allow = (selected | tail) & (keys <= pos)
                word = tl.where(allow, tl.full((2 * BLOCKS_PER_TILE,), 0, tl.uint32),
                                tl.full((2 * BLOCKS_PER_TILE,), masked_word, tl.uint32))
            if pos % 2 == 0:
                word |= tl.where(keys == pos, tl.full((2 * BLOCKS_PER_TILE,), MASK16 << 16, tl.uint32),
                                 tl.full((2 * BLOCKS_PER_TILE,), 0, tl.uint32))
            tl.store(Out + row.to(tl.int64) * (KV // 2) + pair, word, pair < KV // 2)


def _launch_packed16(table, output, *, programs, batch, query_length, kv_length, prefix,
                     sparse_rows, compress_ratio, blocks, table_stride, masked_bits):
    """Launch the generic packed16 kernel, shrinking the tile if compilation fails.

    Returns false once every tile width has been rejected so the caller can
    fall back to the unpacked kernel instead of crashing mid-training.
    """
    global _PACKED16_TILE, _PACKED16_DISABLED
    # A failed compile is not cached by triton, so without this guard every
    # forward would retry the doomed launch and re-log the fallback warning.
    if _PACKED16_DISABLED:
        return False
    if _PACKED16_TILE is None:
        _PACKED16_TILE = _device_tile()
    while True:
        try:
            _qsa_id2mask_block_table_to_mask_packed16_kernel[(programs,)](
                table, output.view(torch.int32), query_length, kv_length, prefix, sparse_rows,
                compress_ratio, blocks, table_stride, batch * query_length,
                triton.next_power_of_2(blocks), _PACKED16_TILE,
                masked_bits & 65535, multibuffer=False,
            )
            return True
        except Exception as e:
            failed = _PACKED16_TILE
            if failed > 1024:
                _PACKED16_TILE = failed // 2
                logger.warning(
                    "qsa_id2mask packed16 kernel failed at TILE=%d (%s: %s); "
                    "retrying with TILE=%d", failed, type(e).__name__, e, _PACKED16_TILE)
                continue
            _PACKED16_DISABLED = True
            logger.warning(
                "qsa_id2mask packed16 kernel failed (%s: %s); falling back to "
                "the unpacked kernel for the rest of this run",
                type(e).__name__, e)
            return False


def _qsa_id2mask_forward(selected_blocks, *, query_length, kv_length,
                token_budget, compress_ratio, dtype):
    """Return [B,1,Q,KV], bool True=allowed or additive 0/finfo.min.

    Dense prefix, cache query offset, incomplete tail, invalid IDs and duplicate
    IDs follow the eager LightningIndexer adapter exactly. Input may be strided.
    """
    if selected_blocks.ndim != 3 or selected_blocks.dtype not in (torch.int32, torch.int64):
        raise ValueError("selected_blocks must be an int32/int64 [B,sparse_rows,topk] tensor")
    if selected_blocks.device.type != "npu":
        raise ValueError("Triton id2mask requires an NPU tensor")
    if not 0 <= query_length <= kv_length or compress_ratio <= 0 or token_budget <= 0:
        raise ValueError("Invalid QSA lengths, compression ratio or token budget")
    if token_budget % compress_ratio:
        raise ValueError("token_budget must be divisible by compress_ratio")
    if dtype not in (torch.bool, torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("id2mask supports bool, float16, bfloat16 and float32 masks")
    prefix = min(query_length, max(0, token_budget + compress_ratio - 1 - (kv_length - query_length)))
    batch, sparse_rows, topk = selected_blocks.shape
    if (sparse_rows, topk) != (query_length - prefix, token_budget // compress_ratio):
        raise ValueError("selected_blocks shape does not match the LightningIndexer complete-block output")
    if batch * query_length * kv_length == 0:
        return torch.empty((batch, 1, query_length, kv_length), device=selected_blocks.device, dtype=dtype)
    vector_num = get_vector_num(selected_blocks.device)
    blocks = triton.cdiv(kv_length, compress_ratio)
    # One sink column for invalid IDs; pad rows so native scatter and Triton
    # loads use aligned addresses. Only 0/1 membership is needed, even for
    # duplicate IDs, so int8 is exact and minimizes intermediate storage.
    table_stride = triton.cdiv(blocks + 1, 32) * 32
    table = torch.zeros((batch, max(1, sparse_rows), table_stride), dtype=torch.int8, device=selected_blocks.device)
    if sparse_rows:
        safe_ids = torch.empty(selected_blocks.shape, dtype=torch.int32, device=selected_blocks.device)
        _qsa_id2mask_sanitize_block_ids_kernel[(vector_num,)](
            selected_blocks, safe_ids, sparse_rows, topk, blocks,
            *selected_blocks.stride(), selected_blocks.numel(), 2048, multibuffer=False,
        )
        table.scatter_(-1, safe_ids, 1)

    output = torch.empty((batch, 1, query_length, kv_length), dtype=dtype, device=selected_blocks.device)
    # Store the exact bit patterns for 0 and finfo(dtype).min. This avoids a
    # BF16 constant-select lowering unsupported by the current Ascend compiler;
    # it does not change the output dtype or introduce a numerical approximation.
    integer_dtype, masked_bits = {
        torch.bool: (torch.bool, 0),
        torch.float16: (torch.int16, -1025),       # 0xfbff
        torch.bfloat16: (torch.int16, -129),       # 0xff7f
        torch.float32: (torch.int32, -8388609),    # 0xff7fffff
    }[dtype]
    storage = output if dtype == torch.bool else output.view(integer_dtype)
    if dtype in (torch.bfloat16, torch.float16) and kv_length % 2 == 0 and compress_ratio % 2 == 0:
        # The ratio-4 path expands each block flag into two output pairs;
        # the tile count follows the actual KV length rather than a fixed cutoff.
        tuned = compress_ratio == 4 and is_arch35()
        packed = True
        if tuned:
            blocks_per_tile = min(1024, triton.next_power_of_2(blocks))
            _qsa_id2mask_ratio4_packed16_kernel[(min(vector_num, batch * query_length),)](
                table, output.view(torch.int32), query_length, kv_length, prefix,
                sparse_rows, blocks, table_stride, batch * query_length,
                blocks_per_tile, masked_bits & 65535, multibuffer=False,
            )
        else:
            packed = _launch_packed16(
                table, output, programs=vector_num, batch=batch, query_length=query_length,
                kv_length=kv_length, prefix=prefix, sparse_rows=sparse_rows,
                compress_ratio=compress_ratio, blocks=blocks,
                table_stride=table_stride, masked_bits=masked_bits)
        if packed:
            return output
    _qsa_id2mask_block_table_to_mask_kernel[(vector_num,)](
        table, storage, query_length, kv_length, prefix, sparse_rows, compress_ratio,
        blocks, table_stride, batch * query_length, triton.next_power_of_2(blocks),
        _device_tile(), masked_bits, dtype == torch.bool, multibuffer=False,
    )
    return output


class _QSAIdToMaskFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selected_blocks, query_length, kv_length,
                token_budget, compress_ratio, dtype):
        output = _qsa_id2mask_forward(
            selected_blocks, query_length=query_length, kv_length=kv_length,
            token_budget=token_budget, compress_ratio=compress_ratio, dtype=dtype,
        )
        ctx.mark_non_differentiable(output)
        return output


def qsa_id_to_mask(selected_blocks, *, query_length, kv_length,
                   token_budget, compress_ratio, dtype):
    """Build a non-differentiable [B, 1, Q, KV] mask from integer block IDs."""
    return _QSAIdToMaskFunction.apply(
        selected_blocks, query_length, kv_length,
        token_budget, compress_ratio, dtype,
    )
