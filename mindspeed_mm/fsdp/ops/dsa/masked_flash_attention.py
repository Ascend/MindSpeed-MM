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
"""Flash attention over the exact token selection of a NoPE MLA indexer."""

from functools import lru_cache
from typing import NamedTuple, Optional

import torch
from torch.nn.functional import linear

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


class PoolSelection(NamedTuple):
    """Compact selected-pool metadata passed directly to ``masked_flash_attention``."""

    pools: torch.Tensor
    valid: torch.Tensor
    num_pools: int
    always_select_tail: bool
    valid_lengths: Optional[torch.Tensor] = None


@lru_cache(maxsize=None)
def _npu_vector_cores(device_index: int) -> int:
    try:
        from triton.runtime import driver

        props = driver.active.utils.get_device_properties(device_index)
        return int(props.get("num_vectorcore") or props.get("num_aicore") or 48)
    except Exception:
        return 48


if triton is not None:

    @triton.jit(do_not_specialize=["N", "LEN"])
    def _sanitize_pool_ids_kernel(ids_ptr, valid_ptr, out_ptr, N, LEN, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        for base in range(tl.program_id(0) * BLOCK, LEN, tl.num_programs(0) * BLOCK):
            idx = base.to(tl.int64) + offs
            m = idx < LEN
            p = tl.load(ids_ptr + idx, mask=m, other=-1).to(tl.int32)
            v = tl.load(valid_ptr + idx, mask=m, other=0).to(tl.int32)
            tl.store(out_ptr + idx, tl.where((v != 0) & (p >= 0) & (p < N), p, N).to(tl.int32), mask=m)

    @triton.jit(do_not_specialize=["S", "W", "N", "TABLE_STRIDE", "ROWS"])
    def _pool_table_to_mask_packed4_kernel(
        table_ptr, valid_len_ptr, out_ptr, nonempty_ptr, S, W, N, TABLE_STRIDE, ROWS,
        HAS_LEN: tl.constexpr, ALWAYS_TAIL: tl.constexpr, SINGLE_TILE: tl.constexpr, TILE_W: tl.constexpr,
    ):
        offs = tl.arange(0, TILE_W)
        all_ignore = tl.full((TILE_W,), 0x01010101, tl.int32)
        all_attend = tl.full((TILE_W,), 0, tl.int32)
        for row in range(tl.program_id(0), ROWS, tl.num_programs(0)):
            b = row // S
            q = row % S
            row64 = row.to(tl.int64)
            if HAS_LEN:
                q_valid = q.to(tl.int64) < tl.load(valid_len_ptr + b).to(tl.int64)
                visible = tl.where(q_valid, q + 1, 0)
            else:
                visible = q + 1
            full_pools = visible // 4
            valid_pool_end = tl.minimum(full_pools, N)
            tail_count = (visible % 4) if ALWAYS_TAIL else 0
            if ALWAYS_TAIL:
                tail_word = tl.where(
                    tail_count == 1, 0x01010100, tl.where(tail_count == 2, 0x01010000, 0x01000000)
                )

            if SINGLE_TILE:
                flags = tl.load(table_ptr + row64 * TABLE_STRIDE + offs, mask=offs < valid_pool_end, other=0)
                has_pool = tl.sum((flags != 0).to(tl.int32), axis=0) > 0
                nonempty = has_pool | (tail_count > 0)
                tl.store(nonempty_ptr + row64, nonempty.to(tl.uint8))
                word = tl.where(flags != 0, all_attend, all_ignore)
                if ALWAYS_TAIL:
                    word = tl.where((tail_count > 0) & (offs == full_pools), tail_word, word)
                word = tl.where((~nonempty) & (offs == 0), 0x01010100, word)
                tl.store(out_ptr + row64 * W + offs, word, mask=offs < W)
            else:
                pool_hits = 0
                for p_start in range(0, valid_pool_end, TILE_W):
                    p = p_start + offs
                    f_tile = tl.load(table_ptr + row64 * TABLE_STRIDE + p, mask=p < valid_pool_end, other=0)
                    pool_hits += tl.sum((f_tile != 0).to(tl.int32), axis=0)
                has_pool = pool_hits > 0
                nonempty = has_pool | (tail_count > 0)
                tl.store(nonempty_ptr + row64, nonempty.to(tl.uint8))
                for w_start in range(0, W, TILE_W):
                    wg = w_start + offs
                    flags = tl.load(table_ptr + row64 * TABLE_STRIDE + wg, mask=wg < valid_pool_end, other=0)
                    word = tl.where(flags != 0, all_attend, all_ignore)
                    if ALWAYS_TAIL:
                        word = tl.where((tail_count > 0) & (wg == full_pools), tail_word, word)
                    word = tl.where((~nonempty) & (wg == 0), 0x01010100, word)
                    tl.store(out_ptr + row64 * W + wg, word, mask=wg < W)


def build_mask(topk_indices: torch.Tensor, kv_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a uint8 ignore mask (1 = ignore, 0 = attend) and nonempty query indicator."""
    valid = (topk_indices >= 0) & (topk_indices < kv_length)
    nonempty = valid.any(dim=-1)
    safe = torch.where(valid, topk_indices, 0).long()
    mask = torch.ones((*topk_indices.shape[:2], kv_length), dtype=torch.uint8, device=topk_indices.device)
    mask.scatter_(-1, safe, 0)
    mask[..., 0].copy_(nonempty & ~topk_indices.eq(0).any(dim=-1))
    # Pass uint8 directly to npu_fusion_attention (CANN 9.2 fwd/bwd) to avoid AI_CPU uint8->bool Cast on A3.
    return mask.unsqueeze(1), nonempty


def build_mask_from_pools(
    selected_pools: torch.Tensor, selected_valid: torch.Tensor, kv_length: int,
    num_pools: int, always_select_tail: bool, valid_lengths: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a uint8 ignore mask directly from selected pool IDs (pool_size=4)."""
    if selected_pools.ndim != 3 or selected_valid.shape != selected_pools.shape or num_pools < 0:
        raise ValueError("selected_pools and selected_valid must match [B, S, K] with num_pools >= 0.")
    if selected_pools.dtype not in (torch.int32, torch.int64) or selected_valid.dtype != torch.bool:
        raise TypeError("selected_pools must be int32/int64 and selected_valid must be bool.")
    batch_size, seq_len, k = selected_pools.shape
    if kv_length != seq_len or kv_length <= 0 or kv_length % 4 != 0:
        raise ValueError("build_mask_from_pools requires kv_length == seq_len > 0 divisible by 4.")
    if valid_lengths is not None and (
        valid_lengths.shape != (batch_size,) or valid_lengths.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("valid_lengths must be an int32/int64 tensor of shape [B].")
    device = selected_pools.device
    if selected_valid.device != device or (valid_lengths is not None and valid_lengths.device != device):
        raise ValueError("selected_pools, selected_valid, and valid_lengths must be on the same device.")

    total_rows, table_stride, w = batch_size * seq_len, num_pools + 1, kv_length // 4
    table = torch.zeros((total_rows, table_stride), dtype=torch.int8, device=device)

    if triton is not None and device.type == "npu":
        vec_cores = _npu_vector_cores(device.index or 0)
        if k > 0:
            safe = torch.empty((total_rows, k), dtype=torch.int32, device=device)
            total_elems = total_rows * k
            _sanitize_pool_ids_kernel[(min(vec_cores, triton.cdiv(total_elems, 4096)),)](
                selected_pools.contiguous(), selected_valid.contiguous(), safe, num_pools, total_elems,
                BLOCK=4096, multibuffer=False,
            )
            table.scatter_(-1, safe, 1)
        output = torch.empty((batch_size, 1, seq_len, kv_length), dtype=torch.uint8, device=device)
        nonempty = torch.empty((batch_size, seq_len), dtype=torch.bool, device=device)
        has_len = valid_lengths is not None
        tile_w = min(max(triton.next_power_of_2(w), 16), 2048)
        _pool_table_to_mask_packed4_kernel[(min(vec_cores, max(1, total_rows)),)](
            table, valid_lengths.contiguous() if has_len else table,
            output.view(torch.int32), nonempty.view(torch.uint8),
            seq_len, w, num_pools, table_stride, total_rows,
            HAS_LEN=has_len, ALWAYS_TAIL=always_select_tail, SINGLE_TILE=(w <= tile_w), TILE_W=tile_w,
            multibuffer=False,
        )
        return output, nonempty

    if k > 0:
        ok = selected_valid & (selected_pools >= 0) & (selected_pools < num_pools)
        safe = torch.where(ok, selected_pools, num_pools).reshape(total_rows, k).long()
        table.scatter_(-1, safe, 1)

    q = torch.arange(seq_len, dtype=torch.int64, device=device)[None, :]
    q_valid = q < valid_lengths.to(torch.int64)[:, None] if valid_lengths is not None else torch.ones(
        (batch_size, seq_len), dtype=torch.bool, device=device
    )
    visible = torch.where(q_valid, q + 1, 0)
    flags = torch.zeros((batch_size, seq_len, w), dtype=torch.bool, device=device)
    if (copy_n := min(num_pools, w)) > 0:
        flags[..., :copy_n] = table.view(batch_size, seq_len, table_stride)[..., :copy_n].bool()
    pool_ok = flags & (torch.arange(w, device=device) < (visible // 4).clamp(max=num_pools).unsqueeze(-1))
    allow = pool_ok.repeat_interleave(4, dim=-1)
    if always_select_tail:
        tail = visible % 4
        keys = torch.arange(kv_length, device=device)
        allow = allow | ((tail.unsqueeze(-1) > 0) & (keys > (q - tail).unsqueeze(-1)) & (keys <= q.unsqueeze(-1)))
    nonempty = q_valid & allow.any(dim=-1)
    mask = (~allow).to(torch.uint8)
    mask[..., 0] &= nonempty.to(torch.uint8)
    return mask.unsqueeze(1), nonempty


def masked_flash_attention(
    query: torch.Tensor, latent: torch.Tensor, w_uk: torch.Tensor, w_uv: torch.Tensor,
    selection: torch.Tensor | PoolSelection, scale: float,
) -> torch.Tensor:
    """Project the shared latent to per-head K/V and run causal BSND attention."""
    import torch_npu

    batch_size, kv_length, _, _ = latent.shape
    if query.size(1) != kv_length:
        raise ValueError("masked_fa requires equal query and key sequence lengths.")

    # Project K and V separately from the shared latent so gradients accumulate back into kv_b_proj.
    key = linear(latent.squeeze(2), w_uk.flatten(0, 1)).view(batch_size, kv_length, w_uk.size(0), w_uk.size(1))
    value = linear(latent.squeeze(2), w_uv.flatten(0, 1)).view(batch_size, kv_length, w_uv.size(0), w_uv.size(1))
    if isinstance(selection, PoolSelection):
        mask, nonempty = build_mask_from_pools(
            selection.pools, selection.valid, kv_length,
            selection.num_pools, selection.always_select_tail, selection.valid_lengths,
        )
    else:
        mask, nonempty = build_mask(selection, kv_length)
    output = torch_npu.npu_fusion_attention(
        query.contiguous(), key, value, query.size(2), "BSND",
        atten_mask=mask, scale=scale, keep_prob=1.0, sparse_mode=0, pre_tockens=2147483647, next_tockens=0,
    )[0]
    return torch.where(nonempty[..., None, None], output, 0)
