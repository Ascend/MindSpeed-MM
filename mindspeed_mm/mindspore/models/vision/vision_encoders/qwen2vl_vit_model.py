# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import lru_cache
import torch
import torch.nn.functional as F
from megatron.core import mpu
from megatron.training import get_args
import mindspore as ms

try:
    from mindspeed.utils import set_actual_seq_len
except ImportError:
    set_actual_seq_len = None


def get_window_index(self, grid_thw):
    # convert to tuple,
    grid_thw_tuple = tuple(map(tuple, grid_thw.numpy()))

    @lru_cache(maxsize=32, typed=True)
    def get_window_index_cache(grid_thw):
        window_index = []
        cu_window_seqlens = [0]
        window_index_id = 0
        vit_merger_window_size = self.config.window_attn_size // self.spatial_merge_size // self.config.patch_size

        for grid_t, grid_h, grid_w in grid_thw:
            grid_t = grid_t.item()
            grid_h = grid_h.item()
            grid_w = grid_w.item()
            llm_grid_h, llm_grid_w = (
                grid_h // self.spatial_merge_size,
                grid_w // self.spatial_merge_size,
            )
            index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
            pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
            pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
            num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
            num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
            index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
            index_padded = index_padded.reshape(
                grid_t,
                num_windows_h,
                vit_merger_window_size,
                num_windows_w,
                vit_merger_window_size,
            )
            index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
                grid_t,
                num_windows_h * num_windows_w,
                vit_merger_window_size,
                vit_merger_window_size,
            )
            seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
            index_padded = index_padded.reshape(-1)
            index_new = index_padded[index_padded != -100]
            window_index.append(index_new + window_index_id)
            cu_seqlens_tmp = seqlens.cumsum(0) * self.spatial_merge_size * self.spatial_merge_size + cu_window_seqlens[-1]
            cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
            window_index_id += (grid_t * llm_grid_h * llm_grid_w)
        window_index = torch.cat(window_index, dim=0)

        return window_index, cu_window_seqlens

    return get_window_index_cache(grid_thw_tuple)


def qwen2vl_vit_forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """
    Forward function of the Qwen2VL ViT Model. This function passes the input tensors
    through the embedding layer and then the transformer.

    """
    if self.pre_process:
        if pixel_values is None or grid_thw is None:
            raise ValueError('You have to specify pixel_values and grid_thw')
        else:
            hidden_states = self.patch_embed(pixel_values)
            hidden_states = hidden_states.unsqueeze(1)
    else:
        hidden_states = None

    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    seq_len = hidden_states.shape[0] if hidden_states is not None else pixel_values.shape[-2]
    window_index = None
    window_mask = None
    cu_window_seqlens = None
    if getattr(self.config, 'window_attn_size', None) is not None:
        if getattr(self.config, 'fullatt_block_indexes', None) is None:
            raise ValueError("The 'fullatt_block_indexes' attribute is required when using 'window_attn_size'.")
        window_index, cu_window_seqlens = self.get_window_index(grid_thw)
        cu_window_seqlens = torch.tensor(
            cu_window_seqlens,
            device=grid_thw.device,
            dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
        )
        cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

        spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size
        if self.pre_process:
            hidden_states = hidden_states.squeeze(1)
            hidden_states = hidden_states.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
            hidden_states = hidden_states[window_index, :, :]
            hidden_states = hidden_states.reshape(seq_len, -1)
            hidden_states = hidden_states.unsqueeze(1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // spatial_merge_unit, spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        if not get_args().use_flash_attn:
            window_mask = torch.full(
                [1, seq_len, seq_len], torch.finfo(pixel_values.dtype).min, device=pixel_values.device,
                dtype=torch.bool
            )
            cu_window_seqlens_np = cu_window_seqlens.numpy()
            window_mask = window_mask.numpy()
            for i in range(1, len(cu_window_seqlens_np)):
                window_mask[..., cu_window_seqlens_np[i - 1]: cu_window_seqlens_np[i],
                cu_window_seqlens_np[i - 1]: cu_window_seqlens_np[i]] = 0
            window_mask = ms.from_numpy(window_mask)

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    if get_args().use_flash_attn:
        if set_actual_seq_len is None:
            raise AssertionError("Please check the commit id of your MindSpeed")
        set_actual_seq_len(tuple(cu_seqlens.numpy()[1:].tolist()))
        attention_mask = None
        window_mask = None
    else:
        attention_mask = torch.full(
            [1, seq_len, seq_len], torch.finfo(pixel_values.dtype).min, device=pixel_values.device,
            dtype=torch.bool
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1]: cu_seqlens[i], cu_seqlens[i - 1]: cu_seqlens[i]] = 0

    if get_args().sequence_parallel:
        hidden_states = scatter_to_sequence_parallel_region(hidden_states)

    if mpu.get_context_parallel_world_size() > 1:
        split_gather_sizes = cal_split_sizes(hidden_states.shape[0], mpu.get_context_parallel_world_size())
        rotary_pos_emb = split_forward_gather_backward(
            rotary_pos_emb,
            mpu.get_context_parallel_group(),
            0,
            split_gather_sizes,
            "down"
        )
        hidden_states = split_forward_gather_backward(
            hidden_states,
            mpu.get_context_parallel_group(),
            0,
            split_gather_sizes,
            "down"
        )

    cos_cache = rotary_pos_emb.cos().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
    sin_cache = rotary_pos_emb.sin().unsqueeze(1).repeat(1, 1, 2).unsqueeze(1).float()
    rotary_pos_emb = (cos_cache, sin_cache)
    hidden_states = self.blocks(
        hidden_states=hidden_states,
        rotary_pos_emb=rotary_pos_emb,
        attention_mask=attention_mask,
        window_mask=window_mask,
        cu_seqlens=cu_seqlens,
        cu_window_seqlens=cu_window_seqlens
    )

    if mpu.get_context_parallel_world_size() > 1:
        hidden_states = gather_forward_split_backward(
            hidden_states,
            mpu.get_context_parallel_group(),
            0,
            split_gather_sizes,
            "up"
        )

    if get_args().sequence_parallel:
        hidden_states = gather_from_sequence_parallel_region(hidden_states)

    if get_args().use_flash_attn:
        set_actual_seq_len(None)
    return hidden_states, window_index
