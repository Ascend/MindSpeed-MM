# coding=utf-8
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import lru_cache
import torch
import torch.nn.functional as F
from mindspeed_mm.models.vision.vision_encoders.qwen2vl_vit_model import apply_rotary_pos_emb_vision


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


def qwen2vlvit_selfattention_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_context=None,
    rotary_pos_emb=None,
    rotary_pos_cos=None,
    rotary_pos_sin=None,
    attention_bias=None,
    packed_seq_params=None,
    sequence_len_offset=None,
    inference_params=None,
):

    # hidden_states shape is [sq, b, h]
    # For self attention we just duplicate the rotary_pos_emb if it isn't already

    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)

    if self.config.context_parallel_size > key.shape[2]:
        key = key.repeat_interleave(
            query.shape[2] // key.shape[2], dim=2
        )
        value = value.repeat_interleave(
            query.shape[2] // value.shape[2], dim=2
        )
    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================
    query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_context,
        query,
        key,
        value,
        rotary_pos_emb,
        rotary_pos_cos,
        rotary_pos_sin,
        sequence_len_offset,
    )

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================

    # TODO, can apply positional embedding to value_layer so it has
    # absolute positional embedding.
    # otherwise, only relative positional embedding takes effect
    if rotary_pos_emb is not None:
        query = apply_rotary_pos_emb_vision(query, rotary_pos_emb,
                                            use_fused_rope=self.config.use_fused_rotary_pos_emb)
        key = apply_rotary_pos_emb_vision(key, rotary_pos_emb,
                                            use_fused_rope=self.config.use_fused_rotary_pos_emb)

    if packed_seq_params is not None:
        query = query.squeeze(1)
        key = key.squeeze(1)
        value = value.squeeze(1)

    # ==================================
    # core attention computation
    # ==================================
    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            packed_seq_params=packed_seq_params,
        )

    if packed_seq_params is not None:
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], 1, -1)

    # =================
    # Output. [sq, b, h]
    # =================
    output, bias = self.linear_proj(core_attn_out)
    return output, bias
    