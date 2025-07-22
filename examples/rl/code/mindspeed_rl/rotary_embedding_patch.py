# coding=utf-8
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from typing import Tuple, Optional
import os

import torch
import vllm.model_executor.layers.rotary_embedding
from vllm.model_executor.layers.rotary_embedding import _apply_rotary_emb


def MRotaryEmbedding_forward(
    self,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """pytorch-native implementation equivalent to forward()
    
        Args:
            positions: 
                [num_tokens,] (text_only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
    """
    import torch_npu
    mrope_section = [0, 0, 0] if positions.ndim == 1 else self.mrope_section

    query, key = torch_npu.npu_mrope(positions.contiguous(),
                                        query.contiguous(),
                                        key.contiguous(),
                                        self.cos_sin_cache.contiguous(),
                                        self.head_size,
                                        mrope_section=mrope_section,
                                        rotary_mode='half')
        
    return query, key


def single_forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    num_tokens = positions.shape[-1]
    cos_sin = self.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    if positions.ndim == 2:
        cos = torch.cat([
            m[i]
            for i, m in enumerate(cos.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)
        sin = torch.cat([
            m[i]
            for i, m in enumerate(sin.split(self.mrope_section, dim=-1))
        ],
                        dim=-1)

    query_shape = query.shape
    query = query.view(num_tokens, -1, self.head_size)
    query_rot = query[..., :self.rotary_dim]
    query_pass = query[..., self.rotary_dim:]
    query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key_shape = key.shape
    key = key.view(num_tokens, -1, self.head_size)
    key_rot = key[..., :self.rotary_dim]
    key_pass = key[..., self.rotary_dim:]
    key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


def MRotaryEmbedding_forward_patch():
    vllm.model_executor.layers.rotary_embedding.MRotaryEmbedding.forward = MRotaryEmbedding_forward
