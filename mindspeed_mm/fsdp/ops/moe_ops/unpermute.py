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

import torch
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


def eager_unpermute(permuted_tokens, sorted_indices, probs, restore_shape):
    num_tokens, topk = (permuted_tokens.size(0), 1) if probs is None else (probs.numel(), probs.size(1))
    if restore_shape:
        unpermuted_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        permuted_tokens = permuted_tokens[:restore_shape[0]]
    else:
        unpermuted_tokens = torch.zeros(
            [num_tokens, permuted_tokens.shape[-1]], dtype=permuted_tokens.dtype, device=permuted_tokens.device
        )
    unpermuted_tokens.index_copy_(0, torch.argsort(sorted_indices, stable=True), permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens *= probs.unsqueeze(-1)
    return unpermuted_tokens.sum(dim=1)


def fused_unpermute(permuted_tokens, sorted_indices, probs, restore_shape):
    if probs is not None:
        permuted_tokens = permuted_tokens.to(probs.dtype)
    return torch_npu.npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs, restore_shape=restore_shape)


def unpermute(permuted_tokens, sorted_indices, probs=None, restore_shape=None, fused=True):
    if restore_shape is None and permuted_tokens.size(0) != sorted_indices.numel():
        raise AssertionError(f"permuted tokens({permuted_tokens.size(0)}) != sorted indices({sorted_indices.size()})")
    if fused and IS_NPU_AVAILABLE:
        return fused_unpermute(permuted_tokens, sorted_indices, probs, restore_shape)
    else:
        return eager_unpermute(permuted_tokens, sorted_indices, probs, restore_shape)
