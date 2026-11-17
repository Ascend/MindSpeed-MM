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


def eager_permute(tokens, indices, num_out_tokens=None):
    topk = 1 if indices.dim() == 1 else indices.size(1)
    indices_dtype = indices.dtype
    sorted_indices = torch.argsort(indices.float().view(-1), stable=True)
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True).to(indices_dtype)
    if num_out_tokens:
        permuted_tokens = permuted_tokens[:num_out_tokens]
    return permuted_tokens, sorted_indices1


def fused_permute(tokens, indices, num_out_tokens=None):
    return torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=num_out_tokens)


def permute(tokens, indices, num_out_tokens=None, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_permute(tokens, indices, num_out_tokens)
    else:
        return eager_permute(tokens, indices, num_out_tokens)
