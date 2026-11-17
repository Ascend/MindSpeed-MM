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
from torch import nn
from torch.nn import functional as F


def fp32_layer_norm_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    origin_dtype = inputs.dtype
    return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float() if self.weight is not None else None,
                        self.bias.float() if self.bias is not None else None, self.eps).to(origin_dtype)


def fp32_silu_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(inputs.float(), inplace=self.inplace).to(inputs.dtype)


def fp32_gelu_forward(self, inputs: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(inputs.float(), approximate=self.approximate).to(inputs.dtype)


def replace_with_fp32_forwards():
    nn.GELU.forward = fp32_gelu_forward
    nn.SiLU.forward = fp32_silu_forward
    nn.LayerNorm.forward = fp32_layer_norm_forward
