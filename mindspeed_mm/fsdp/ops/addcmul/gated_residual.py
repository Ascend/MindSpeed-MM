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

"""
Gated residual injection operations based on addcmul operations.
"""

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

def eager_residual_injection(hidden_states, injection_weights, hyper_input):
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)

def fused_residual_injection(hidden_states, injection_weights, hyper_input):
    from .addcmul_triton import triton_addcmul
    return triton_addcmul(hidden_states, injection_weights, hyper_input, round_product=True)


def gated_residual_injection(hidden_states, injection_weights, hyper_input, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_residual_injection(hidden_states, injection_weights, hyper_input)
    else:
        return eager_residual_injection(hidden_states, injection_weights, hyper_input)
