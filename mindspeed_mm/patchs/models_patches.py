# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.transformer.module import fp32_to_float16, float16_to_fp32, Float16Module

# support qwen3omni multi-card communication
from torch import nn
from torch.nn import functional as F
try:
    # support newer transformers for qwen3omni
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerTextExperts
except ImportError:
    Qwen3OmniMoeThinkerConfig = None
    Qwen3OmniMoeThinkerTextExperts = None


def _transform_model_dtype(module, transform, ignore_child_modules):
    for child_module in module.children():
        if child_module in ignore_child_modules:
            continue
        child_module = transform(child_module)
    return module


def float16Module_init(self, config, module):
    super(Float16Module, self).__init__(config)

    # if AEModel use fp32
    ae_config = getattr(config.mm.model, "ae", None)
    ae_float32 = False
    if ae_config is not None and getattr(ae_config, "dtype", None) == torch.float32:
        module.ae = module.ae.float()
        ae_float32 = True

    if config.fp16:
        if ae_float32:
            self.add_module("module", _transform_model_dtype(module, lambda module: module.half(), [module.ae]))
        else:
            self.add_module("module", module.half())

        def float16_convertor(val):
            return val.half()

    elif config.bf16:
        if ae_float32:
            self.add_module("module", _transform_model_dtype(module, lambda module: module.bfloat16(), [module.ae]))
        else:
            self.add_module("module", module.bfloat16())

        def float16_convertor(val):
            return val.bfloat16()
    else:
        raise Exception("should not be here")

    self.float16_convertor = float16_convertor


def float16Module_forward(self, *inputs, **kwargs):
    args = get_args()
    if mpu.is_pipeline_first_stage():
        # if AEModel use fp16 or bf16
        ae_config = getattr(args.mm.model, "ae", None)
        if ae_config is not None and getattr(ae_config, "dtype", None) != torch.float32:
            inputs = fp32_to_float16(inputs, self.float16_convertor)
    outputs = self.module(*inputs, **kwargs)
    if mpu.is_pipeline_last_stage():
        outputs = float16_to_fp32(outputs)
    return outputs


class Qwen3OmniMoeThinkerTextSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3OmniMoeThinkerConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3OmniMoeThinkerTextExperts(config)
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)
        
        # orresponding to the original route_tokens_to_experts function: start
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        full_routing_weights = torch.zeros_like(routing_weights)
        full_routing_weights.scatter_(1, top_k_indices, top_k_weights)
        
        if self.norm_topk_prob:
            top_k_sum = full_routing_weights.sum(dim=-1, keepdim=True)
            top_k_sum = torch.clamp(top_k_sum, min=1e-9) # avoid division by zero
            full_routing_weights /= top_k_sum
        
        full_routing_weights = full_routing_weights.to(hidden_states.dtype)
        # orresponding to the original route_tokens_to_experts function: end

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )
        for expert_idx in range(self.experts.num_experts):
            expert_layer = self.experts[expert_idx]
            expert_weights = full_routing_weights[:, expert_idx, None] # expert_weights: (batch*seq, 1)
            # [warning] all samples are involved in the current expert calculation
            current_hidden_states = expert_layer(hidden_states_reshaped) * expert_weights
            final_hidden_states += current_hidden_states
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states
