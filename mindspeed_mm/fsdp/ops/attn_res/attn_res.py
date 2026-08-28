# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The multi-head latent attention, MoE gating and sparse MoE block in this file are
# adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py). They have been
# extensively modified and extended for the Kimi-Linear architecture.
#
# Licensing Information:
# - Code adapted from DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
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
import torch.nn as nn

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

if IS_NPU_AVAILABLE:
    import torch_npu


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
    attn_res_implementation: str = "eager"
) -> torch.Tensor:
    """
    prefix_sum:     (num_tokens, hidden_size)
    block_residual: (num_tokens, num_blocks, hidden_size)
    """
    if not IS_NPU_AVAILABLE or attn_res_implementation == "eager":
        v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
        v_float = v.float()
        variance = v_float.pow(2).mean(-1, keepdim=True)
        k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
        score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
        scores = (k * score_weight).sum(-1)
        probs = scores.softmax(-1).unsqueeze(1)
        hidden_states = torch.matmul(probs, v_float).squeeze(1)
        return hidden_states.to(v.dtype)
    elif attn_res_implementation == "ascendc":
        from cann_ops_transformer.ops.attn_res import attn_res
        return attn_res(prefix_sum, block_residual, proj.weight, norm.weight,
                        norm_eps=norm.variance_epsilon)
    else:
        raise ValueError(
            f"Unsupported attn_res_implementation: {attn_res_implementation}. "
            "Expected 'eager' or 'ascendc'."
        )
