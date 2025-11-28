# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from importlib.metadata import version as get_version

import torch
import torch_npu
from torch_npu import npu_rotary_mul as apply_rotary_emb
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.utils import logging
from transformers.activations import ACT2FN
import verl.third_party.vllm as vllm_sleep_level
from torch import nn

if get_version("transformers") > "4.57.1":
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3_moe import modeling_qwen3_moe
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe
    from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextExperts, \
        Qwen3VLMoeTextSparseMoeBlock
else:
    from transformers.modeling_utils import PretrainedConfig, PreTrainedModel


class GmmFunction_vl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        fwd_output = torch_npu.npu_grouped_matmul([x], [weight], bias=None, group_list=group_list,
                                                  split_item=2, group_type=0, group_list_type=1)[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)

        grad_input = torch_npu.npu_grouped_matmul([grad_output], [weight], bias=None, group_list=group_list,
                                                  split_item=2, group_type=0, group_list_type=1)[0]

        grad_weight = torch_npu.npu_grouped_matmul([input_tensor.T], [grad_output], bias=None, group_list=group_list,
                                                   split_item=3, group_type=2, group_list_type=1)[0]

        return grad_input, grad_weight, None


def npu_group_gemm(x, weight, group_list):
    output = GmmFunction_vl.apply(x, weight, group_list)
    return output


class Qwen3VLMoeTextExperts_npu(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.intermediate_size = config.moe_intermediate_size
        self.hidden_size = config.hidden_size
        self.expert_dim = self.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.expert_dim, self.hidden_size)))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
            self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        When training it is more efficient to just loop over the experts and compute the output for each expert
        as otherwise the memory would explode.

        For inference we can sacrifice some memory and compute the output for all experts at once. By repeating the inputs.

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            routing_weights (torch.Tensor): (batch_size * token_num, num_experts)
            router_indices (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)  # (num_tokens, hidden_size)
        if self.training:
            permuted_hidden_states, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states,
                                                                                  router_indices.to(torch.int32))
            tokens_per_expert = torch.histc(router_indices, bins=self.num_experts, min=0, max=self.num_experts)
            intermediate_hidden_states = npu_group_gemm(permuted_hidden_states, self.gate_up_proj, tokens_per_expert)
            intermediate_activations = torch_npu.npu_swiglu(intermediate_hidden_states, dim=-1)
            output = npu_group_gemm(intermediate_activations, self.down_proj, tokens_per_expert)
            next_states = torch_npu.npu_moe_token_unpermute(output, row_ids_map, probs=routing_weights)
            next_states = next_states.view(batch_size, -1, self.hidden_size)
        else:
            hidden_states = hidden_states.repeat(self.num_experts, 1)
            hidden_states = hidden_states.view(self.num_experts, -1, self.hidden_size)
            gate_up = torch.bmm(hidden_states, self.gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
            next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
            next_states = next_states.reshape(self.num_experts, batch_size, -1, self.hidden_size)
            next_states = (
                    next_states * routing_weights.transpose(0, 1).view(self.num_experts, batch_size, -1)[..., None]
            )
            next_states = next_states.sum(dim=0)
        return next_states


class Qwen3VLMoeTextSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3VLMoeTextExperts_npu(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        if not self.training:
            routing_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        routed_out = self.experts(hidden_states, routing_weights, router_indices)
        return routed_out


def rms_norm_forward(self, x):
    if x.dtype != self.weight.dtype:
        x = x.to(self.weight.dtype)
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def apply_rotary_pos_emb_qwen3_npu(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def silu_forward(self, hidden_state):
    """NPU optimized silu"""
    gate_up = torch.cat((self.gate_proj(hidden_state), self.up_proj(hidden_state)), dim=-1)
    return self.down_proj(torch_npu.npu_swiglu(gate_up, dim=-1))


def apply_npu_plugin():
    """
    Apply NPU optimization patches in correct order.
    Patches must be applied in specific sequence to ensure proper functionality.
    """

    # 1. Configure vLLM sleep level to reduce resource consumption
    # Setting to level 1 optimizes performance for NPU devices
    vllm_sleep_level.VLLM_SLEEP_LEVEL = 1

    # 2. Fix TensorDict synchronization for NPU devices
    #
    # Background:
    # - VERL uses Ray for distributed computation aggregation
    # - Workers package outputs in TensorDict and transfer to CPU
    # - TensorDict.to() is non-blocking by default
    #
    # Issue:
    # - Data must be fully transferred before host usage to avoid precision issues
    # - TensorDict fixed this for CUDA/MPS, but NPU support is missing
    #
    # Solution:
    # - Patch TensorDict synchronization for NPU devices
    # - This is temporary until official TensorDict support is added
    from tensordict.base import TensorDictBase

    def _sync_all_patch(self):
        from torch._utils import _get_available_device_type, _get_device_module
        try:
            from torch.compiler import is_compiling
        except ImportError:  # torch 2.0
            from torch._dynamo import is_compiling

        device_type = _get_available_device_type()
        if device_type is None:
            return

        if device_type == "cuda":
            if not is_compiling() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        else:
            device_module = _get_device_module(device_type)
            device_module.synchronize()

    TensorDictBase._sync_all = _sync_all_patch

    if get_version("transformers") > "4.57.1":
        modeling_qwen3_vl_moe.Qwen3VLMoeTextSparseMoeBlock = Qwen3VLMoeTextSparseMoeBlock
        modeling_qwen3_vl_moe.Qwen3VLMoeTextRMSNorm.forward = rms_norm_forward
        modeling_qwen3_vl_moe.apply_rotary_pos_emb = apply_rotary_pos_emb_qwen3_npu
        modeling_qwen3_vl.Qwen3VLTextRMSNorm.forward = rms_norm_forward
        modeling_qwen3_vl.Qwen3VLTextMLP.forward = silu_forward