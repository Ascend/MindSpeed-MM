import os
from typing import List, Optional

import torch
import torch.distributed as dist

from mindspeed.fsdp.distributed.dist_ops import all_to_all as _all_to_all
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.ops.moe_ops.gemm_mc2 import grouped_matmul_all2all, all2all_grouped_matmul
from mindspeed_mm.fsdp.ops.swiglu import swiglu, clamp_swiglu

# Enable forced expert balance for debugging purposes only.
# Set environment variable export MM_FORCE_EP_BALANCE=1 to activate.
# MUST BE DISABLED during formal training.
FORCE_EP_BALANCE = int(os.getenv("MM_FORCE_EP_BALANCE", "0")) == 1


def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_sizes: List = None,
    gather_sizes: List = None
):
    return _all_to_all(process_group, input_, gather_sizes, scatter_sizes)


def force_ep_balance(
    num_experts: int,
    selected_experts: torch.Tensor
) -> torch.Tensor:
    seq_len, activation_num = selected_experts.shape

    _indices = torch.arange(
        seq_len * activation_num,
        dtype=selected_experts.dtype,
        device=selected_experts.device
    ) % num_experts
    selected_experts = _indices.view(seq_len, activation_num)

    return selected_experts


def ep_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    if FORCE_EP_BALANCE:
        selected_experts = force_ep_balance(num_experts, selected_experts)

    if routing_weights.size() != selected_experts.size():
        routing_weights = routing_weights.gather(1, selected_experts)

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert = (
        dispatch_preprocess(selected_experts, num_experts, ep_group)
    )
    hidden_states, unpermute_indices, post_dispatch_unpermute_indices = alltoall_dispatch(
        hidden_states,
        selected_experts,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
        fused=fused,
    )

    # If no tokens are assigned to the expert in the current EP shard, no computation is performed
    if hidden_states.shape[0] > 0:
        intermediate_hidden_states = grouped_matmul(hidden_states, fc1_weight, num_global_sum_tokens_per_local_expert, fused=fused)
        if swiglu_limit > 0:
            intermediate_activations = clamp_swiglu(intermediate_hidden_states, dim=-1, fused=fused, limit=swiglu_limit)
        else:
            intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=fused)
        hidden_states = grouped_matmul(
            intermediate_activations, fc2_weight, num_global_sum_tokens_per_local_expert, fused=fused
        )
    else:
        # empty operation to avoid no grads for experts' weights
        intermediate_hidden_states = hidden_states @ fc1_weight.sum(0)
        gate_output, down_output = torch.chunk(intermediate_hidden_states, 2, dim=-1)
        hidden_states = (gate_output + down_output) @ fc2_weight.sum(0) * 0.

    hidden_states = alltoall_combine(
        hidden_states,
        routing_weights,
        post_dispatch_unpermute_indices,
        unpermute_indices,
        input_splits,
        output_splits,
        num_experts,
        num_global_tokens_per_local_expert,
        ep_group,
    )
    return hidden_states


def dispatch_preprocess(
    selected_experts: torch.Tensor,
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    if ep_group is None:
        ep_size = 1
        ep_rank = 0
    else:
        ep_size = dist.get_world_size(ep_group)
        ep_rank = dist.get_rank(ep_group)
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )
    num_local_experts = num_global_experts // ep_size

    num_local_tokens_per_expert = torch.histc(selected_experts.view(-1), bins=num_global_experts, min=0, max=num_global_experts)

    if ep_group is None or ep_size <= 1:
        num_global_tokens_per_expert = num_local_tokens_per_expert.view(1, -1)
    else:
        num_global_tokens_per_expert = torch.zeros(
            ep_size,
            num_global_experts,
            dtype=num_local_tokens_per_expert.dtype,
            device=num_local_tokens_per_expert.device,
        )
        dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()

    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()

    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0)
    return input_splits, output_splits, num_global_tokens_per_local_expert, num_global_sum_tokens_per_local_expert


def alltoall_dispatch(
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
):
    hidden_states, unpermute_indices = permute(hidden_states, selected_experts.to(torch.int32), fused=fused)
    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=input_splits, gather_sizes=output_splits)

    # No tokens have been assigned to the expert in the current EP shard
    if hidden_states.shape[0] == 0:
        return hidden_states, unpermute_indices, None

    ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
    num_local_experts = num_global_experts // ep_size
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
    )

    _expert_ids_per_ep_rank = torch.arange(num_global_experts, dtype=torch.int32, device=hidden_states.device) % num_local_experts
    global_input_tokens_local_experts_indices = torch.repeat_interleave(_expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    hidden_states, post_dispatch_unpermute_indices = permute(hidden_states, global_input_tokens_local_experts_indices, fused=fused)

    return hidden_states, unpermute_indices, post_dispatch_unpermute_indices


def alltoall_combine(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    post_dispatch_unpermute_indices: torch.Tensor,
    unpermute_indices: torch.Tensor,
    input_splits: List,
    output_splits: List,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
):
    # If no tokens are assigned to the expert in the current EP shard, no computation is performed
    if hidden_states.shape[0] > 0:
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        if num_global_experts % ep_size != 0:
            raise ValueError(
                f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
        )

        hidden_states = unpermute(hidden_states, post_dispatch_unpermute_indices, fused=fused)

    hidden_states = all_to_all(hidden_states, ep_group, scatter_sizes=output_splits, gather_sizes=input_splits)
    hidden_states = unpermute(hidden_states.to(routing_weights.dtype), unpermute_indices,
                                                      probs=routing_weights, fused=fused)
    return hidden_states


def ep_mc2_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    swiglu_limit: float = 0.0,
) -> torch.Tensor:
    if FORCE_EP_BALANCE:
        selected_experts = force_ep_balance(num_experts, selected_experts)

    if not fused:
        raise ValueError(f"ep mc2 only support fused = True")

    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    ep_size = dist.get_world_size(ep_group)
    ep_rank = dist.get_rank(ep_group)
    num_local_experts = num_experts // ep_size

    num_local_tokens_per_expert = torch.histc(selected_experts.view(-1), bins=num_experts, min=0, max=num_experts)

    num_global_tokens_per_expert = torch.zeros(
        ep_size,
        num_experts,
        dtype=num_local_tokens_per_expert.dtype,
        device=num_local_tokens_per_expert.device
    ) # [ep_size, num_experts]
    dist.all_gather_into_tensor(num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group)

    start_idx, end_idx = ep_rank * num_local_experts, (ep_rank + 1) * num_local_experts

    send_counts = num_local_tokens_per_expert
    recv_counts = num_global_tokens_per_expert[:, start_idx:end_idx].reshape(-1)

    hidden_states, unpermute_indices = permute(hidden_states, selected_experts.to(torch.int32), fused=fused)

    intermediate_hidden_states = all2all_grouped_matmul(
        inputs=hidden_states, weights=fc1_weight, group=ep_group, send_counts=send_counts, recv_counts=recv_counts
    )
    if swiglu_limit > 0:
        intermediate_activations = clamp_swiglu(intermediate_hidden_states, dim=-1, fused=fused, limit=swiglu_limit)
    else:
        intermediate_activations = swiglu(intermediate_hidden_states, dim=-1, fused=fused)

    hidden_states = grouped_matmul_all2all(
        inputs=intermediate_activations, weights=fc2_weight, group=ep_group, send_counts=recv_counts, recv_counts=send_counts
    )
    hidden_states = unpermute(
        hidden_states, unpermute_indices, probs=routing_weights, fused=True
    )

    return hidden_states