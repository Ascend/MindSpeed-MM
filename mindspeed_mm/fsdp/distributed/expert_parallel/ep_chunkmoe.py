import math
from typing import Callable, List, Optional, Tuple
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher import (
    apply_activation,
    force_ep_balance,
    get_num_global_tokens_per_local_expert
)
from mindspeed_mm.fsdp.params.parallel_args import EPPlanConfig

# Enable forced expert balance for debugging purposes only.
FORCE_EP_BALANCE = envs.MM_FORCE_EP_BALANCE
EP_BINCOUNT_DISPATCH = envs.MM_EP_BINCOUNT_DISPATCH


def _a2a_single_async(
    inputs: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    output_split_sizes: List = None,
    input_split_sizes: List = None,
    async_op: bool = True,
):
    if group is None or dist.get_world_size(group=group) == 1:
        return None, inputs
    assert inputs.shape[0] == sum(input_split_sizes), (
        f"inputs.shape[0]={inputs.shape[0]} != sum(input_split_sizes)={sum(input_split_sizes)}; "
        f"input_split_sizes={input_split_sizes}"
    )
    inputs = inputs.contiguous()
    if output_split_sizes is None:
        output = torch.empty_like(inputs)
    else:
        output = inputs.new_empty(size=[sum(output_split_sizes)] + list(inputs.size()[1:]),
                                      dtype=inputs.dtype, device=inputs.device)
    if async_op:
        work = dist.all_to_all_single(
            output, inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
    else:
        dist.all_to_all_single(
            output, inputs,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        work = None
    return work, output


class _A2AStart(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, group, output_split_sizes, input_split_sizes):
        work, output = _a2a_single_async(
            inputs, group, output_split_sizes, input_split_sizes, async_op=True)
        handle = [work]
        ctx.comm = handle
        return handle, output
    @staticmethod
    def backward(ctx, grad_comm, grad_output):
        if ctx.comm:
            work = ctx.comm.pop()
            if work is not None:
                work.wait()
        return None, None, None, None


class _A2AEnd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, comm, inputs, output, group, output_split_sizes, input_split_sizes):
        ctx.comm = comm
        if ctx.comm:
            work = ctx.comm.pop()
            if work is not None:
                work.wait()
        ctx.group = group
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        return output
    @staticmethod
    def backward(ctx, grad_y):
        work, grad_input = _a2a_single_async(
            grad_y.contiguous(),
            ctx.group,
            ctx.output_split_sizes,
            ctx.input_split_sizes,
            async_op=True,
        )
        ctx.comm.append(work)
        return None, grad_input, grad_y, None, None, None


def batched_pre_dispatch_metadata(
    chunk_se_list: List[torch.Tensor],
    num_global_experts: int,
    ep_group: Optional[dist.ProcessGroup] = None,
):
    """Do a batched all_gather plus a single D2H transfer of the expert token counts for all chunks.
    Returns per-chunk (input_splits, output_splits, ngt, ngs):
      input_splits[c]  : list[int], tokens this rank sends to each EP rank (host)
      output_splits[c] : list[int], tokens this rank receives from each EP rank (host)
      ngt[c]           : Tensor[ep_size, num_local_experts], global per-rank counts for local experts
      ngs[c]           : Tensor[num_local_experts], total tokens received by each local expert
    """
    num_chunks = len(chunk_se_list)
    if ep_group is None:
        ep_size, ep_rank = 1, 0
    else:
        ep_size, ep_rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
    if num_global_experts % ep_size != 0:
        raise ValueError(
            f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
        )
    num_local_experts = num_global_experts // ep_size
    counts = torch.stack([
        get_num_global_tokens_per_local_expert(se, num_global_experts) for se in chunk_se_list
    ], dim=0)
    if ep_group is None or ep_size <= 1:
        global_counts = counts.view(num_chunks, 1, num_global_experts)
        counts_host = global_counts.tolist()
    else:
        gathered = torch.zeros(
            ep_size, num_chunks, num_global_experts,
            dtype=counts.dtype, device=counts.device,
        )

        dist.all_gather_into_tensor(gathered.view(-1), counts.view(-1), group=ep_group)
        global_counts = gathered.permute(1, 0, 2).contiguous()  # [num_chunks, ep_size, num_experts]
        counts_host = global_counts.tolist()
    start_idx = ep_rank * num_local_experts
    end_idx = (ep_rank + 1) * num_local_experts
    input_splits_list, output_splits_list, ngt_list, ngs_list = [], [], [], []
    for c in range(num_chunks):
        if ep_size == 1:
            input_splits = [sum(counts_host[c][0])]
            output_splits = [sum(counts_host[c][0])]
            ngt = global_counts[c, :, start_idx:end_idx].contiguous()
        else:
            per_rank_counts = counts_host[c]  # [ep_size][num_global_experts]
            my_counts = per_rank_counts[ep_rank]
            input_splits = [
                sum(my_counts[r * num_local_experts:(r + 1) * num_local_experts])
                for r in range(ep_size)
            ]
            output_splits = [
                sum(per_rank_counts[r][start_idx:end_idx])
                for r in range(ep_size)
            ]
            ngt = global_counts[c, :, start_idx:end_idx].contiguous()
        ngs = ngt.sum(dim=0)
        input_splits_list.append(input_splits)
        output_splits_list.append(output_splits)
        ngt_list.append(ngt)
        ngs_list.append(ngs)
    return input_splits_list, output_splits_list, ngt_list, ngs_list


def post_dispatch_pre_combine(
    hidden_states: torch.Tensor,
    num_global_experts: int,
    num_global_tokens_per_local_expert: torch.Tensor,
    num_global_sum_tokens_per_local_expert: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    activation: Optional[Callable] = None,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
):
    post_dispatch_unpermute_indices = None
    if hidden_states.shape[0] > 0:
        ep_size = 1 if ep_group is None else dist.get_world_size(ep_group)
        num_local_experts = num_global_experts // ep_size
        if num_global_experts % ep_size != 0:
            raise ValueError(
                f"Number of experts ({num_global_experts}) must be divisible by expert parallel size ({ep_size})."
            )
        _expert_ids_per_ep_rank = torch.arange(num_global_experts, dtype=torch.int32, device=hidden_states.device) % num_local_experts
        global_input_tokens_local_experts_indices = torch.repeat_interleave(_expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
        hidden_states, post_dispatch_unpermute_indices = permute(hidden_states, global_input_tokens_local_experts_indices, fused=fused)
        intermediate_hidden_states = grouped_matmul(
            hidden_states, fc1_weight, num_global_sum_tokens_per_local_expert, fused=fused,
        )
        intermediate_activations = apply_activation(
            intermediate_hidden_states,
            dim=-1, fused=fused,
            swiglu_limit=swiglu_limit, swiglu_alpha=swiglu_alpha, activation=activation,
        )
        hidden_states = grouped_matmul(
            intermediate_activations, fc2_weight, num_global_sum_tokens_per_local_expert, fused=fused
        )
        hidden_states = unpermute(hidden_states, post_dispatch_unpermute_indices, fused=fused)
    else:
        intermediate_hidden_states = hidden_states @ fc1_weight.sum(0)
        gate_output, down_output = torch.chunk(intermediate_hidden_states, 2, dim=-1)
        hidden_states = (gate_output + down_output) @ fc2_weight.sum(0) * 0.
    return hidden_states


def make_forward_pair_closure(
    num_chunks: int,
    output_splits_list,      # cpu tensor / list[int]
    input_splits_list,       # cpu tensor / list[int]
    ngt_list,                # cpu tensor / list[int]
    ngs_list,                # cpu tensor / list[int]
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    swiglu_limit,
    swiglu_alpha,
    activation,
    ep_group,
    fused: bool,
    num_experts: int,
):
    def _inner(
        hs_c: torch.Tensor,
        se_c: torch.Tensor,
        rw_c: torch.Tensor,
        hs_n: torch.Tensor,
        se_n: torch.Tensor,
        rw_n: torch.Tensor,
        has_next: bool,
        c: int,
    ):
        # ---------------- c chunk permute + A2A1 Start ----------------
        _hs, _unp_idx = permute(hs_c, se_c.to(torch.int32), fused=fused)
        comm1_c, a2a1_out_c = _A2AStart.apply(
            _hs, ep_group, output_splits_list[c], input_splits_list[c])
        # ---------------- c+1 prefetch permute + A2A1 Start ----------------
        _hs_n: torch.Tensor | None = None
        _unp_idx_n = None
        comm1_n = None
        a2a1_out_n = None
        if has_next:
            _hs_n, _unp_idx_n = permute(hs_n, se_n.to(torch.int32), fused=fused)
            comm1_n, a2a1_out_n = _A2AStart.apply(
                _hs_n, ep_group, output_splits_list[c + 1], input_splits_list[c + 1])
        # ---------------- wait A2A1(c) → compute2(c) → A2A2 Start(c) ----------------
        a2a1_out_c = _A2AEnd.apply(
            comm1_c, _hs, a2a1_out_c, ep_group,
            output_splits_list[c], input_splits_list[c])
        compute2_out_c = post_dispatch_pre_combine(
            a2a1_out_c, num_experts, ngt_list[c], ngs_list[c],
            fc1_weight, fc2_weight, swiglu_limit, swiglu_alpha,
            activation, ep_group, fused
        )
        comm2_c, a2a2_out_c = _A2AStart.apply(
            compute2_out_c, ep_group, input_splits_list[c], output_splits_list[c])
        # ---------------- wait A2A1(c+1) → compute2(c+1) → A2A2 Start(c+1) ----------------
        compute2_out_n = None
        comm2_n = None
        a2a2_out_n = None
        if has_next:
            assert comm1_n is not None
            assert _hs_n is not None
            a2a1_out_n = _A2AEnd.apply(
                comm1_n, _hs_n, a2a1_out_n, ep_group,
                output_splits_list[c + 1], input_splits_list[c + 1])
            compute2_out_n = post_dispatch_pre_combine(
                a2a1_out_n, num_experts, ngt_list[c+1], ngs_list[c+1],
                fc1_weight, fc2_weight, swiglu_limit, swiglu_alpha,
                activation, ep_group, fused
            )
            comm2_n, a2a2_out_n = _A2AStart.apply(
                compute2_out_n, ep_group,
                input_splits_list[c + 1], output_splits_list[c + 1])
        # ---------------- wait A2A2(c) → post_combine(c) ----------------
        a2a2_out_c = _A2AEnd.apply(
            comm2_c, compute2_out_c, a2a2_out_c, ep_group,
            input_splits_list[c], output_splits_list[c])
        chunk_out_c = unpermute(
            a2a2_out_c.to(rw_c.dtype), _unp_idx,
            probs=rw_c, fused=fused,
        )
        # ---------------- wait A2A2(c+1) → post_combine(c+1) ----------------
        chunk_out_n = None
        if has_next:
            assert comm2_n is not None
            assert compute2_out_n is not None
            assert _unp_idx_n is not None
            a2a2_out_n = _A2AEnd.apply(
                comm2_n, compute2_out_n, a2a2_out_n, ep_group,
                input_splits_list[c + 1], output_splits_list[c + 1])
            chunk_out_n = unpermute(
                a2a2_out_n.to(rw_n.dtype), _unp_idx_n,
                probs=rw_n, fused=fused,
            )
        return chunk_out_c, chunk_out_n
    return _inner


def ep_chunkmoe_forward(
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    ep_group: Optional[dist.ProcessGroup] = None,
    fused: bool = True,
    swiglu_limit: Optional[float] = None,
    swiglu_alpha: Optional[float] = None,
    ep_balance_strategy = None,
    skip_moe_pad_tokens: bool = False,
    seq_mask: Optional[torch.Tensor] = None,
    activation: Optional[Callable] = None,
    ep_plan: Optional[EPPlanConfig] = None,
) -> torch.Tensor:
    # todo：add skip_moe_pad_tokens & ep_balance_strategy
    if FORCE_EP_BALANCE:
        selected_experts = force_ep_balance(num_experts, selected_experts)
    if routing_weights.size() != selected_experts.size():
        routing_weights = routing_weights.gather(1, selected_experts)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    num_tokens = hidden_states.shape[0]
    # Chunk size and dynamic balance planning params must come from ep_plan; raise if missing.
    if ep_plan is None:
        raise ValueError("ep_pipeline_forward requires an ep_plan (EPPlanConfig) to be provided.")
    moe_chunk_size = ep_plan.moe_chunk_size
    moe_num_chunks = ep_plan.moe_num_chunks
    recompute = ep_plan.chunk_moe_recompute

    chunk_size = moe_chunk_size if moe_chunk_size > 0 else max(num_tokens, 1)
    # Pre-split inputs into chunks. Three modes (highest priority first):
    #   1. moe_num_chunks: split into a fixed number of chunks; all ranks get
    #      the same count, so no all_reduce alignment is needed.
    #   2. moe_chunk_size: split by a fixed chunk size.
    chunk_hs_list, chunk_rw_list, chunk_se_list = [], [], []

    if moe_num_chunks > 0:
        chunk_len = (num_tokens + moe_num_chunks - 1) // moe_num_chunks
        for i in range(moe_num_chunks):
            start = min(i * chunk_len, num_tokens)
            end = min(start + chunk_len, num_tokens)
            chunk_hs_list.append(hidden_states[start:end])
            chunk_rw_list.append(routing_weights[start:end])
            chunk_se_list.append(selected_experts[start:end])
        num_chunks = moe_num_chunks
    else:
        num_chunks = (num_tokens + chunk_size - 1) // chunk_size
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            chunk_hs_list.append(hidden_states[start:end])
            chunk_rw_list.append(routing_weights[start:end])
            chunk_se_list.append(selected_experts[start:end])

    if moe_num_chunks <= 0 and ep_group is not None and dist.get_world_size(ep_group) > 1:
        nmax = torch.tensor([num_chunks], dtype=torch.int64, device=hidden_states.device)
        dist.all_reduce(nmax, op=dist.ReduceOp.MAX, group=ep_group)
        num_chunks = int(nmax.item())
    for _ in range(num_chunks - len(chunk_se_list)):
        chunk_hs_list.append(hidden_states[:0])
        chunk_rw_list.append(routing_weights[:0])
        chunk_se_list.append(selected_experts[:0])
    input_splits_list, output_splits_list, ngt_list, ngs_list = \
        batched_pre_dispatch_metadata(chunk_se_list, num_experts, ep_group)
    chunk_outputs = [None] * num_chunks

    pair_closure = make_forward_pair_closure(
        num_chunks=num_chunks,
        output_splits_list=output_splits_list,
        input_splits_list=input_splits_list,
        ngt_list=ngt_list,
        ngs_list=ngs_list,
        fc1_weight=fc1_weight,
        fc2_weight=fc2_weight,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        activation=activation,
        ep_group=ep_group,
        fused=fused,
        num_experts=num_experts,
    )

    for c in range(0, num_chunks, 2):
        has_next = (c + 1) < num_chunks
        n = c + 1 if has_next else c
        if recompute:
            out_c, out_n = checkpoint(
                pair_closure,
                chunk_hs_list[c], chunk_se_list[c], chunk_rw_list[c],
                chunk_hs_list[n], chunk_se_list[n], chunk_rw_list[n],
                has_next, c,
                use_reentrant=False,
            )
        else:
            out_c, out_n = pair_closure(
                chunk_hs_list[c], chunk_se_list[c], chunk_rw_list[c],
                chunk_hs_list[n], chunk_se_list[n], chunk_rw_list[n],
                has_next, c,
            )
        chunk_outputs[c] = out_c
        if has_next:
            chunk_outputs[c + 1] = out_n
    hidden_states = torch.cat(chunk_outputs, dim=0)
    return hidden_states
