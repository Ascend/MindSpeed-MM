# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""Conv1d CP causal-boundary tail exchange (collective, recompute-safe).

The pipeline send-recv state-passing primitives (_RecvState/_SendState) that used
to live here were REMOVED (2026-08-19): the scan CP mode (parallel_scan.py) won on
throughput (pipeline bubble (kv-1)/(kv-1+L) x2 vs <5% compose overhead) and pack
multi-segment correctness. Full removal record incl. recovery commit:
gdn_pipeline_cp_removal.md.

What remains is _ConvTailExchange, used by BOTH the (removed) pipeline mode and
the current scan mode: each rank's first W-1 conv1d outputs need the previous
rank's last W-1 tokens. Uses all_gather (NOT P2P): collectives are recompute-safe
because all ranks recompute the same checkpointed layer simultaneously, whereas
P2P send/recv cross-wires with real-backward communication under gradient
checkpointing and deadlocks. See the class docstring for the autograd design.
"""
import torch
import torch.distributed as dist

class _ConvTailExchange(torch.autograd.Function):
    """Conv1d CP causal-boundary exchange via all_gather (collective, recompute-safe).

    Replaces the old P2P _RecvTail + _SendTail pair. One all_gather in forward, one
    in backward. Merges recv+send because all_gather is a collective: every rank must
    contribute its tail in the SAME op that produces prev rank's tail.

    forward:  all_gather tail_to_send -> gathered[rank-1] (zeros if is_first).
    backward: all_gather grad_recv_tail -> gathered_grad[rank+1] (zeros if is_last).
              grad_recv_tail[r] is the grad for what rank r received (from r-1), i.e.
              the grad for rank r-1's tail. So this rank's tail-grad = gathered_grad[rank+1].

    Unlike P2P (dist.send/recv, which cross-wires with backward under recompute and
    deadlocks), all_gather is a collective: all ranks recompute the same checkpointed
    layer simultaneously, so the recompute-forward all_gather matches on every rank —
    no cross-wire with real-backward all_gather. Recompute-safe by construction.
    """

    @staticmethod
    def forward(ctx, tail_to_send, group, is_first, is_last):
        ctx.group = group
        ctx.is_first = is_first
        ctx.is_last = is_last
        ctx.shape = tuple(tail_to_send.shape)
        ctx.dtype = tail_to_send.dtype
        ctx.device = tail_to_send.device
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        gathered = torch.empty((world_size,) + tuple(tail_to_send.shape),
                               dtype=tail_to_send.dtype, device=tail_to_send.device)
        # tail_to_send is .contiguous() at the call site; guard anyway.
        dist.all_gather_into_tensor(gathered, tail_to_send.contiguous(), group=group)
        if is_first:
            recv_tail = torch.zeros_like(tail_to_send)
        else:
            recv_tail = gathered[rank - 1].clone()   # clone: gathered buffer is transient
        return recv_tail

    @staticmethod
    def backward(ctx, grad_recv_tail):
        rank = dist.get_rank(ctx.group)
        world_size = dist.get_world_size(ctx.group)
        gathered_grad = torch.empty((world_size,) + tuple(grad_recv_tail.shape),
                                    dtype=grad_recv_tail.dtype, device=grad_recv_tail.device)
        # grad_recv_tail is a slice of torch.cat's grad -> non-contiguous; .contiguous() required.
        dist.all_gather_into_tensor(gathered_grad, grad_recv_tail.contiguous(), group=ctx.group)
        if ctx.is_last:
            grad_tail = torch.zeros(ctx.shape, dtype=ctx.dtype, device=ctx.device)
        else:
            grad_tail = gathered_grad[rank + 1].clone()
        return (grad_tail, None, None, None)


def conv_tail_exchange(tail_to_send, group, is_first, is_last):
    """Conv1d CP causal-boundary tail exchange via all_gather (replaces recv_tail+send_tail)."""
    return _ConvTailExchange.apply(tail_to_send, group, is_first, is_last)
