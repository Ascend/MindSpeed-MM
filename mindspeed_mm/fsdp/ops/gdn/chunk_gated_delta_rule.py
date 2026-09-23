# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings
from typing import Optional

import torch

from mindspeed_mm.fsdp.utils.import_utils import IS_TRITON_AVAILABLE
from mindspeed_mm.fsdp.train.training_context import TrainingContext, TrainingStage
from mindspeed_mm.fsdp.features.memory.async_offload import OffloadManager, SwapTensor
from mindspeed_mm.fsdp.ops.gdn.parallel_scan import CpInitStateStore
from mindspeed_mm.fsdp.utils.device import get_current_stream

if IS_TRITON_AVAILABLE:
    from .triton.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
    from .triton.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
    from .triton.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
    from .triton.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
    from .triton.solve_tril import solve_tril
    from .triton.cumsum import chunk_local_cumsum
    from .triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
else:
    def _identity_decorator(fn):
        return fn
    input_guard = _identity_decorator
    autocast_custom_fwd = _identity_decorator
    autocast_custom_bwd = _identity_decorator


def chunk_gated_delta_rule_fwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        cp_group=None,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_pre_num_ranks=None,
        cp_is_first_rank=None,
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens, head_first=False)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    # CP pre_process: compute [A_p, B_p] = [M, S_ext] and all_gather + merge for init_state
    if cp_size > 1:
        from mindspeed_mm.fsdp.ops.gdn.parallel_scan import compute_a_p_b_p, prefix_scan_compose
        _B, _T, _H, _K = k.shape
        _V = u.shape[-1]
        # The last rank's [A_p, B_p] is never consumed by anyone (no rank j > last reads it),
        # so computing it is wasted work. Send zeros in the all_gather instead.
        if cp_rank < cp_size - 1:
            # compute_a_p_b_p returns FLAT [1,H,K,K]/[1,H,K,V] (varlen) or [B,H,K,K]/[B,H,K,V]
            # (fixed-length). The flat varlen shape keeps all_gather consistent across ranks
            # (neat_packing gives each rank a different local segment count -> [N,...] would
            # mismatch). _hm is [1,H,K,K+V] (varlen) or [_B,H,K,K+V] (fixed).
            A_p, B_p = compute_a_p_b_p(k, w, u, g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
            _hm = torch.cat([A_p, B_p], dim=-1).contiguous()  # [1|B,H,K,K+V]
        else:
            _hm = torch.zeros(_B, _H, _K, _K + _V, device=k.device, dtype=torch.float32)
        # all_gather (no grad — inside Function.forward). Use all_gather_into_tensor (single
        # contiguous buffer) instead of list+stack — saves cp_size allocations + 1 stack memcpy.
        # Verified available on NPU/HCCL (ep_dispatcher.py uses the same op).
        all_hm_stacked = torch.empty((cp_size,) + _hm.shape, dtype=_hm.dtype, device=_hm.device)
        torch.distributed.all_gather_into_tensor(all_hm_stacked, _hm, group=cp_group)
        all_A_p = all_hm_stacked[..., :_K]
        all_B_p = all_hm_stacked[..., _K:]
        # merge: S = 0; for j < rank: S = M_j @ S + S_ext_j (only compute this rank's state)
        _merged = prefix_scan_compose(all_A_p, all_B_p, cp_rank,
                                      pre_num_ranks=cp_pre_num_ranks, is_first_rank=cp_is_first_rank)
        # Expand the flat merged state [1,H,K,V] into [N_local,H,K,V] at slot [0]: the
        # cross-rank incoming state seeds ONLY the first segment (the continuation from the
        # previous rank); all other segments start fresh (init=0). For fixed-length
        # (cu_seqlens is None) _merged is already [B,H,K,V] with no N dim to expand — pass
        # through as-is.
        if cu_seqlens is not None and _merged is not None:
            _N_local = len(cu_seqlens) - 1
            if _N_local > 1:
                _init_full = torch.zeros(_N_local, _H, _K, _V, device=k.device, dtype=_merged.dtype)
                _init_full[0] = _merged.squeeze(0)
                initial_state = _init_full
            else:
                initial_state = _merged  # N_local==1: [1,H,K,V], slot0==slot-1, no expand needed
        else:
            initial_state = _merged

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return g, o, A, final_state, initial_state


def chunk_gated_delta_rule_bwd(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        do: torch.Tensor,
        dht: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        cp_group=None,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_post_num_ranks=None,
        cp_is_last_rank=None,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    # CP backward pre_process: compute [dA_p, dB_p] = [dM, dS_ext] and all_gather + merge_bwd for dht
    if cp_size > 1:
        # An external dht (the autograd gradient of the returned final_state, i.e.
        # output_final_state=True) is NOT yet supported under CP: the cross-rank
        # merge below replaces dht wholesale, so the external contribution would be
        # silently dropped and dq/dk/dv/db/dg would all be wrong. The modeling never
        # combines CP with output_final_state (dht is None there); refuse loudly
        # instead of computing wrong silently. (PR review, htwang.)
        if dht is not None:
            raise NotImplementedError(
                "chunk_gated_delta_rule backward under CP (cp_size > 1) does not "
                "support an external dht (output_final_state=True): the cross-rank "
                "merge overwrites it. Do not combine output_final_state with "
                "kvallgather scan CP."
            )
        from mindspeed_mm.fsdp.ops.gdn.parallel_scan import compute_da_p_db_p, prefix_scan_compose_bwd
        _B, _T, _H, _K = k.shape
        _V = do.shape[-1]
        # The first rank's [dA_p, dB_p] is never consumed (no rank j < first reads it), so
        # computing it is wasted work. Send zeros in the all_gather instead.
        if cp_rank > 0:
            dA_p, dB_p = compute_da_p_db_p(q, k, w, do, dv, g, scale, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
            _dhm = torch.cat([dA_p, dB_p], dim=-1).contiguous()
        else:
            _dhm = torch.zeros(_B, _H, _K, _K + _V, device=k.device, dtype=torch.float32)
        all_dhm_stacked = torch.empty((cp_size,) + _dhm.shape, dtype=_dhm.dtype, device=_dhm.device)
        torch.distributed.all_gather_into_tensor(all_dhm_stacked, _dhm, group=cp_group)
        all_dA_p = all_dhm_stacked[..., :_K]
        all_dB_p = all_dhm_stacked[..., _K:]
        # backward merge: dht = 0; for j > rank (descending): dht = dM_j @ dht + dS_ext_j
        _merged_dht = prefix_scan_compose_bwd(all_dA_p, all_dB_p, cp_rank, cp_size,
                                       post_num_ranks=cp_post_num_ranks, is_last_rank=cp_is_last_rank)
        # Mirror of the fwd expand: the flat merged dht [1,H,K,V] lands at slot [-1] (the
        # spanning/outbound segment whose gradient flows back to the previous rank); all
        # other segments get zero inbound gradient. For fixed-length (cu_seqlens is None)
        # _merged_dht is already [B,H,K,V] — pass through.
        if cu_seqlens is not None and _merged_dht is not None:
            _N_local = len(cu_seqlens) - 1
            if _N_local > 1:
                _dht_full = torch.zeros(_N_local, _H, _K, _V, device=k.device, dtype=_merged_dht.dtype)
                _dht_full[-1] = _merged_dht.squeeze(0)
                dht = _dht_full
            else:
                dht = _merged_dht
        else:
            dht = _merged_dht
        # Free CP bwd pre_process intermediates — consumed by prefix_scan_compose_bwd above.
        del all_dhm_stacked, all_dA_p, all_dB_p
        # dv is the LOCAL dv from chunk_bwd_dv_local. compute_da_p_db_p does NOT modify it
        # (reads dv_c via reshape view only to form db_t). The cross-rank dv contribution
        # is embedded in dB_p (dS_ext) → dht, propagated by bwd_dhu.

    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state if cp_size <= 1 else None,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    # dht consumed by bwd_dhu above; free before per-token gradient kernels.
    del dht

    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        chunk_size=chunk_size,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    # h/v_new/w consumed by chunk_bwd_dqkwg above; free the headroom.
    del h, v_new, w
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size
    )
    dk.add_(dk2)
    dg.add_(dg2)
    if dg.dtype != torch.float32:
        raise ValueError(
            f"dg current type is {dg.dtype} , should be float32"
        )
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens, head_first=False)
    # varlen padding zero-fill: GDN triton kernels write only real-token positions
    # (chunk_indices + eos-bos masking), leaving torch.empty_like garbage at
    # [num_real:local_T]. Two consumers read that tail: conv1d backward (its W-1-wide
    # window past the real-token end overlaps REAL tokens under the SHIFTED cu_seqlens
    # used in CP mode, _conv1d_cu_seqlens in the modeling files) and the A_log/dt_bias
    # grads (d(A_log)=Σ dg·g over ALL positions). Zero the returned grads at padding
    # to break both corruption chains (0×anything=0). Fixed-len (cu_seqlens=None):
    # kernel writes all positions, block skipped.
    if cu_seqlens is not None:
        _nr = cu_seqlens[-1].item()
        _Tl = q.shape[1]
        if _nr < _Tl:
            dq[:, _nr:_Tl].zero_()
            dk[:, _nr:_Tl].zero_()
            dv[:, _nr:_Tl].zero_()
            db[:, _nr:_Tl].zero_()
            dg[:, _nr:_Tl].zero_()
    return dq, dk, dv, db, dg, dh0


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
            ctx,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            scale: float,
            initial_state: torch.Tensor,
            output_final_state: bool,
            cu_seqlens: Optional[torch.LongTensor] = None,
            use_qk_l2norm_in_kernel: bool = False,
            chunk_size: int = 64,
            skip_recompute: bool = False,
            cp_group=None,
            cp_rank: int = 0,
            cp_size: int = 1,
            cp_pre_num_ranks=None,
            cp_is_first_rank=None,
            cp_post_num_ranks=None,
            cp_is_last_rank=None,
    ):
        q_rstd, k_rstd = None, None

        training_stage = TrainingContext().get_training_stage()
        layer_idx, depth = TrainingContext().get_layer_index(), TrainingContext().get_model_depth()
        h2d_stream = OffloadManager().swap_stream
        d2h_stream = OffloadManager().swap_stream

        if skip_recompute and training_stage == TrainingStage.BACKWARD:
            if layer_idx == depth - 1:
                if output_final_state:
                    final_state = OffloadManager().pop_npu_tensor().tensor
                else:
                    final_state = None
                A = OffloadManager().pop_npu_tensor().tensor
                o = OffloadManager().pop_npu_tensor().tensor
                g = OffloadManager().pop_npu_tensor().tensor

            else:
                layer_items_keys = OffloadManager().get_layer_items_keys(layer_idx)
                swap_tensor_nums = 4 if output_final_state else 3
                swap_tensors = []
                for swap_key in reversed(layer_items_keys[-swap_tensor_nums:]):
                    swap_tensor = OffloadManager().get(swap_key)
                    swap_tensor.launch_h2d(h2d_stream)
                    get_current_stream().wait_event(swap_tensor.h2d_event)
                    swap_tensors.append(swap_tensor.tensor)
                    OffloadManager().clear(swap_key)
                if output_final_state:
                    final_state, A, o, g = swap_tensors
                else:
                    A, o, g = swap_tensors
                    final_state = None
            # CP scan: restore the merged initial_state from its dedicated per-layer slot
            # (see the stash site below for why the backward needs the exact merged value).
            if cp_size > 1:
                initial_state = CpInitStateStore().pop_cp_init_state(layer_idx)
                # None is LEGITIMATE for the stream-first rank (cp_is_first_rank=True:
                # no ranks precede it, the merged state is zero by definition and fwd
                # stores it as None). For every OTHER rank None means the stash/pop
                # pairing broke (chunk_mbs overwrite, inconsistent skip_recompute...):
                # the fwd_h recompute would silently start from 0 and mis-compute
                # dq/dk/dw/dg for the first chunks -- refuse loudly instead.
                if initial_state is None and not cp_is_first_rank:
                    raise RuntimeError(
                        f"CpInitStateStore has no stashed initial_state for layer {layer_idx}: "
                        "CP backward requires the merged state stashed in the forward pass."
                    )
        else:
            g, o, A, final_state, initial_state = chunk_gated_delta_rule_fwd(
                q=q, k=k, v=v, g=g, beta=beta, scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens, chunk_size=chunk_size,
                cp_group=cp_group, cp_rank=cp_rank, cp_size=cp_size,
                cp_pre_num_ranks=cp_pre_num_ranks,
                cp_is_first_rank=cp_is_first_rank,
            )
            # `initial_state` is now the merged (prefix-composed) state computed inside fwd
            # (None for cp_size<=1). The backward's fwd_h recompute needs this EXACT value —
            # see the stash site below for the failure mode.

        if skip_recompute and training_stage == TrainingStage.FORWARD:
            swap_tensors = [g, o, A]
            if output_final_state:
                swap_tensors.append(final_state)
            for swap_tensor in swap_tensors:
                key, after_block = OffloadManager().get_cnt(layer_idx)
                if after_block:
                    OffloadManager().del_npu_tensor("{}_".format(layer_idx - 1))
                if layer_idx == depth - 1:
                    OffloadManager().put_npu_tensor(SwapTensor(swap_tensor, key))
                else:
                    swap_tensor = SwapTensor(swap_tensor, key)
                    swap_tensor.launch_d2h(d2h_stream)
                    OffloadManager().put(key, swap_tensor)

            # CP scan: stash the merged initial_state in a DEDICATED per-layer slot (NOT the
            # npu_item LIFO stack, which is shared with g/o/A/final_state and is order-sensitive
            # — putting init_state on it would desync the pop order on the last-layer path).
            # It is ~2MB fp32 [B,H,K,V], kept on NPU — far cheaper than re-running the CP
            # forward pre_process (incl. the cross-rank all_gather). The backward's fwd_h
            # recompute needs this EXACT value: without it cp_rank>0 recomputes h from 0 and
            # dq/dk/dw/dg are wrong for the first ~1-2 chunks. cp_size<=1: no merged state.
            if cp_size > 1:
                CpInitStateStore().put_cp_init_state(layer_idx, initial_state)

        ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens)
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.cp_pre_num_ranks = cp_pre_num_ranks
        ctx.cp_is_first_rank = cp_is_first_rank
        ctx.cp_post_num_ranks = cp_post_num_ranks
        ctx.cp_is_last_rank = cp_is_last_rank
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        q, q_rstd, k, k_rstd, v, g, beta, A, initial_state, cu_seqlens = ctx.saved_tensors

        dq, dk, dv, db, dg, dh0 = chunk_gated_delta_rule_bwd(
            q=q, k=k, v=v, g=g, beta=beta, A=A,
            scale=ctx.scale, initial_state=initial_state,
            do=do, dht=dht,
            cu_seqlens=cu_seqlens, chunk_size=ctx.chunk_size,
            cp_group=ctx.cp_group, cp_rank=ctx.cp_rank, cp_size=ctx.cp_size,
            cp_post_num_ranks=ctx.cp_post_num_ranks,
            cp_is_last_rank=ctx.cp_is_last_rank,
        )

        # 19 inputs: q,k,v,g,beta,scale,initial_state,output_final_state,cu_seqlens,
        # use_qk_l2norm_in_kernel,chunk_size,skip_recompute,cp_group,cp_rank,cp_size,
        # cp_pre_num_ranks,cp_is_first_rank,cp_post_num_ranks,cp_is_last_rank
        return (dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta),
                None, dh0, None, None, None, None, None, None, None, None,
                None, None, None, None)


@torch.compiler.disable
def chunk_gated_delta_rule(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens: Optional[torch.LongTensor] = None,
        chunk_size: int = 64,
        head_first: bool = False,
        skip_recompute: bool = False,
        cp_group=None,
        cp_rank: int = 0,
        cp_size: int = 1,
        cp_pre_num_ranks=None,
        cp_is_first_rank=None,
        cp_post_num_ranks=None,
        cp_is_last_rank=None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
            This argument has been deprecated.
        skip_recompute (bool):
            Whether skip recomupte and async offload the outputs to cpu.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise ValueError(
            f"q current type is {q.dtype} , k current type is {k.dtype} ,v current type is {v.dtype} , they should are equal"
        )
    if q.dtype == torch.float32:
        raise ValueError(
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
    if len(beta.shape) != 3:
        raise ValueError(
            f"beta current shape len is {len(beta.shape)}, beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

    if head_first:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
    if not head_first and q.shape[1] < q.shape[2]:
        warnings.warn(
            f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
            "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
            "when head_first=False was specified. "
            "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        """This function is intended to align with the l2norm implementation in the FLA library."""
        original_dtype = x.dtype
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        # Counteract verl's autocast promotion (bf16 -> fp32) by restoring original dtype
        return (x * inv_norm).to(original_dtype)

    if use_qk_l2norm_in_kernel:
        q = l2norm(q, dim=-1, eps=1e-6)
        k = l2norm(k, dim=-1, eps=1e-6)

    if cp_size > 1:
        # CP scan mode: all_gather + merge happen inside Function.forward/backward (no_grad),
        # so no autograd graph for all_gather. Gradient correctness via symmetric design.
        initial_state = None  # CP computes init_state internally

    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        False,
        chunk_size,
        skip_recompute,
        cp_group,
        cp_rank,
        cp_size,
        cp_pre_num_ranks,
        cp_is_first_rank,
        cp_post_num_ranks,
        cp_is_last_rank,
    )
    return o, final_state
