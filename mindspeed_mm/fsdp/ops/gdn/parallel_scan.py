"""Parallel scan for GDN context parallelism (no cross-rank pipeline bubble).

Computes per-chunk transition matrix A_p [N,H,K,K] and accumulated B_p [N,H,K,V]
in Python (autograd-aware), then AllGather + prefix-scan across kvallgather ranks
to get each rank's correct initial_state. The cross-rank compose has an optional
triton kernel (triton/merge_scan.py); the GDN fwd/bwd kernels are unchanged.

Math (from the Triton kernel chunk_gated_delta_rule_fwd_kernel_h_blockdim64):
  Per sub-chunk t (BT=64 tokens):
    gate_t = exp(g_last_t - g_chunk_t)          [BT]
    kg_t = k_t * gate_t[None, :]                [K, BT]
    A_t = exp(g_last_t) * I_K - kg_t @ w_t      [K, K]
    B_t = kg_t @ u_t                             [K, V]
  A_p = A_{NT-1} @ ... @ A_0                     [K, K]
  B_p = accumulated (compose)                     [K, V]  (= final_state with init=0)

Cross-rank prefix-scan:
  compose((A2,B2), (A1,B1)) = (A2@A1, A2@B1 + B2)
  rank j's initial_state = B_prefix[j-1]  (rank 0 = 0)
"""
import torch
import torch.distributed as dist

from mindspeed_mm.fsdp.features.memory.async_offload import OffloadManager


def build_cp_context(cu_seqlens, group, conv1d_kernel_size=None, rank_start=None, rank_end=None,
                     rank_boundaries=None):
    """Split global cu_seqlens into rank-local for CP scan.

    Returns dict with: local_cu_seqlens (GPU int32), pre_num_ranks, post_num_ranks,
    is_first_rank, is_last_rank, pre_num_conv_tokens.

    rank_start/rank_end (optional): the EXACT [rank_start, rank_end) token range this
    kv_rank holds after inputs_embeds split (cal_split_sizes + ulysses aggregation).
    Must be passed when total_tokens % kv_size != 0 — the legacy fallback
    `part_len = total // world_size` floor-divides and drops the remainder, so the
    rank that got the +1 remainder token ends up with local_cu_seqlens covering
    [0, part_len) while its tensor has part_len+1 tokens -> fwd_h/chunk_o write
    uninitialized garbage at the tail position -> NaN from iter 1.

    rank_boundaries (optional): the KV-group (kvallgather group) rank boundaries as a
    1-D list/tensor [0, b1, ..., total], len = world_size+1. Must be the KV-group boundaries
    (each kv_rank's token range), NOT the cp_group per-cp_rank boundaries — the caller
    aggregates ulysses_size cp chunks per kv_rank. When provided, the else-branch rank-of-position
    lookup uses searchsorted on these EXACT (ceil-split) boundaries instead of the legacy
    floor `pos // part_len`, which is only correct when total % world_size == 0. Without
    rank_boundaries, total%ws!=0 + non-full-span silently miscomputes pre/post_num_ranks.
    """
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)
    cu_seqlens_cpu = cu_seqlens.cpu().to(dtype=torch.long)
    # Flatten batch dim if cu_seqlens is 2-D [B, N+1] (data pipeline's HF tokenizer.pad stacks
    # per-sample lists). We must produce a 1-D cumulative tensor [0, s1, s1+s2, ..., total]
    # that PRESERVES all intra-row segment boundaries — this is what the GDN kernels use as
    # a state-reset descriptor (each segment resets the recurrent state). The old flatten
    # took only each row's first/last element (per_sample_lens = [:, -1] - [:, 0]),
    # collapsing a multi-segment row like [0, 400, 800, 1M] into [0, 1M] — silently dropping
    # the 400/800 boundaries so the scan never reset state between packed samples
    # -> cross-sample state contamination.
    if cu_seqlens_cpu.ndim == 2:
        rows = cu_seqlens_cpu.unbind(0)
        if len(rows) == 1:
            # mbs=1: the single row is already the 1-D multi-segment cu_seqlens [0, s1, ..., total].
            cu_seqlens_cpu = rows[0].clone()
        else:
            # mbs>1: concatenate per-row segments with a running global offset, dropping each
            # subsequent row's leading 0 to avoid duplicate boundaries.
            # KNOWN LIMITATION: this treats multiple batch rows as one concatenated stream,
            # which only matches a CP split that acts on that stream. The USP path splits
            # per-row (dim=1), so mbs>1 is NOT actually supported under scan CP — callers
            # should keep micro_batch_size=1. We assert to fail loud rather than silently
            # produce wrong local_cu_seqlens.
            raise NotImplementedError("varlen scan CP with micro_batch_size>1 is not supported "
                                      "(USP splits per-row dim=1, incompatible with a single "
                                      "concatenated stream). Set micro_batch_size=1.")
    # Defensive: cu_seqlens must be strictly increasing (unique_consecutive below relies on it).
    assert cu_seqlens_cpu.ndim == 1 and cu_seqlens_cpu.numel() >= 2, \
        f"cu_seqlens must be 1-D with >=2 entries, got shape {cu_seqlens_cpu.shape}"
    assert (cu_seqlens_cpu[1:] > cu_seqlens_cpu[:-1]).all(), \
        f"cu_seqlens must be strictly increasing, got {cu_seqlens_cpu.tolist()}"
    total_tokens = cu_seqlens_cpu[-1].item()
    if world_size == 1:
        # scan CP not active (kv_size=1): no-op, return as-is.
        local_cu_seqlens = cu_seqlens_cpu.to(torch.int32).to(device=cu_seqlens.device, non_blocking=True)
        return {
            'local_cu_seqlens': local_cu_seqlens,
            'pre_num_ranks': 0, 'post_num_ranks': 0,
            'is_first_rank': True, 'is_last_rank': True,
            'pre_num_conv_tokens': 0, 'conv1d_kernel_size': conv1d_kernel_size,
        }
    # rank_start/rank_end: caller-provided exact range (matches inputs_embeds split),
    # else legacy floor-division fallback (drops remainder — only correct when total%world_size==0).
    if rank_start is None or rank_end is None:
        part_len = total_tokens // world_size
        rank_start = part_len * rank
        rank_end = rank_start + part_len
    else:
        part_len = total_tokens // world_size  # approximate, for metadata rank-of-position lookup only

    # Find sequences overlapping this rank's [rank_start, rank_end)
    start_seq_idx = torch.searchsorted(cu_seqlens_cpu[1:], rank_start, side='right')
    end_seq_idx = torch.searchsorted(cu_seqlens_cpu[:-1], rank_end, side='left')
    subset = cu_seqlens_cpu[start_seq_idx: end_seq_idx + 1]
    local_cu_seqlens_cpu = (subset.clamp(min=rank_start, max=rank_end) - rank_start).unique_consecutive().to(torch.int32)
    local_cu_seqlens = local_cu_seqlens_cpu.to(device=cu_seqlens.device, non_blocking=True)

    # Rank metadata.
    # part_len = total//world_size is approximate when total (cu_seqlens real-token count,
    # excludes pad_to_multiple_of padding) differs from the inputs_embeds length the split used.
    # For a "full-span" pack — the single packed sequence spans [0, total_tokens] across every
    # rank, exactly like fixed-length — pre/post reduce to rank / world_size-1-rank. This covers
    # both the true single-sequence case (numel==2, [0, total]) AND multi-segment neat_packing
    # where the whole bin spans [0, total] (the common pack case). The pos//part_len fallback
    # below is only for the rare non-full-span case (segment boundaries mid-rank with the first
    # segment not starting at 0 / last not ending at total) and inherits the part_len caveat.
    first_seq_global_start = cu_seqlens_cpu[start_seq_idx].item()
    last_seq_global_end = cu_seqlens_cpu[end_seq_idx].item()
    pre_num_conv_tokens = max(0, rank_start - first_seq_global_start)
    full_span = (first_seq_global_start == 0 and last_seq_global_end == total_tokens)
    if cu_seqlens_cpu.numel() == 2 or full_span:
        # single sequence OR a full-span pack spanning [0, total] -> fixed-length semantics
        pre_num_ranks = rank
        post_num_ranks = world_size - 1 - rank
        is_first_rank = (rank == 0)
        is_last_rank = (rank == world_size - 1)
    else:
        # Non-full-span neat_packing: the current rank's first overlapping sequence does not
        # start at 0, or the last does not end at total. Map sequence start/end global
        # positions to rank indices. Floor `pos // part_len` (part_len=total//world_size)
        # is only correct when total % world_size == 0; under the ceil split the caller
        # actually uses (cal_split_sizes), total%ws!=0 makes floor land in a different
        # rank than the true boundary -> pre/post_num_ranks/is_first/is_last wrong ->
        # compose chains the wrong number of ranks -> cross-sample state contamination.
        # Use searchsorted on the caller-provided EXACT kv-group boundaries (ceil split).
        if rank_boundaries is not None:
            rb = torch.as_tensor(rank_boundaries, dtype=torch.long,
                                 device=cu_seqlens_cpu.device)
            assert rb.numel() == world_size + 1, \
                f"rank_boundaries must have world_size+1={world_size+1} entries, got {rb.numel()}"
            assert rb[0].item() == 0, \
                f"rank_boundaries[0] must be 0, got {rb[0].item()}"
            # rb comes from the caller's split_sizes, which are based on total_seq_len =
            # inputs_embeds.shape[1] — the PADDED length (pad_to_multiple_of rounds 131045 up to
            # 131056). cu_seqlens/total_tokens are the UNPADDED real-token count (131045). The
            # padding is appended only at the very end, so all rank boundaries agree except the
            # last (rb[-1] = padded total > total_tokens). Clamp the last entry to the unpadded
            # total so searchsorted over sequence positions (which never exceed total_tokens-1)
            # lands in the correct rank.
            if rb[-1].item() != total_tokens:
                assert rb[-1].item() >= total_tokens, \
                    f"rank_boundaries[-1]={rb[-1].item()} < total_tokens={total_tokens} (truncation?)"
                rb = rb.clone()
                rb[-1] = total_tokens
            # rank k holds [rb[k], rb[k+1]); a global position pos's owning rank =
            # searchsorted(rb, pos, 'right') - 1 (the last boundary <= pos).
            first_rank_of_first_seq = int((torch.searchsorted(rb, first_seq_global_start, side='right') - 1).item())
            last_rank_of_last_seq = int((torch.searchsorted(rb, last_seq_global_end - 1, side='right') - 1).item())
        else:
            # Legacy floor fallback — ONLY correct when total % world_size == 0.
            first_rank_of_first_seq = first_seq_global_start // part_len
            last_rank_of_last_seq = (last_seq_global_end - 1) // part_len
        pre_num_ranks = rank - first_rank_of_first_seq
        is_first_rank = (rank == first_rank_of_first_seq)
        post_num_ranks = last_rank_of_last_seq - rank
        is_last_rank = (rank == last_rank_of_last_seq)

    return {
        'local_cu_seqlens': local_cu_seqlens,
        'pre_num_ranks': pre_num_ranks,
        'post_num_ranks': post_num_ranks,
        'is_first_rank': is_first_rank,
        'is_last_rank': is_last_rank,
        'pre_num_conv_tokens': pre_num_conv_tokens,
        'conv1d_kernel_size': conv1d_kernel_size,
    }


def compute_a_p_b_p(k, w, v, g, chunk_size=64, cu_seqlens=None):
    """Vectorized A_p [B,H,K,K] and B_p [B,H,K,V] (fp32, autograd-aware).

    Batch-computes per-chunk (At, Bt) then sequentially composes the prefix
    (A_p = A_{NT-1}@...@A_0,  B_p = A_{NT-1}@...@A_1@B_0 + ... + B_{NT-1})
    with an NT-step loop over pre-sliced chunks (one baddbmm + one matmul per
    step; replaced an earlier tree-reduce for numerical stability, see below).

    Inputs (head_first=False, before fwd_h permute):
      k: [B,T,H,K], w: [B,T,H,K], v: [B,T,H,V], g: [B,H,T] (or [B,T,H])
    Returns A_p [B,H,K,K], B_p [B,H,K,V] (fp32).
    Assumes cu_seqlens is None and T is a multiple of BT (scan path, sequence-parallel).

    Bt uses the WY-decomposed u (= A_wy @ (beta*v)), matching fwd_h's chunk-level
    recurrence h_{t+1} = A_t@h_t + B_t with B_t = kg^T @ u. fwd_h is called with v=u
    (recompute_w_u_fwd in chunk_gated_delta_rule.py), NOT raw v — raw v diverges from
    fwd_h for kv>1 (kv=1 unaffected: _init_state=None, B_p unused for forward).
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    BT = chunk_size

    # --- varlen (pack/cu_seqlens) path: only process the LAST local sequence ---
    # In CP scan, only the sequence spanning to the NEXT rank needs A_p/B_p for cross-rank
    # merge. Other sequences have init_state=0 (handled by fwd_h kernel internally).
    #
    # Returns FLAT [1,H,K,K]/[1,H,K,V] (no N dim) — NOT [N_local,...]. Two reasons:
    #   1) all_gather requires equal N_local across the kv group; neat_packing/multi-segment
    #      packs give each rank a different segment count -> all_gather shape mismatch.
    #      Flattening to [1,...] makes every rank's shape identical.
    #   2) slot semantics: the merged cross-rank state must land in slot [0] (the
    #      continuation/inbound segment), NOT [-1]. The caller (chunk_gated_delta_rule.py)
    #      expands the flat [1,...] into [N_local,...] at the [0] slot, leaving [1..N-1]=0.
    if cu_seqlens is not None:
        # Extract the last local sequence segment (the one spanning to the NEXT rank)
        last_start = cu_seqlens[-2].item()
        last_end = cu_seqlens[-1].item()
        last_T = last_end - last_start
        if last_T == 0:
            # Empty last segment (rank boundary exactly at sequence boundary) — no transition
            A_p_out = torch.zeros(1, H, K, K, device=k.device, dtype=torch.float32)
            B_p_out = torch.zeros(1, H, K, V, device=k.device, dtype=torch.float32)
            return A_p_out, B_p_out
        # Slice k/w/v/g for the last segment (B=1 for varlen, so k is [1, T, H, K])
        k_last = k[:, last_start:last_end, :, :]      # [1, last_T, H, K]
        w_last = w[:, last_start:last_end, :, :]
        v_last = v[:, last_start:last_end, :, :]
        # g layout normalization must happen BEFORE slicing. The caller contract is
        # [B,T,H] (chunk_local_cumsum(head_first=False)). An UNAMBIGUOUS [B,H,T]
        # input is still accepted (shape[1] != T excludes [B,T,H]); the naive
        # shape[-1]==T test alone misfires when H == T -- a CORRECT [B,T,H] gets
        # permuted, then the dim=1 slice cuts the HEAD axis and A_p/B_p are poisoned
        # with exp() of cross-head g differences (NaN). See PR review and the
        # H==T_total regression case in tests/ut_fsdp/ops/gdn/test_parallel_scan.py.
        if g.dim() == 3 and g.shape[1] != T and g.shape[-1] == T:
            g = g.permute(0, 2, 1).contiguous()       # [B,H,T] -> [B,T,H]
        g_last = g[:, last_start:last_end, :]         # [B, last_T, H] (time-dim slice)
        # Recursively compute A_p/B_p for this single segment (cu_seqlens=None, fixed-length)
        A_p_seg, B_p_seg = compute_a_p_b_p(k_last, w_last, v_last, g_last, chunk_size=chunk_size, cu_seqlens=None)
        # A_p_seg is [1, H, K, K], B_p_seg is [1, H, K, V] — already the flat shape we want.
        # The merged cross-rank state will land in slot [0] after prefix_scan_compose;
        # the caller expands to [N_local,...] at [0], leaving [1..N-1]=0 (reset).
        return A_p_seg, B_p_seg

    # --- fixed-length path (original logic, cu_seqlens=None) ---
    # NT = ceil(T/BT): the last chunk may be incomplete. We pad T up to NT*BT so the
    # [B,H,NT,BT,*] reshape works, but mask the padded positions so they contribute NOTHING
    # (matching the triton kernel's m_t = idx < T boundary mask):
    #   - padded v/k/w are 0 -> kg=0 -> Bt=0, and the kg^T@w term is 0
    #   - g_last for the LAST chunk uses the REAL last token (idx T-1), NOT the padded tail,
    #     so alpha = exp(g_last_real) and At = alpha*I for the last chunk — this is correct:
    #     the state IS scaled by the last real token's gate (the triton kernel does the same).
    #   - for the last chunk, padded positions in g_c are set so gate=exp(g_last-g_pad)=1
    #     (kg=0 there anyway, so gate value is irrelevant), achieved by g_pad=g_last.
    NT = (T + BT - 1) // BT
    T_pad = NT * BT
    pad_len = T_pad - T
    if pad_len > 0:
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        w = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad_len))

    # g from chunk_local_cumsum(head_first=False) is [B,T,H]; permute to [B,H,T]
    if g.dim() == 3 and g.shape[1] == T:
        g = g.permute(0, 2, 1).contiguous()  # [B,T,H] -> [B,H,T]
    # mask: True for real tokens (idx < T). Padded positions get k=w=v=0 (already), and we
    # set g at padded positions = g_last_real so exp(g_last-g_pad)=1 (no-op gate there).
    idx = torch.arange(T_pad, device=k.device)
    real_mask = (idx < T).view(1, 1, 1, T_pad)  # [1,1,1,T_pad] broadcast
    if g.shape[-1] != T_pad:
        # extend g to T_pad: padded positions take the last real g value (so gate=1 there)
        g_real_last = g[..., T - 1:T]  # [B,H,1]
        g = torch.cat([g, g_real_last.expand(*g.shape[:-1], pad_len)], dim=-1)
    # overwrite padded positions in g with the last real value (in case g was already T_pad)
    g = torch.where(real_mask, g, g[..., T - 1:T].expand_as(g))

    device = k.device
    dtype = torch.float32
    eye_K = torch.eye(K, device=device, dtype=dtype)

    # reshape k,w,v [B,T,H,D] -> [B,H,NT,BT,D]
    k_c = k.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, K)
    w_c = w.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, K)
    v_c = v.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, V)
    g_c = g.float().reshape(B, H, NT, BT)             # [B,H,NT,BT]
    # g_last per chunk: for full chunks it's the chunk's last token; for the last (possibly
    # incomplete) chunk it's the REAL last token at idx T-1, NOT the padded tail.
    chunk_last_idx = torch.minimum(torch.full((NT,), BT, device=k.device, dtype=torch.long),
                                   torch.tensor([T - (t * BT) for t in range(NT)], device=k.device, dtype=torch.long).clamp(min=1))
    g_last = g_c.gather(-1, (chunk_last_idx - 1).view(1, 1, NT, 1).expand(B, H, NT, 1)).squeeze(-1)  # [B,H,NT]

    gate = torch.exp(g_last.unsqueeze(-1) - g_c)      # [B,H,NT,BT]
    kg = k_c * gate.unsqueeze(-1)                     # [B,H,NT,BT,K]
    alpha = torch.exp(g_last)                         # [B,H,NT]

    # At = alpha*I - kg^T @ w   [B,H,NT,K,K] ;  Bt = kg^T @ v   [B,H,NT,K,V]
    At = alpha[..., None, None] * eye_K - torch.matmul(kg.transpose(-1, -2), w_c)
    Bt = torch.matmul(kg.transpose(-1, -2), v_c)
    # Free the large fp32 intermediates (k_c/w_c/v_c/kg/g_c/gate/alpha/g_last) — they are
    # dead after At/Bt are computed (the sequential loop below only uses At/Bt one at a time).
    del k_c, w_c, v_c, kg, g_c, gate, alpha, g_last

    # Sequential chunk compose: O(1) memory (one chunk's A_t/B_t at a time) and more
    # stable than a tree-reduce at long sequences — NT-fold full-matrix products compound
    # fp32 rounding, while the running state (M, B) stays bounded since rho(A_t)<1.
    # Forward compose order: hi@lo (A_{t} @ M).
    A_p = eye_K.expand(B, H, K, K).clone()    # M_0 = I
    B_p = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    for t in range(NT):
        At_t = At[:, :, t]                       # [B,H,K,K]
        Bt_t = Bt[:, :, t]                       # [B,H,K,V]
        # baddbmm fuses the matmul+add (Bt_t + At_t@B_p) into one kernel; the [B,H]
        # dims merge into a uniform-stride batch view (zero-copy reshape).
        B_p = torch.baddbmm(
            Bt_t.reshape(-1, K, V), At_t.reshape(-1, K, K), B_p.reshape(-1, K, V)
        ).view(B, H, K, V)
        A_p = torch.matmul(At_t, A_p)            # state_t = A_t @ state_{t-1} + B_t
    del At, Bt

    return A_p, B_p


def compute_da_p_db_p(q, k, w, do, dv, g, scale, chunk_size=64, cu_seqlens=None):
    """Backward pre_process: compute dA_p [B,H,K,K] (= dM_p) and dB_p [B,H,K,V] (= dS_ext)
    in fp32. Vectorized counterpart of compute_a_p_b_p for the backward pass.

    Backward chunk recurrence:
      dS_t = dM_t @ dS_{t+1} + db_t,   t = NT-1 .. 0,  dS_NT = dht
      dM_t = alpha_t*I - w_t^T @ kg_t        (form1; = transpose of forward A_t)
      db_t = (q_t * exp(g_cumsum_t))^T @ do_t * scale  -  w_t^T @ dv_local_t
    db_t is INDEPENDENT of dS_{t+1} (clean affine -> tree-reduce valid). q is gated by
    exp(g_cumsum) to match the forward o_inter=(q*exp(g))@h. Tree-reduce uses the BACKWARD
    compose (lo@hi, dM_0 leftmost) -- NOT the forward hi@lo, which would reverse the chain.

    Inputs (head_first=False, before fwd_h permute):
      q: [B,T,H,K], k: [B,T,H,K], w: [B,T,H,K], do: [B,T,H,V], dv: [B,T,H,V]
      g: [B,H,T] (or [B,T,H]), scale: float   (g is the chunk-local cumsum, as in compute_a_p_b_p)
    Returns (dA_p [B,H,K,K], dB_p [B,H,K,V]) in fp32.
    Assumes cu_seqlens is None and T is a multiple of BT.
    """
    B, T, H, K = k.shape
    V = dv.shape[-1]
    BT = chunk_size

    # --- varlen (pack/cu_seqlens) path: only process the FIRST local sequence ---
    # In CP scan backward, only the sequence spanning from the PREVIOUS rank needs
    # dA_p/dB_p for cross-rank merge.
    #
    # Returns FLAT [1,H,K,K]/[1,H,K,V] (no N dim) — see compute_a_p_b_p for rationale
    # (all_gather shape consistency + slot semantics). The merged cross-rank dht lands in
    # slot [-1] (the spanning/outbound segment) after prefix_scan_compose_bwd; the caller
    # expands the flat [1,...] into [N_local,...] at the [-1] slot, leaving [0..N-2]=0.
    if cu_seqlens is not None:
        # Extract the first local sequence segment (the one spanning from the PREV rank)
        first_start = cu_seqlens[0].item()
        first_end = cu_seqlens[1].item()
        first_T = first_end - first_start
        if first_T == 0:
            dA_p_out = torch.zeros(1, H, K, K, device=k.device, dtype=torch.float32)
            dB_p_out = torch.zeros(1, H, K, V, device=k.device, dtype=torch.float32)
            return dA_p_out, dB_p_out
        # Slice for the first segment
        q_first = q[:, first_start:first_end, :, :]
        k_first = k[:, first_start:first_end, :, :]
        w_first = w[:, first_start:first_end, :, :]
        do_first = do[:, first_start:first_end, :, :]
        dv_first = dv[:, first_start:first_end, :, :]
        # g layout normalization must happen BEFORE slicing. The caller contract is
        # [B,T,H] (chunk_local_cumsum(head_first=False)). An UNAMBIGUOUS [B,H,T]
        # input is still accepted (shape[1] != T excludes [B,T,H]); the naive
        # shape[-1]==T test alone misfires when H == T -- a CORRECT [B,T,H] gets
        # permuted, then the dim=1 slice cuts the HEAD axis and dA_p/dB_p are
        # poisoned with exp() of cross-head g differences (NaN). See PR review and
        # the H==T_total regression case in tests/ut_fsdp/ops/gdn/test_parallel_scan.py.
        if g.dim() == 3 and g.shape[1] != T and g.shape[-1] == T:
            g = g.permute(0, 2, 1).contiguous()       # [B,H,T] -> [B,T,H]
        g_first = g[:, first_start:first_end, :]      # [B, first_T, H] (time-dim slice)
        dA_p_seg, dB_p_seg = compute_da_p_db_p(q_first, k_first, w_first, do_first, dv_first,
                                                  g_first, scale, chunk_size=chunk_size, cu_seqlens=None)
        # dA_p_seg/dB_p_seg are [1,H,K,K]/[1,H,K,V] — already flat. The merged cross-rank
        # dht will land in slot [-1] via prefix_scan_compose_bwd; caller expands to
        # [N_local,...] at [-1], leaving [0..N-2]=0 (those segments get no inbound gradient).
        return dA_p_seg, dB_p_seg

    # --- fixed-length path (original logic, cu_seqlens=None) ---
    NT = (T + BT - 1) // BT
    T_pad = NT * BT
    pad_len = T_pad - T
    if pad_len > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad_len))
        w = torch.nn.functional.pad(w, (0, 0, 0, 0, 0, pad_len))
        do = torch.nn.functional.pad(do, (0, 0, 0, 0, 0, pad_len))
        dv = torch.nn.functional.pad(dv, (0, 0, 0, 0, 0, pad_len))

    if g.dim() == 3 and g.shape[1] == T:
        g = g.permute(0, 2, 1).contiguous()
    idx = torch.arange(T_pad, device=k.device)
    real_mask = (idx < T).view(1, 1, 1, T_pad)
    if g.shape[-1] != T_pad:
        g_real_last = g[..., T - 1:T]
        g = torch.cat([g, g_real_last.expand(*g.shape[:-1], pad_len)], dim=-1)
    g = torch.where(real_mask, g, g[..., T - 1:T].expand_as(g))

    device = k.device
    dtype = torch.float32
    eye_K = torch.eye(K, device=device, dtype=dtype)

    q_c = q.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, K)
    k_c = k.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, K)
    w_c = w.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, K)
    do_c = do.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, V)
    dv_c = dv.float().permute(0, 2, 1, 3).reshape(B, H, NT, BT, V)
    g_c = g.float().reshape(B, H, NT, BT)

    chunk_last_idx = torch.minimum(torch.full((NT,), BT, device=k.device, dtype=torch.long),
                                   torch.tensor([T - (t * BT) for t in range(NT)], device=k.device, dtype=torch.long).clamp(min=1))
    g_last = g_c.gather(-1, (chunk_last_idx - 1).view(1, 1, NT, 1).expand(B, H, NT, 1)).squeeze(-1)

    gate = torch.exp(g_last.unsqueeze(-1) - g_c)      # [B,H,NT,BT]
    kg = k_c * gate.unsqueeze(-1)                     # [B,H,NT,BT,K]
    alpha = torch.exp(g_last)                         # [B,H,NT]

    # dM per chunk: alpha*I - w^T @ kg  (form1; = transpose of forward A_t = alpha*I - kg^T@w).
    # alpha multiplies ONLY the identity, NOT the w^T@kg term — kg already carries its own
    # per-token gate exp(g_last-g). Do NOT use form2 alpha*(I-w^T@kg).
    dM = alpha[..., None, None] * eye_K - torch.matmul(w_c.transpose(-1, -2), kg)  # [B,H,NT,K,K]

    # Backward recurrence:
    #   dS_t = dM_t @ dS_{t+1} + db_t   (t = NT-1 .. 0, dS_NT = dht)
    # db_t is INDEPENDENT of dS_{t+1} (the kg@dS coupling from the full dv folds into
    # dM_t, leaving a clean affine recurrence).
    # db_t = (q_t * exp(g_cumsum_t))^T @ do_t * scale  -  w_t^T @ dv_local_t
    # The q MUST be gated by exp(g_cumsum), matching the forward chunk_fwd_o
    # (o_inter=(q*exp(g))@h) and the bwd_dhu kernel; an ungated q^T@do makes db
    # ~exp(g_cumsum) too large.

    # q gated by exp(g_cumsum) per token (g_c is the chunk-local cumsum, same as forward)
    q_gated = q_c * torch.exp(g_c).unsqueeze(-1)                          # [B,H,NT,BT,K]
    # db_t = q_gated^T @ do_t * scale - w^T @ dv_local_t   [B,H,NT,K,V]
    db = torch.matmul(q_gated.transpose(-1, -2), do_c) * scale            # [B,H,NT,K,V]
    db = db - torch.matmul(w_c.transpose(-1, -2), dv_c)                   # [B,H,NT,K,V]
    # Free the large fp32 intermediates — dead after dM/db are computed (sequential loop below
    # only uses dM/db one at a time). bwd has 5 input chunks (q/k/w/do/dv) + kg/g_c/gate/alpha/q_gated.
    del q_c, k_c, w_c, do_c, dv_c, kg, g_c, gate, alpha, g_last, q_gated

    # Sequential chunk compose. BACKWARD compose order (lo@hi):
    # dS_t = dM_t @ dS_{t+1} + db_t (t ascending 0..NT-1, dS_NT=dht) yields
    #   dA_p = dM_0 @ dM_1 @ ... @ dM_{NT-1}  (dM_0 leftmost = lo@hi)
    #   dB_p = dM_0@..@dM_{NT-2}@db_{NT-1} + .. + db_0
    # So accumulate from t=NT-1 down to 0: M = dM_t @ M, B = dM_t @ B + db_t.
    dA_p = eye_K.expand(B, H, K, K).clone()
    dB_p = torch.zeros(B, H, K, V, device=device, dtype=dtype)
    for t in range(NT - 1, -1, -1):
        dM_t = dM[:, :, t]                          # [B,H,K,K]
        db_t = db[:, :, t]                          # [B,H,K,V]
        # baddbmm fusion as in the forward loop (db_t + dM_t@dB_p).
        dB_p = torch.baddbmm(
            db_t.reshape(-1, K, V), dM_t.reshape(-1, K, K), dB_p.reshape(-1, K, V)
        ).view(B, H, K, V)
        dA_p = torch.matmul(dM_t, dA_p)             # M = dM_t @ M
    del dM, db

    return dA_p, dB_p


def prefix_scan_compose(all_A_p, all_B_p, cp_rank, pre_num_ranks=None, is_first_rank=None):
    """Prefix-scan compose for the CURRENT rank only (no over-computation).

    Computes init_state: S = 0; for j in 1..pre_num_ranks: S = M_{j-1} @ S + S_ext_{j-1}.
    For varlen, pre_num_ranks may differ from cp_rank (sequences spanning multiple ranks).
    all_A_p: [kv, N, H, K, K], all_B_p: [kv, N, H, K, V]
    Returns prefix_B [N, H, K, V] (None if first rank, since init_state = 0).
    """
    # is_first_rank=None (default, fixed-length): use cp_rank==0
    # is_first_rank=bool (varlen): use the provided value
    _is_first = (cp_rank == 0) if is_first_rank is None else is_first_rank
    if _is_first:
        return None  # first rank's initial_state = 0
    num_ranks = pre_num_ranks if pre_num_ranks is not None else cp_rank
    # triton merge kernel (merge_scan.py:merge_fwd_triton) hardcodes cur_rank=idx, i.e.
    # first_rank_of_first_seq==0 (full_span). For non-full-span varlen (current rank's
    # first overlapping sequence starts mid-stream), triton would chain the wrong ranks.
    # Force the Python path for the non-full-span case until merge_fwd_triton is taught
    # pre_num_ranks/is_first_rank. Fixed-length (is_first_rank is None) is always full_span.
    _full_span_triton = (is_first_rank is None) or (cp_rank == num_ranks)
    if _full_span_triton:
        from .triton.merge_scan import merge_fwd_triton
        return merge_fwd_triton(all_A_p, all_B_p, cp_rank, all_A_p.shape[0])
    # fall through to Python path for non-full-span varlen
    N, H, K, _ = all_A_p.shape[1:]
    V = all_B_p.shape[-1]
    device, dtype = all_A_p.device, all_A_p.dtype

    prefix_B = torch.zeros(N, H, K, V, device=device, dtype=dtype)
    # Chain exactly [first_rank_of_first_seq, cp_rank - 1] (ascending). Chaining from
    # rank 0 instead (full-span assumption) mixes transition matrices from a DIFFERENT
    # sequence under non-full-span neat_packing -> cross-sample state contamination.
    # Full-span degenerates to [0, cp_rank-1] since pre_num_ranks=cp_rank.
    start_rank = cp_rank - num_ranks
    for j in range(start_rank, cp_rank):
        prefix_B = torch.matmul(all_A_p[j], prefix_B) + all_B_p[j]
    return prefix_B


def prefix_scan_compose_bwd(all_dA_p, all_dB_p, kv_rank, kv_size,
                            post_num_ranks=None, is_last_rank=None):
    """Backward merge: dht_r = chain dM_j @ dht + dS_ext_j for j > r (descending).

    For varlen, post_num_ranks may differ from kv_size-1-kv_rank.
    Returns dht [N, H, K, V] for the current rank (zeros if last rank).
    """
    # num_ranks must be computed BEFORE the full-span guard below (which references it;
    # the old ordering computed it after the guard -> UnboundLocalError).
    num_ranks = post_num_ranks if post_num_ranks is not None else (kv_size - 1 - kv_rank)
    # triton merge kernel (merge_scan.py:merge_bwd_triton) hardcodes the full_span
    # assumption (last_rank_of_last_seq == kv_size-1). For non-full-span varlen it would
    # chain the wrong ranks. Force the Python path for non-full-span until merge_bwd_triton
    # is taught post_num_ranks/is_last_rank. Fixed-length (is_last_rank is None) is full_span.
    _full_span_triton = (is_last_rank is None) or (kv_rank + num_ranks == kv_size - 1)
    if _full_span_triton:
        from .triton.merge_scan import merge_bwd_triton
        return merge_bwd_triton(all_dA_p, all_dB_p, kv_rank, kv_size)
    # fall through to Python path for non-full-span varlen
    N, H, K, _ = all_dA_p.shape[1:]
    V = all_dB_p.shape[-1]
    device, dtype = all_dA_p.device, all_dA_p.dtype

    # is_last_rank=None (default, fixed-length): use kv_rank == kv_size-1
    # is_last_rank=bool (varlen): use the provided value
    _is_last = (kv_rank == kv_size - 1) if is_last_rank is None else is_last_rank
    if _is_last:
        return torch.zeros(N, H, K, V, device=device, dtype=dtype)

    dht = torch.zeros(N, H, K, V, device=device, dtype=dtype)
    # Chain exactly [kv_rank + 1, last_rank_of_last_seq] (descending). Chaining to
    # kv_size-1 instead (full-span assumption) reaches past the current sequence's end
    # into a different sequence -> cross-sample gradient contamination. Full-span
    # degenerates to [kv_rank+1, kv_size-1] since post_num_ranks=kv_size-1-kv_rank.
    end_rank = kv_rank + num_ranks
    for j in range(end_rank, kv_rank, -1):
        dht = torch.matmul(all_dA_p[j], dht) + all_dB_p[j]
    return dht


class CpInitStateStore(OffloadManager):
    """OffloadManager reserved for the CP-merged initial_state (per-layer slots).

    Kept off the shared OffloadManager singleton on purpose (public manager only
    does generic put/pop; PR review, htwang): routing the slots through its
    primitives would share the g/o/A swap `items` dict (ref-counted OffloadItem,
    clear(None) hazards) or the single-slot npu_item LIFO whose order the swap
    chain depends on. __init__ skips the base swap infrastructure -- this
    instance carries ONLY the cp slots (a dedicated singleton via the same
    metaclass).
    """

    def __init__(self):
        # Per-layer LIFO: under chunk_mbs several micro-batches stash the SAME
        # layer before any backward pops (autograd pops the last-built micro
        # first); a single-key dict would return None for the first micro and
        # silently mis-compute dq/dk/dw/dg for cp_rank>0's first chunks.
        self.cp_init_state = {}

    def put_cp_init_state(self, layer_idx, tensor):
        """Stash the CP-merged initial_state for this layer (CP scan, cp_size>1 only)."""
        self.cp_init_state.setdefault(layer_idx, []).append(tensor)

    def pop_cp_init_state(self, layer_idx):
        """Retrieve (LIFO) and clear the CP-merged initial_state for this layer."""
        stack = self.cp_init_state.get(layer_idx)
        if not stack:
            return None
        tensor = stack.pop()
        if not stack:
            del self.cp_init_state[layer_idx]
        return tensor
