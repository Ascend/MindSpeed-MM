"""Qwen3.5 dense/MoE shared modeling helpers (imported by both modeling files;
must not import either of them)."""

from typing import List

import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes
from mindspeed_mm.fsdp.distributed.context_parallel.communication import (
    split_forward_gather_backward,
)
from mindspeed_mm.fsdp.distributed.context_parallel.pipeline_comm import conv_tail_exchange
from mindspeed_mm.fsdp.ops.gdn.parallel_scan import build_cp_context


def cp_split_inputs_and_build_context(position_ids, text_position_ids, inputs_embeds,
                                      total_seq_len, use_packing, config, kwargs):
    """USP + KV-AllGather input-side preprocessing (the `is_kvallgather_enable` branch
    body of TextModel.forward, shared by the dense/MoE twins).

    CP-splits position_ids / text_position_ids / inputs_embeds over the total cp group,
    and for packed varlen builds the rank-local scan-CP context into `kwargs` (mutated
    in place). Returns the split (position_ids, text_position_ids, inputs_embeds).
    """
    ps = get_parallel_state()
    # USP + KV AllGather: contiguous split over the total cp group (ulysses x
    # kvallgather) so ulysses gathers a contiguous Q block and kv_ag reconstructs
    # full K/V in natural order (causal via K-slice + is_causal).
    cp_group = ps.get_cp_group()
    split_sizes = cal_split_sizes(total_seq_len, ps.get_cp_group_size())
    position_ids = split_forward_gather_backward(position_ids, process_group=cp_group, dim=2, split_sizes=split_sizes)
    text_position_ids = split_forward_gather_backward(text_position_ids, process_group=cp_group, dim=1, split_sizes=split_sizes)
    inputs_embeds = split_forward_gather_backward(inputs_embeds, process_group=cp_group, dim=1, split_sizes=split_sizes)
    # Varlen packing: build rank-local cu_seqlens for scan CP
    if use_packing:
        global_cu_seqlens = kwargs["cu_seqlens"].to(torch.int64)
        kv_group = ps.get_kvallgather_group()
        # Exact [rank_start, rank_end) this kv_rank holds. split_sizes is per-cp_rank
        # (cal_split_sizes above); a kv_rank owns the contiguous ulysses sub-group
        # [kv_rank*ulysses_size, (kv_rank+1)*ulysses_size) of cp chunks. Passing the
        # exact range (instead of total//world_size floor-division) keeps local_cu_seqlens
        # aligned with inputs_embeds when total % kv_size != 0 — otherwise the rank with
        # the +1 remainder token gets a local_cu_seqlens short by one, fwd_h writes
        # uninitialized garbage at the tail, and loss goes NaN from iter 1.
        ulysses_size = ps.get_ulysses_group_size()
        kv_rank = ps.get_kvallgather_rank()
        kv_size = ps.get_kvallgather_group_size()
        cp_start_idx = kv_rank * ulysses_size
        cp_end_idx = (kv_rank + 1) * ulysses_size
        kv_rank_start = sum(split_sizes[:cp_start_idx])
        kv_rank_end = sum(split_sizes[:cp_end_idx])
        # KV-group rank boundaries (ceil split): each kv_rank owns ulysses_size contiguous
        # cp chunks. Aggregate split_sizes (per-cp_rank) by ulysses_size to get per-kv_rank
        # token counts, then cumsum -> [0, b1, ..., total] (len = kv_size+1). Passed to
        # build_cp_context so the else-branch rank-of-position lookup uses EXACT ceil
        # boundaries (searchsorted) instead of floor pos//part_len (wrong when total%kv_size!=0).
        _split_t = torch.tensor(split_sizes, dtype=torch.int64)
        _kv_split = _split_t.view(kv_size, ulysses_size).sum(dim=1)
        kv_rank_boundaries = torch.cat(
            [torch.zeros(1, dtype=torch.int64), _kv_split.cumsum(0)]
        ).tolist()
        cp_ctx = build_cp_context(global_cu_seqlens, kv_group,
                                  conv1d_kernel_size=getattr(config, "conv_kernel_size", 4),
                                  rank_start=kv_rank_start, rank_end=kv_rank_end,
                                  rank_boundaries=kv_rank_boundaries)
        kwargs["cu_seqlens"] = cp_ctx["local_cu_seqlens"]
        kwargs["cu_seq_lens_q"] = cp_ctx["local_cu_seqlens"]
        kwargs["cp_pre_num_ranks"] = cp_ctx["pre_num_ranks"]
        kwargs["cp_post_num_ranks"] = cp_ctx["post_num_ranks"]
        kwargs["cp_is_first_rank"] = cp_ctx["is_first_rank"]
        kwargs["cp_is_last_rank"] = cp_ctx["is_last_rank"]
        # B1: fresh-start input for the conv tail (see the conv1d block in the
        # GatedDeltaNet forward) -- tokens of the first local sample living on
        # previous ranks. When < conv W-1, the recv tail crosses a sample
        # boundary and its foreign leading tokens must be zeroed.
        kwargs["cp_pre_num_conv_tokens"] = int(cp_ctx["pre_num_conv_tokens"])
    return position_ids, text_position_ids, inputs_embeds


def _vision_image_dp_partition(image_grid_thw: torch.Tensor, kv_size: int) -> List[int]:
    """Split N images into K contiguous, token-balanced groups (one per kvallgather rank).

    Returns boundaries [b_0, b_1, ..., b_K] (length K+1); rank r owns image range
    [b_r, b_{r+1}). Each image's patch token count is prod(grid_thw[i]) -- the same
    count used by cu_seqlens and pos_embed.

    Deterministic: depends only on image_grid_thw (identical on every CP rank), no
    communication -- every rank independently computes the same boundaries. The
    midpoint rule (image i goes to rank r when its cumulative-token midpoint falls
    in [r, r+1)) keeps the load roughly balanced; when num_images < K some ranks
    get an empty range (handled by the empty-subset fallback in vision_image_dp_forward).
    """
    num_images = image_grid_thw.shape[0]
    # Compute on CPU throughout: grid is only [N, 3] and NPU has limited float64 support
    counts = image_grid_thw.detach().cpu().prod(-1).to(torch.float64)
    total = counts.sum()
    ends = counts.cumsum(0)
    starts = ends - counts
    midpoints = (starts + ends) / 2.0
    # image i -> rank floor(mid_i * K / total); clamp guards float edge overflow
    assign = (midpoints * kv_size / total).long().clamp_(max=kv_size - 1)
    # b_r = first image index with assign >= r (assign non-decreasing -> monotone bounds)
    bounds = torch.searchsorted(
        assign, torch.arange(kv_size + 1, dtype=torch.long, device=assign.device)
    ).tolist()
    bounds[0] = 0
    bounds[kv_size] = num_images
    return bounds


class _VisionImageDPGather(torch.autograd.Function):
    """Variable-length all-gather of vision output embeddings across the kvallgather group.

    forward:  rank r contributes its image subset's output local_embeds [L_r, D]
    (D=out_hidden_size); concatenated in kv-rank order every rank ends with the full
    [sum(L_r), D] -- row order identical to the unsplit case (subsets are contiguous
    image ranges, so kv-rank order = original image order). Pure data movement (the
    zero-padded rows are sliced off before the concat), no arithmetic -> bitwise exact.

    backward: the upstream gradient (the text-side CP split's backward already
    all-gathers across the whole cp group and divides by cp_size, so every rank
    receives the SAME grad_output) only needs slicing to this rank's range plus a
    kv_size compensation. Derivation (K=kv_size, U=ulysses_size, cp_size=K*U):
      - unsplit baseline: every ulysses pair runs the FULL vision F, receiving
        h=g/cp_size at its output; pair contribution (the internal ulysses gather's
        "up" xU) = U*F^T*h; summed over K pairs = K*U*F^T*g/cp_size = F^T*g
        (correct after FSDP sums over all cp ranks).
      - image DP: pair k computes only subset F_k. If this backward multiplies by m,
        pair k contributes U*F_k^T*m*g[slice_k]/cp_size; summed over K pairs =
        (U*m/cp_size)*F^T*g. Equating to F^T*g gives m = cp_size/U = K.
    For K=16 the x16 is a power of two: an exact exponent shift, no rounding.

    Note: this scheme requires both ranks of a pair to receive the SAME vision
    output gradient -- guaranteed by "grad_output identical on all cp ranks" plus
    "the same kv_rank slices the same range".
    """

    @staticmethod
    def forward(ctx, local_embeds, kv_group, kv_rank, kv_size, grad_multiplier):
        device = local_embeds.device
        my_len = local_embeds.shape[0]
        # 1) Exchange per-rank subset output token counts (metadata for varlen all-gather)
        len_t = torch.tensor([my_len], dtype=torch.int64, device=device)
        len_list = [torch.empty(1, dtype=torch.int64, device=device) for _ in range(kv_size)]
        torch.distributed.all_gather(len_list, len_t, group=kv_group)
        lengths = torch.cat(len_list).cpu().tolist()
        ctx.lengths = lengths
        ctx.kv_rank = kv_rank
        ctx.grad_multiplier = grad_multiplier

        max_len = max(lengths)
        if max_len == 0:
            # Whole group has no images (unreachable given the upstream guarantee of
            # at least one image; defensive. All ranks share grid_thw so the branch
            # is consistent). clone avoids returning the input itself as output.
            return local_embeds.clone()

        # 2) Zero-pad to the group max length, then symmetric all-gather; padding
        # rows exist only during communication
        dim = local_embeds.shape[1]
        dtype = local_embeds.dtype
        padded = torch.zeros(max_len, dim, dtype=dtype, device=device)
        if my_len > 0:
            padded[:my_len] = local_embeds
        gathered = [torch.empty(max_len, dim, dtype=dtype, device=device) for _ in range(kv_size)]
        torch.distributed.all_gather(gathered, padded.contiguous(), group=kv_group)

        # 3) Strip padding and concat in kv-rank order = the full output in
        # original image order
        return torch.cat([gathered[k][: lengths[k]] for k in range(kv_size)], dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        lengths = ctx.lengths
        start = sum(lengths[: ctx.kv_rank])
        end = start + lengths[ctx.kv_rank]
        grad_local = grad_output[start:end]
        if ctx.grad_multiplier != 1:
            grad_local = grad_local * ctx.grad_multiplier
        return grad_local, None, None, None, None


def vision_image_dp_forward(visual, pixel_values, image_grid_thw, kwargs):
    """Vision tower image-level data parallelism (shared by the dense/MoE twins).

    Auto-enabled: active whenever kvallgather CP is on (and ring==1). No env
    switch -- the path is bitwise-exact and the memory win at long sequences is
    unconditional, so an off state has no use case.
    Baseline: every cp rank runs the FULL image set through the vision tower
    (27 layers of activations x all patch tokens; at 1M pack ~250K vision
    tokens per rank -> OOM). Here images are split into contiguous subsets
    along the kvallgather dimension: the ulysses pair sharing a kv_rank gets
    the same subset (token-level split inside the pair is unchanged), so each
    rank computes only 1/K of the images; _VisionImageDPGather reassembles
    the complete set (image order preserved) -> downstream merge/CP-split
    logic is identical to the baseline. Equivalence: vision attention is
    per-image varlen (cu_seqlens segments by image, is_causal=False) and
    pos_embed/patch_embed/merger are per-image or per-token ops, so images
    have zero numerical coupling -> any contiguous split yields the same
    per-image results (modulo kernel-shape effects, the same class as
    ulysses splits).
    ring>1: the gradient compensation factor is not derived for ring -- the
    feature stays off (replicated baseline vision) instead.

    Returns the merged BaseModelOutputWithPooling when the image-DP path runs,
    or None when the caller must take the replicated baseline path (image-DP
    inactive, or the empty-subset fallback below).
    """
    ps = get_parallel_state()
    use_image_dp = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and ps.is_cp_enable()
        and ps.get_kvallgather_group_size() > 1
        and ps.get_ring_group_size() == 1
        and pixel_values.shape[0] > 0
    )
    dp_bounds = None
    if use_image_dp:
        kv_group = ps.get_kvallgather_group()
        kv_size = ps.get_kvallgather_group_size()
        kv_rank = ps.get_kvallgather_rank()
        # Gradient multiplier m = cp_size / ulysses_size = kv_size (ring==1);
        # see the derivation in _VisionImageDPGather's docstring
        if ps.get_cp_group_size() // ps.get_ulysses_group_size() != kv_size:
            raise RuntimeError(
                "vision image DP: cp_size/ulysses_size != kv_size, unexpected topology"
            )
        row_counts = image_grid_thw.prod(-1)
        if int(row_counts.sum()) != pixel_values.shape[0]:
            raise ValueError(
                "vision image DP: pixel_values rows != sum(grid_thw.prod(-1)); "
                "cannot split pixel_values on image boundaries "
                f"({pixel_values.shape[0]} vs {int(row_counts.sum())})"
            )
        dp_bounds = _vision_image_dp_partition(image_grid_thw, kv_size)
        if any(dp_bounds[i] == dp_bounds[i + 1] for i in range(kv_size)):
            # Some kv rank got 0 images (num_images < K, or extremely uneven
            # sizes, e.g. 15 tiny images + 1 huge one with every midpoint inside
            # rank0's range): a rank with an empty subset never runs the vision
            # tower -> its vision params leave this rank's autograd graph, the
            # FSDP post-backward hook never fires, and other ranks' gradient
            # reduce-scatter may hang (FSDP2 semantics for unused params are
            # not assumed). bounds are computed independently from the same
            # grid_thw on every rank -> the decision is uniform; fall back to
            # the baseline path (bitwise equal to DP off; such batches have few
            # images and vision is not the memory bottleneck anyway).
            dp_bounds = None

    if dp_bounds is None:
        return None
    bounds = dp_bounds
    img_start, img_end = bounds[kv_rank], bounds[kv_rank + 1]
    row_start = int(row_counts[:img_start].sum())
    row_end = int(row_counts[:img_end].sum())
    # the guard above guarantees a non-empty subset; this branch is defensive
    # clone (not contiguous -- a row slice is already contiguous so
    # contiguous() is a no-op): a view would let the conv's backward
    # saved tensor pin the WHOLE pixel_values storage (~3-11 GiB at 1M
    # pack) even though only 1/K is used. The batch-side reference is
    # popped by train_step after forward.
    my_pixel_values = pixel_values[row_start:row_end].clone()
    my_grid_thw = image_grid_thw[img_start:img_end]
    vision_output: BaseModelOutputWithPooling = visual(
        my_pixel_values, grid_thw=my_grid_thw, return_dict=True, **kwargs
    )
    local_embeds = vision_output.pooler_output
    image_embeds_full = _VisionImageDPGather.apply(
        local_embeds, kv_group, kv_rank, kv_size, kv_size
    )
    # pooler_output stays a FLAT tensor (no split): the old path split it
    # and forward immediately cat'd it back -- an identity round-trip that
    # held one extra full [T_out, D] (~1-2 GiB at 1M pack). forward
    # accepts both tuple and flat forms. last_hidden_state is this rank's
    # subset intermediate (small: L_r x D), kept for API consistency with
    # the non-DP path; unused on the training path (pooler_output is the
    # merged full-sequence embeds).
    return BaseModelOutputWithPooling(
        last_hidden_state=vision_output.last_hidden_state,
        pooler_output=image_embeds_full,
    )


def prepend_conv_cp_tail(
    mixed_qkv: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
    *,
    conv_kernel_size: int,
    kv_group,
    is_first: bool,
    is_last: bool,
    pre_num_conv_tokens: int | None,
) -> tuple[torch.Tensor, int, torch.LongTensor | None]:
    """Scan CP: causal conv1d cross-kv-team boundary. One collective all_gather
    exchanges every team's last W-1 tokens; take the previous team's, prepend it,
    and run the conv on the padded tensor. The first W-1 conv outputs (the tail's
    own) are dropped by the caller after the conv impl dispatch.

    Call ONLY when kvag scan CP is enabled (the caller gates on kvag_enabled).

    Returns (mixed_qkv, pipe_conv_pad, conv1d_cu_seqlens):
      mixed_qkv          [recv_tail(W-1) | local_T]
      pipe_conv_pad      W-1 (the tail was prepended)
      conv1d_cu_seqlens  cu_seqlens shifted right by W-1 (boundaries relative to
                         the padded input) for the conv impls; None when the
                         prepended tail is a single continuous stream (fixed-len);
                         the original cu_seqlens when no CP padding happened
    """
    _W = conv_kernel_size
    _tail_send = mixed_qkv[:, -(_W - 1):, :].contiguous()
    _tail_recv = conv_tail_exchange(_tail_send, kv_group, is_first, is_last)
    # B1 fix: fresh-start semantics when the kv boundary lands within W-1
    # tokens after seg0's start (pre_num_conv_tokens < W-1): the tail's
    # leading (W-1-pre) tokens belong to the PREVIOUS sample, while the
    # non-CP reference zero-pads the segment start. Zero them: exact in
    # forward (the conv accumulates 0-weighted terms; adding exact FP
    # zeros never changes a finite value) and in backward (zero grad at
    # those positions -- the foreign tokens do not participate in the
    # reference either). is_first: recv_tail is already zeros (no-op);
    # fixed-len (cu_seqlens=None): single continuous segment, the tail is
    # in-sample history -- no zeroing.
    if cu_seqlens is not None and not is_first and pre_num_conv_tokens is not None:
        _zero_n = _W - 1 - int(pre_num_conv_tokens)
        if _zero_n > 0:
            _tail_recv[:, :_zero_n, :].zero_()
    mixed_qkv = torch.cat([_tail_recv, mixed_qkv], dim=1)
    _pipe_conv_pad = _W - 1

    # When a recv_tail was prepended, the conv1d input is
    # [recv_tail(W-1) | local_T]. cu_seqlens must NOT be dropped here: conv1d's
    # W-1 causal window would then cross packed-sample boundaries -> cross-sample
    # contamination. Instead shift all cu_seqlens boundaries right by W-1
    # (= _pipe_conv_pad). With the tail prepended:
    #   - seg0: bos=0, the tail IS its in-segment history (not zeroed), so the first real
    #     token reads the recv'd cross-rank tail (correct — seg0 is the continuation).
    #   - seg1..N-1: bos=W-1+..., the kernel masks positions before bos (yi_offset<0) to
    #     zero -> each new sample resets (no cross-sample leak).
    # The first W-1 conv outputs (the tail's own outputs) are dropped by the caller.
    # GDN kernels still use the ORIGINAL cu_seqlens (aligned to post-drop local_T).
    #
    # Cross-sample tail (pre_num_conv_tokens < W-1, e.g. sample-aligned rank
    # boundaries or short samples): the foreign leading tail tokens are zeroed
    # in the exchange block above (fresh-start semantics, bitwise-equal to the
    # non-CP reference).
    if _pipe_conv_pad and cu_seqlens is not None:
        _conv1d_cu_seqlens = cu_seqlens.clone()
        _conv1d_cu_seqlens[1:] += _pipe_conv_pad   # shift boundaries by W-1; [0] stays 0
    elif _pipe_conv_pad:
        # CP + non-packing (single sequence): one continuous stream, tail prepend exact.
        _conv1d_cu_seqlens = None
    else:
        # Non-CP: original cu_seqlens (seg0 resets at 0, no tail) — unchanged, no regression.
        _conv1d_cu_seqlens = cu_seqlens
    return mixed_qkv, _pipe_conv_pad, _conv1d_cu_seqlens


class ChunkedPosEmbedInterp(torch.autograd.Function):
    """Chunked bilinear pos_embed interpolation with original-order gradient.

    Forward: processes (4 bilinear corners × N tokens) in token-dimension chunks,
    peak memory ~2 x [chunk, D] instead of [4, N, D].
    Backward: accumulates grad_weight via index_add_ in corner-major order
    (all tokens of corner 0, then corner 1, ...) — the SAME sequential order as
    nn.Embedding's backward on a flat [4, N] input, giving bitwise-identical
    gradients. Chunking within each corner preserves the token order."""

    @staticmethod
    def forward(ctx, weight, idx_tensor, weight_tensor, chunk_size):
        ctx.save_for_backward(idx_tensor, weight_tensor)
        ctx.chunk_size = chunk_size
        ctx.num_embeddings = weight.shape[0]

        N = idx_tensor.shape[1]
        D = weight.shape[1]
        device = weight.device
        dtype = weight.dtype
        result = torch.empty(N, D, dtype=dtype, device=device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            acc = F.embedding(idx_tensor[0, start:end], weight) * weight_tensor[0, start:end, None]
            for i in range(1, 4):
                acc = acc + F.embedding(idx_tensor[i, start:end], weight) * weight_tensor[i, start:end, None]
            result[start:end] = acc
        return result

    @staticmethod
    def backward(ctx, grad_output):
        idx_tensor, weight_tensor = ctx.saved_tensors
        N = idx_tensor.shape[1]
        cs = ctx.chunk_size
        grad_weight = torch.zeros(ctx.num_embeddings, grad_output.shape[1],
                                   dtype=grad_output.dtype, device=grad_output.device)
        # Corner-major order with index_put_(accumulate=True) — the ONLY scatter variant
        # that is bitwise-identical to nn.Embedding's backward (verified: index_add_ and
        # scatter_add_ differ by ~8 ULP in bf16 for repeated indices; index_put_ matches
        # exactly). See the equivalence test preserved in git history
# (tests/test_pos_embed_equivalence.py, commit 3342d70f^).
        for i in range(4):
            for start in range(0, N, cs):
                end = min(start + cs, N)
                contrib = grad_output[start:end] * weight_tensor[i, start:end, None]
                grad_weight.index_put_((idx_tensor[i, start:end],), contrib, accumulate=True)
        return grad_weight, None, None, None
