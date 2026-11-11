from typing import Callable, Optional, Union, List, Tuple
import logging

from einops import rearrange
import torch
import torch.distributed as dist
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward as _transformers_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from ...distributed.parallel_state import get_parallel_state
from ...utils.device import IS_NPU_AVAILABLE
from ...distributed.context_parallel.utils import cal_split_sizes
from ...distributed.context_parallel.communication import (
    all_to_all,
    split_forward_gather_backward,
)
from ...distributed.context_parallel.kvallgather_context_parallel.kvallgather_context_parallel import (
    kv_all_gather_attention_qkv,
)
from .skip_recompute_flash_attn import skip_recompute_flash_attention

if IS_NPU_AVAILABLE:
    import torch_npu
    from ...distributed.context_parallel.ring_context_parallel.ring_context_parallel import (
        ringattn_context_parallel,
        ringattn_context_parallel_tnd_general,
    )


logger = logging.getLogger(__name__)

_flash_attention_forward = None


ATTN_MASK_NPU_CACHE = {}
ATTN_MASK_SHAPE = (2048, 2048)


def _per_head_softmax_max(
    softmax_max: torch.Tensor, valid_tokens: Optional[int] = None,
    query_mask: Optional[torch.Tensor] = None,
    cu_seq_lens: Optional[List[int]] = None,
) -> torch.Tensor:
    """Reduce the NPU attention kernel's row maxima to one FP32 value per head.

    ``softmax_max`` is ``[B, N, S, 8]``, or ``[T, N, 8]`` for packed sequences;
    the kernel stores each row's maximum logit 8 times in the last dimension.
    TND statistics are stored head-first within each sequence despite their
    tensor shape, so sequence bounds are needed before masking query rows.
    ``valid_tokens`` drops the padding rows at the end of a packed layout.
    ``query_mask`` is ``[B, S]`` (``True`` for real tokens) and drops padded
    query rows anywhere in either layout.
    """
    if softmax_max.ndim not in (3, 4):
        raise ValueError(
            f"expected 3-D (TND) or 4-D softmax_max, got shape {tuple(softmax_max.shape)}"
        )
    if softmax_max.ndim == 3:
        tokens, heads, repeats = softmax_max.shape
        if query_mask is not None and query_mask.numel() != tokens:
            raise ValueError("QK-Clip padding mask does not match the packed query rows")
        bounds = [0] + [int(end) for end in (cu_seq_lens or [tokens]) if end > 0]
        maxima = []
        for start, end in zip(bounds, bounds[1:]):
            valid_end = end if valid_tokens is None else min(end, valid_tokens)
            if valid_end <= start:
                continue
            # The raw buffer concatenates [heads, sequence length, 8] segments.
            segment = softmax_max[start:end].reshape(1, heads, end - start, repeats)
            segment = segment[:, :, :valid_end - start]
            mask = None if query_mask is None else query_mask.reshape(-1)[start:valid_end].unsqueeze(0)
            maxima.append(_per_head_softmax_max(segment, query_mask=mask))
        if not maxima:
            return softmax_max.new_full((heads,), -torch.inf, dtype=torch.float32)
        return torch.stack(maxima).amax(dim=0)
    if query_mask is not None:
        if query_mask.shape == (softmax_max.size(0), softmax_max.size(2)):
            keep = query_mask[:, None, :, None]
        else:
            raise ValueError(
                f"QK-Clip padding mask {tuple(query_mask.shape)} does not match the "
                f"local query rows of softmax_max {tuple(softmax_max.shape)}"
            )
        softmax_max = softmax_max.masked_fill(~keep.to(softmax_max.device), -torch.inf)
    reduction_dims = tuple(d for d in range(softmax_max.ndim) if d != 1)
    return softmax_max.detach().amax(dim=reduction_dims).float()


def _padding_mask(attention_mask: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
    """Return ``attention_mask`` if it is a ``[batch, seq]`` padding mask.

    Only QK-Clip uses the result. Its only model, Kimi, passes padding masks
    and never square kernel masks, so batch == seq is still a padding mask.
    """
    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape[0] == batch_size:
        return attention_mask
    return None


def _require_right_padding(padding_mask: torch.Tensor) -> None:
    """QK-Clip drops padded query rows; the NPU call omits the key-padding mask.

    With right padding, causal attention keeps real queries away from padded
    keys; left or internal padding would put them into real queries' maxima.
    """
    valid = padding_mask >= 1
    if (valid[:, 1:] & ~valid[:, :-1]).any():
        raise ValueError("QK-Clip requires right padding; left or internal padding is unsupported")


def _place_qk_clip_heads_in_full_width(per_head_max: torch.Tensor) -> torch.Tensor:
    """Place a contiguous Ulysses head shard in a full-width vector.

    Unowned heads are negative infinity for the later maximum reduction.
    """
    ps = get_parallel_state()
    ulysses_size = ps.get_ulysses_group_size()
    if ulysses_size <= 1:
        return per_head_max
    local_heads = per_head_max.numel()
    full = per_head_max.new_full((local_heads * ulysses_size,), float("-inf"))
    start = ps.get_ulysses_rank() * local_heads
    full[start:start + local_heads] = per_head_max
    return full


def _make_qk_clip_metric_sink(
    module: torch.nn.Module,
    valid_tokens: Optional[int] = None,
    gather_ulysses: bool = False,
    padding_mask: Optional[torch.Tensor] = None,
    cu_seq_lens_q: Optional[List[int]] = None,
) -> Optional[Callable[[torch.Tensor], None]]:
    """Build a callback that publishes full-width per-head maximum logits.

    ``padding_mask`` is the ``[B, S]`` mask (0 at padding) that the kernel
    does not receive; padded query rows still produce row maxima.
    """
    if not getattr(module, "_qk_clip_enabled", False) or not module.training:
        return None
    publish = getattr(module, "_publish_qk_clip_max_logits", None)
    if not callable(publish):
        return None
    query_mask = None if padding_mask is None else padding_mask >= 1

    def metric_sink(softmax_max: torch.Tensor, token_offset: int = 0) -> None:
        # Chunked packed calls pass the first query row of each chunk.
        mask = query_mask
        if mask is not None and softmax_max.ndim == 3:
            mask = mask.reshape(-1)[token_offset:token_offset + softmax_max.size(0)]
        valid = None if valid_tokens is None else max(0, valid_tokens - token_offset)
        bounds = None
        if softmax_max.ndim == 3 and cu_seq_lens_q is not None:
            bounds = [0] + [end - token_offset for end in cu_seq_lens_q
                            if token_offset < end <= token_offset + softmax_max.size(0)]
        per_head_max = _per_head_softmax_max(
            softmax_max, valid_tokens=valid, query_mask=mask, cu_seq_lens=bounds,
        )
        if gather_ulysses:
            per_head_max = _place_qk_clip_heads_in_full_width(per_head_max)
        publish(per_head_max)

    return metric_sink


def get_attn_mask(device, seq_len=2048):
    """Get or create NPU attention mask"""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([seq_len, seq_len], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


def _convert_cu_seq_lens(
    cu_seq_lens_q: Optional[Union[torch.Tensor, List[int]]],
    cu_seq_lens_k: Optional[Union[torch.Tensor, List[int]]],
    input_layout: Optional[str] = None
) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Converts cu_seq_lens to a list and validates input_layout constraints.

    Args:
        cu_seq_lens: Cumulative sequence lengths, can be a Tensor or List.
        input_layout: The layout of the input tensor (e.g., 'BNSD', 'BSND').

    Returns:
        A tuple containing the processed cu_seq_lens (as list) and the validated layout.
    """

    # 1. Convert Tensor to List if necessary
    if isinstance(cu_seq_lens_q, torch.Tensor):
        cu_seq_lens_q = cu_seq_lens_q.tolist()
    if isinstance(cu_seq_lens_k, torch.Tensor):
        cu_seq_lens_k = cu_seq_lens_k.tolist()

    if input_layout:
        # 2. Normalize layout string
        input_layout = input_layout.upper()

        # Check constraints for packed layouts (BNSD/BSND)
        if input_layout in ["BNSD", "BSND"]:
            if (cu_seq_lens_q is not None or cu_seq_lens_k is not None) and (len(cu_seq_lens_q) > 2 or len(cu_seq_lens_k) > 2):
                raise RuntimeError(
                    f"NPU flash attention layout {input_layout} does not support packing data "
                    f"when cu_seq_lens length > 2"
                )

        # Auto-correct layout if cu_seq_lens is missing
        elif input_layout == "1TND" and (cu_seq_lens_q is None or len(cu_seq_lens_q) == 2) and (cu_seq_lens_k is None or len(cu_seq_lens_k) == 2):
            # Fallback to standard layout if packing info is missing
            input_layout = "BSND"
            cu_seq_lens_q = None
            cu_seq_lens_k = None

        elif input_layout == "1NTD":
            if (cu_seq_lens_q is None or len(cu_seq_lens_q) == 2) and (cu_seq_lens_k is None or len(cu_seq_lens_k) == 2):
                # Fallback to standard layout if packing info is missing
                input_layout = "BNSD"
                cu_seq_lens_q = None
                cu_seq_lens_k = None
            elif len(cu_seq_lens_q) > 2 and len(cu_seq_lens_k) > 2:
                # Judge whether cu_seq_lens is an arithmetic sequence (i.e., all sequences have equal length).
                # If true, switch to BNSD computation
                cu_seq_q = torch.tensor(cu_seq_lens_q)
                diffs_q = torch.diff(cu_seq_q)
                cu_seq_k = torch.tensor(cu_seq_lens_k)
                diffs_k = torch.diff(cu_seq_k)
                if torch.all(diffs_q == diffs_q[0]).item() and torch.all(diffs_k == diffs_k[0]).item():
                    input_layout = "BNSD"
                    cu_seq_lens_q = None
                    cu_seq_lens_k = None

    return cu_seq_lens_q, cu_seq_lens_k, input_layout


def transformers_flash_attention_forward(
    query,
    key,
    value,
    attention_mask,
    **kwargs,
):
    attn_implementation = kwargs.pop("attn_implementation")
    return _transformers_flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        implementation=attn_implementation,
        **kwargs,
    )


def do_ring_attention(
    q,
    k,
    v,
    head_num,
    softmax_scale,
    is_causal,
    fa_layout="SBH",
    attn_mask=None,
    dropout_p=0.0,
    seq_split_lens: Optional[list[int] | torch.Tensor] = None,
    metric_sink: Optional[Callable[[torch.Tensor], None]] = None,
):
    ps = get_parallel_state()
    cp_group = ps.get_ring_group()
    cp_size = ps.get_ring_group_size()
    rank = ps.get_ring_rank()
    cp_global_ranks = ps.get_ring_device_mesh().mesh.tolist()

    cp_para = dict()

    cp_para["causal"] = is_causal
    cp_para["cp_group"] = cp_group
    cp_para["cp_size"] = cp_size
    cp_para["rank"] = rank

    cp_para["cp_global_ranks"] = cp_global_ranks
    cp_para["cp_group_for_send_recv_overlap"] = None
    cp_para["megatron_cp_in_bnsd"] = fa_layout.upper() == "BNSD"
    cp_para["metric_sink"] = metric_sink

    if fa_layout.upper() == "SBH" or fa_layout.upper() == "BNSD":
        # input shapes is a 1-D list
        if seq_split_lens is not None:
            seq_split_lens = seq_split_lens.cpu().tolist()
        output = ringattn_context_parallel(
            q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens
        )
    elif not is_causal and fa_layout.upper() == "TND":
        # input shapes is a 2-D tensor
        output = ringattn_context_parallel_tnd_general(
            q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens
        )
    elif is_causal and fa_layout.upper() == "TND":
        raise NotImplementedError(f"Ring Attention TND layout not support causal mask now.")
    else:
        raise ValueError(
            f"Ring Attention only support fa layout: `SBH`、`SBND` and `TND`, bug got {fa_layout.upper()}."
        )

    return output


def _npu_fa_chunked_varlen_attention(
    query,
    key,
    value,
    q_head_num,
    layout,
    attention_mask,
    cu_seq_lens_q,
    scaling,
    dropout,
    chunk_tokens,
    sparse_mode,
    metric_sink=None,
):
    """Sample-aligned chunked npu_fusion_attention calls for packed varlen
    (features.fa_varlen_chunk_tokens).
    Motivation: in gather-full-Q exact mode every rank passes the FULL
    sequence Q at once (~2GiB at 512K pack); the backward's
    aclnnFlashAttentionUnpaddingScoreGrad workspace scales with it
    (~7GiB) and OOMs directly.
    Approach: packed samples are mutually independent (per-sample causal,
    cu segments by sample), so split the single full-sequence call into
    whole-sample-group sub-calls -- each sub-call only needs that
    group's Q/K/V (no prefix needed: samples are independent!),
    and the workspace drops linearly with chunk size.
    Mathematical equivalence: the varlen kernel already processes
    segments independently; each segment's Q/K/V rows and boundaries
    are identical to the single call (sparse_mode=3's bottom-right
    offset = S_k-S_q = 0 in both); torch.split/cat autograd backward
    is a single contiguous buffer per side, no per-chunk zero-padding sums.
    """
    cu = [int(v) for v in cu_seq_lens_q]  # [0, s1, ..., sN(=padded total)]
    # Greedily close a chunk at sample ends (v > bounds[-1] avoids
    # empty chunks from duplicate cu values)
    bounds = [0]
    for v in cu[1:]:
        if v > bounds[-1] and v - bounds[-1] >= chunk_tokens:
            bounds.append(v)
    if bounds[-1] != cu[-1]:
        bounds.append(cu[-1])
    sizes = [bounds[i + 1] - bounds[i] for i in range(len(bounds) - 1)]
    q_parts = torch.split(query, sizes, dim=0)
    k_parts = torch.split(key, sizes, dim=0)
    v_parts = torch.split(value, sizes, dim=0)
    # Preallocate the full output + per-chunk in-place copy_: NOT
    # torch.cat(outs) -- at the end cat would hold [all chunk outputs]
    # + [the concat result] simultaneously (~4 GiB each at 1M pack), one
    # whole extra output vs the old single call. With
    # preallocation the peak = full output + one chunk (~1/16), and each
    # chunk's kernel output is released right after its copy_. copy_ is
    # pure data movement; for disjoint slices, autograd routes each
    # slice's gradient back to its chunk correctly (equal to cat).
    attn_output = torch.empty(
        query.shape[0], query.shape[1], query.shape[2],
        dtype=query.dtype, device=query.device,
    )
    for i in range(len(sizes)):
        # Sample ends within this chunk (relative to chunk start);
        # the last chunk's final segment includes padding
        seg_ends = [v - bounds[i] for v in cu if bounds[i] < v <= bounds[i + 1]]
        cu_chunk = [0] + seg_ends
        chunk_outputs = torch_npu.npu_fusion_attention(
            q_parts[i],
            k_parts[i],
            v_parts[i],
            q_head_num,
            layout,
            pse=None,
            padding_mask=None,
            atten_mask=attention_mask,
            actual_seq_qlen=cu_chunk,
            actual_seq_kvlen=cu_chunk,
            scale=scaling,
            keep_prob=1 - dropout,
            inner_precise=0,
            sparse_mode=sparse_mode,
        )[:2]
        if metric_sink is not None:
            # QK-Clip keeps the running per-head maximum across chunks.
            metric_sink(chunk_outputs[1], token_offset=bounds[i])
        attn_output.narrow(0, bounds[i], sizes[i]).copy_(chunk_outputs[0])
        del chunk_outputs
    return attn_output


def flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    cu_seq_lens_q: Optional[Union[torch.LongTensor, List[int]]] = None,
    cu_seq_lens_k: Optional[Union[torch.LongTensor, List[int]]] = None,
    input_layout: str = "1NTD",  # Input layout for query, key, and value.
    # Only effective on NPU when attn_implementation is 'flash_attention_2'.
    # Valid options: BNSD, BSND, 1NTD, 1TND.
    ring_in_bnsd: bool = False,  # do_ring_attention input_layout is sbh, calculate in bnsd layout, only used for ring cp or hybrid cp.
    skip_ulysses: bool = False,  # Skip ulysses for some ViT cases like internvl3.5
    total_seq_len: int = None,  # unaligned cp need this
    seq_split_lens: torch.Tensor = None,  # unaligned cp need this
    skip_flash_attn_recompute: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # Notice: Device & Implementation Detection
    # Detect if we are running on NPU with Flash Attention 2/3 implementation.
    use_npu_fusion_fa = IS_NPU_AVAILABLE and module.config._attn_implementation in [
        "flash_attention_2",
        "flash_attention_3",
    ]

    # Input Preprocessing (NPU Specific)
    if use_npu_fusion_fa:
        # Normalize layout and validate supported formats.
        input_layout = input_layout.upper()
        if input_layout not in ["BSND", "BNSD", "1TND", "1NTD"]:
            raise RuntimeError(
                f"Flash Attention input layout only supports `BSND`, `BNSD`, `1TND`, and `1NTD`. " f"Got {input_layout}"
            )

        # Convert cumulative sequence lengths to list format for NPU API.
        # (For ring cp, cu_seqlens info in seq_split_lens)
        if not dist.is_initialized() or not (get_parallel_state().is_ring_enable() and seq_split_lens is not None):
            cu_seq_lens_q, cu_seq_lens_k, input_layout = _convert_cu_seq_lens(cu_seq_lens_q, cu_seq_lens_k, input_layout=input_layout)

        # Determine sequence length based on layout.
        if input_layout in ["BSND", "1TND"]:
            seq_len = query.shape[1]
        elif input_layout in ["BNSD", "1NTD"]:
            seq_len = query.shape[2]
    else:
        # This is before the transpose
        seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )

    if not use_npu_fusion_fa:
        # FA2 uses non-transposed inputs
        query = query.transpose(1, 2)  # bsnd or 1tnd
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = kwargs.pop("is_causal", None)
    if is_causal is None:
        is_causal = module.is_causal

    # Modification: ============= CONTEXT PARALLEL (CP) =============
    ps = get_parallel_state()
    is_ulysses_enabled = ps.is_ulysses_enable() if dist.is_initialized() else False
    is_ring_enabled = ps.is_ring_enable() if dist.is_initialized() else False

    if use_npu_fusion_fa:
        if input_layout in ["BSND", "1TND"]:
            head_dim_index = 2
            seq_dim_index = 1
        elif input_layout in ["BNSD", "1NTD"]:
            head_dim_index = 1
            seq_dim_index = 2
    else:
        head_dim_index = 2
        seq_dim_index = 1
    q_head_num = query.shape[head_dim_index]
    kv_head_num = key.shape[head_dim_index]

    # ulysses validation
    if is_ulysses_enabled:
        ulysses_size = ps.get_ulysses_group_size()
        if q_head_num % ulysses_size != 0:
            raise ValueError(f"num_query_heads ({q_head_num}) must be divisible by ulysses_size ({ulysses_size})")
        if ulysses_size > kv_head_num:
            if ulysses_size % kv_head_num != 0:
                raise ValueError(
                    f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})"
                )
            n_repeat = ulysses_size // kv_head_num
            # Shape before: (batch_size, seq_len, kv_head_num, head_dim)
            # This repeats the K/V heads (dim 2) to match the ulysses_size (SP world size)
            # Shape after: (batch_size, seq_len, kv_head_num * n_repeat, head_dim) where (kv_head_num * n_repeat) == ulysses_size
            key = torch.repeat_interleave(key, dim=head_dim_index, repeats=n_repeat)
            value = torch.repeat_interleave(value, dim=head_dim_index, repeats=n_repeat)

    if seq_split_lens is not None:
        if not isinstance(seq_split_lens, torch.Tensor):
            raise ValueError(f"seq_split_lens should be instance of torch.Tensor, bug got {type(seq_split_lens)}")
        if seq_split_lens.ndim != 1 and seq_split_lens.ndim != 2:
            raise ValueError(
                f"seq_split_lens should be a 1-dimensional tensor or a 2-dimensional tensor, bug got {seq_split_lens.shape}"
            )

    if is_ring_enabled:
        if not IS_NPU_AVAILABLE:
            raise ValueError(f"Ring Attention now only support in NPU.")
        # The ring metric sink cannot drop padded query rows.
        if getattr(module, "_qk_clip_enabled", False) and module.training and _padding_mask(attention_mask, query.shape[0]) is not None:
            raise NotImplementedError("QK-Clip does not support padded batches with ring attention")

        if skip_flash_attn_recompute:
            raise ValueError(f"Not support skip_flash_attn_recompute for ring attention.")

        if use_npu_fusion_fa:
            if input_layout in ["1TND", "1NTD"]:
                ring_fa_layout = "TND"
            else:
                ring_fa_layout = "SBH"
        else:
            raise RuntimeError(f"Ring Attention now only support when _attn_implementation is flash_attention_2")

        # Validate tensor layout for Ring Attention
        # For TND format, ensure input is [1, n, t, d] where t = seq_len * batch_size
        if ring_fa_layout.upper() == "TND" and query.shape[0] != 1:
            raise ValueError(
                f"When Ring Attention's fa layout is `TND`, input format should be [1, n, t, d], which t equals seq_len * batch_size."
            )

        # For causal attention, Ring Attention doesn't need mask
        if is_causal:
            attention_mask = None

        # Split attention mask across ring groups
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:  # [S_q, S_k]
                seq_dim = 0
            elif len(attention_mask.shape) == 3:  # [B, S_q, S_k]
                seq_dim = 1
            else:  # [B, 1, S_q, S_k]
                seq_dim = 2

            mask_row = attention_mask.chunk(ps.get_ring_group_size(), dim=seq_dim)[ps.get_ring_rank()].contiguous()
            attention_mask = [m.contiguous() for m in mask_row.chunk(ps.get_ring_group_size(), dim=seq_dim + 1)]

        if is_ulysses_enabled:
            # Calculate sequence length per ring rank
            if seq_split_lens is not None:
                if seq_split_lens.ndim == 1:
                    # For 1D seq_split_lens: directly get the sequence length for this ring rank
                    seq_len_this_ring_rank = seq_split_lens[ps.get_ring_rank()]
                else:
                    # For 2D seq_split_lens: sum the elements for this ring rank
                    seq_len_this_ring_rank = seq_split_lens[ps.get_ring_rank()].sum()
            elif total_seq_len is not None:
                # Calculate split sizes based on total sequence length and ring group size, then get this ring's portion
                seq_len_this_ring_rank = cal_split_sizes(total_seq_len, ps.get_ring_group_size())[ps.get_ring_rank()]
            else:
                seq_len_this_ring_rank = None

            # ulysses a2a
            query = all_to_all(
                query,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )
            key = all_to_all(
                key,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )
            value = all_to_all(
                value,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=seq_len_this_ring_rank,
            )

            # Update number of query heads after all-to-all
            q_head_num = q_head_num // ps.get_ulysses_group_size()

        # ring attention only support input layout: TND or SBH
        if ring_fa_layout.upper() == "TND":
            if input_layout in ["1NTD"]:
                # layout: 1NTD -> 1TND
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            # layout：1TND -> TND
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            if input_layout in ["BSND"]:
                # layout：BSND -> SBH
                query = rearrange(query, "B S N D -> S B (N D)")
                key = rearrange(key, "B S N D -> S B (N D)")
                value = rearrange(value, "B S N D -> S B (N D)")
            else:
                # layout：BNSD -> SBH
                query = rearrange(query, "B N S D -> S B (N D)")
                key = rearrange(key, "B N S D -> S B (N D)")
                value = rearrange(value, "B N S D -> S B (N D)")

        ring_cal_layout = "BNSD" if ring_in_bnsd else ring_fa_layout
        if ring_fa_layout == "TND" and ring_cal_layout == "BNSD":
            raise NotImplementedError(f"Ring attention calculated in bnsd not support packing data")

        qk_clip_valid_tokens = None
        if (ring_fa_layout == "TND" and seq_split_lens is not None
                and getattr(module, "_qk_clip_enabled", False) and module.training):
            qk_clip_valid_tokens = int(seq_split_lens[ps.get_ring_rank()].sum())

        # ring attention calculate
        attn_output = do_ring_attention(
            query,
            key,
            value,
            q_head_num,
            softmax_scale=scaling,
            is_causal=is_causal,
            fa_layout=ring_cal_layout,
            attn_mask=attention_mask,
            dropout_p=dropout,
            seq_split_lens=seq_split_lens,
            metric_sink=_make_qk_clip_metric_sink(
                module, valid_tokens=qk_clip_valid_tokens, gather_ulysses=is_ulysses_enabled
            ),
        )  # Output in sbh or tnd layout

        # Convert back to original layout: BSND or 1TND
        if ring_fa_layout.upper() == "TND":
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = rearrange(attn_output, "S B (N D) -> B S N D", N=q_head_num)

        # usp
        if is_ulysses_enabled:
            attn_output = all_to_all(attn_output, ps.get_ulysses_group(), scatter_dim=1, gather_dim=2)

        return attn_output, None

    else:
        # KV AllGather applies to the (causal) text full-attention layers, whose input
        # is cp-split.  Vision attention is non-causal and its packed data is split over
        # the ulysses group only (packed_data_*_with_cp) -- it is replicated across the
        # kvallgather group and gathered back to full before merging into the text stream,
        # so it stays on the plain Ulysses path below.
        is_kvallgather_enabled = ps.is_kvallgather_enable() if dist.is_initialized() else False
        is_kvag = is_kvallgather_enabled and not skip_ulysses and is_causal
        # packed varlen under kvag: gather-full-Q exact mode (set by kv_all_gather_attention_qkv)
        _kvag_varlen = False
        if is_kvag:
            (query, key, value, seq_len, q_head_num,
             kv_group, _kv_split_sizes, _kvag_varlen,
             cu_seq_lens_q, cu_seq_lens_k) = kv_all_gather_attention_qkv(
                query, key, value,
                head_dim_index=head_dim_index,
                seq_dim_index=seq_dim_index,
                total_seq_len=total_seq_len,
                is_causal=is_causal,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
            )
        elif is_ulysses_enabled and not skip_ulysses:
            # ulysses a2a
            query = all_to_all(
                query,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )
            key = all_to_all(
                key,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )
            value = all_to_all(
                value,
                ps.get_ulysses_group(),
                scatter_dim=head_dim_index,
                gather_dim=seq_dim_index,
                gather_size=total_seq_len,
            )

            # Only after all_to_all we got the full seq_len
            seq_len = query.shape[seq_dim_index]
            q_head_num = query.shape[head_dim_index]

        if use_npu_fusion_fa:
            if input_layout in ["BNSD", "BSND"]:
                layout = input_layout
            elif input_layout in ["1TND"]:
                layout = "TND"
            elif input_layout in ["1NTD"]:
                layout = "TND"
                # layout: 1NTD -> 1TND
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

            if input_layout in ["1TND", "1NTD"]:
                query = query.squeeze(0)
                key = key.squeeze(0)
                value = value.squeeze(0)

            # The kernel never applies a [batch, seq] mask as padding, so
            # QK-Clip uses it to drop padded query rows. With a KV
            # cache it also covers earlier keys; the queries are its last columns.
            # Local-Q KV-all-gather cannot map the full padding mask.
            padding_mask = _padding_mask(attention_mask, 1 if layout == "TND" else query.shape[0])
            if (padding_mask is not None and is_kvag and not _kvag_varlen
                    and padding_mask.size(1) != seq_len
                    and getattr(module, "_qk_clip_enabled", False) and module.training):
                raise NotImplementedError("QK-Clip does not support padded local-query KV-all-gather attention")
            if padding_mask is not None and not is_kvag:
                padding_mask = padding_mask[:, -seq_len:]

            # Only causal attention uses the fixed-size compressed mask.
            if is_causal:
                if attention_mask is None or tuple(attention_mask.shape) != ATTN_MASK_SHAPE:
                    attention_mask = get_attn_mask(device=query.device)
            elif attention_mask is not None and (
                attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]
            ):
                attention_mask = None

            # QK-Clip: record the maximum attention logit of each head, excluding
            # padded query rows. Packed TND inputs also exclude rows after the last sequence.
            # This runs after kv_all_gather_attention_qkv, so cu_seq_lens_q
            # already holds the bounds of the query this rank attends with. Both
            # the KV all-gather path and the Ulysses path split heads across the
            # Ulysses group, so local heads are placed in a full-width vector.
            qk_clip_valid_tokens = None
            if layout == "TND" and cu_seq_lens_q:
                qk_clip_valid_tokens = int(cu_seq_lens_q[-1])
            metric_sink = _make_qk_clip_metric_sink(
                module,
                valid_tokens=qk_clip_valid_tokens,
                gather_ulysses=is_kvag or (is_ulysses_enabled and not skip_ulysses),
                padding_mask=padding_mask,
                cu_seq_lens_q=cu_seq_lens_q if layout == "TND" else None,
            )

            # features.fa_varlen_chunk_tokens (default 0 = off): per-sub-call token budget for
            # splitting the packed-varlen full-sequence FA into sample-group sub-calls, bounding
            # the backward grad workspace (scales with full Q length). See _npu_fa_chunked_varlen_attention.
            _varlen_chunk_tokens = kwargs.get("fa_varlen_chunk_tokens", 0)
            _fa_chunk_active = (
                _kvag_varlen
                and not skip_flash_attn_recompute
                and _varlen_chunk_tokens > 0
                and cu_seq_lens_q is not None
                # The chunked impl below is written for the 3-D TND layout
                # (torch.split/narrow on dim=0 = time). Other layouts would split
                # the batch/head axis instead -- fall back to the single full call.
                and input_layout == "1TND"
                # Chunking requires identical q/k segment bounds (the premise that
                # in-chunk K/V share Q's range). The kvag packed path currently
                # passes the same cu for q/k; a mismatch means an unexpected call
                # shape -- fall back to the single full call rather than compute wrong silently.
                and cu_seq_lens_q == cu_seq_lens_k
            )
            if _fa_chunk_active:
                attn_output = _npu_fa_chunked_varlen_attention(
                    query, key, value,
                    q_head_num=q_head_num,
                    layout=layout,
                    attention_mask=attention_mask,
                    cu_seq_lens_q=cu_seq_lens_q,
                    scaling=scaling,
                    dropout=dropout,
                    chunk_tokens=_varlen_chunk_tokens,
                    sparse_mode=3 if is_causal else 0,
                    metric_sink=metric_sink,
                )
            elif skip_flash_attn_recompute:
                attn_output = skip_recompute_flash_attention(
                    query,
                    key,
                    value,
                    q_head_num,
                    layout,
                    pse=None,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    actual_seq_qlen=cu_seq_lens_q,
                    actual_seq_kvlen=cu_seq_lens_k,
                    scale=scaling,
                    keep_prob=1 - dropout,
                    inner_precise=0,
                    sparse_mode=3 if is_causal else 0,
                    metric_sink=metric_sink,
                )
            else:
                attn_output, softmax_max = torch_npu.npu_fusion_attention(
                    query,
                    key,
                    value,
                    q_head_num,
                    layout,
                    pse=None,
                    padding_mask=None,
                    atten_mask=attention_mask,
                    actual_seq_qlen=cu_seq_lens_q,
                    actual_seq_kvlen=cu_seq_lens_k,
                    scale=scaling,
                    keep_prob=1 - dropout,
                    inner_precise=0,
                    sparse_mode=3 if is_causal else 0,
                )[:2]
                if metric_sink is not None:
                    metric_sink(softmax_max)
                del softmax_max

            if input_layout in ["1TND", "1NTD"]:
                attn_output = attn_output.unsqueeze(0)

        else:
            if skip_flash_attn_recompute:
                raise NotImplementedError(
                    "Skipping flash_attn recompute is only supported on NPU when `_attn_implementation` is `flash_attention_2`."
                )

            attn_output = _flash_attention_forward(
                query,
                key,
                value,
                attention_mask,
                query_length=seq_len,
                is_causal=is_causal,
                dropout=dropout,
                softmax_scale=scaling,
                sliding_window=sliding_window,
                softcap=softcap,
                use_top_left_mask=False,
                target_dtype=target_dtype,
                attn_implementation=module.config._attn_implementation,
                layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                **kwargs,
            )

        # Ulysses: attention a2a (inverse) -- restore full heads on the local sequence shard.
        # For KV AllGather local-Q, the output is already the local S/kv shard (Q was
        # local), so the inverse ulysses a2a alone yields the local S/cp shard.
        if is_kvag or (is_ulysses_enabled and not skip_ulysses):
            if use_npu_fusion_fa and input_layout in ["1NTD"]:
                # attn_output layout: TND
                seq_dim_index = 1
                head_dim_index = 2
            if _kvag_varlen:
                # packed varlen mode: output is the FULL sequence (replicated across the kv
                # group) — take this rank's Q block back out. Forward split / backward
                # all_gather (blocks disjoint => concat == sum; grad_scale=None — see above),
                # feeding Q's split-backward on the gather side.
                attn_output = split_forward_gather_backward(
                    attn_output, kv_group, dim=seq_dim_index,
                    split_sizes=_kv_split_sizes, grad_scale=None,
                )
            attn_output = all_to_all(
                attn_output, ps.get_ulysses_group(), scatter_dim=seq_dim_index, gather_dim=head_dim_index
            )

        if use_npu_fusion_fa:
            if input_layout in ["BNSD"]:
                attn_output = attn_output.transpose(1, 2)

        return attn_output, None


def apply_transformers_attention_patch():
    # ============= REGISTER CP-ENABLED FLASH ATTENTION IMPLEMENTATIONS =============
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_3", flash_attention_forward)

    global _flash_attention_forward
    _flash_attention_forward = transformers_flash_attention_forward
