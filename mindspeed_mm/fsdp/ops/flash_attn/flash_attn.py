from typing import Optional
import logging

from einops import rearrange
import torch
from transformers.modeling_flash_attention_utils import (
    _flash_attention_forward as _transformers_flash_attention_forward,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, flash_attention_mask

from ...distributed.parallel_state import get_parallel_state
from ...utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ...distributed.context_parallel.utils import cal_split_sizes
from ...distributed.context_parallel.communication import all_to_all

if IS_NPU_AVAILABLE:
    from ...distributed.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel, ringattn_context_parallel_tnd_general



logger = logging.getLogger(__name__)
_flash_attention_forward = None


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
    is_caual,
    fa_layout="SBH",
    attn_mask=None,
    dropout_p=0.,
    seq_split_lens: Optional[list[int] | torch.Tensor] = None
):
    ps = get_parallel_state()
    cp_group = ps.get_ring_group()
    cp_size = ps.get_ring_group_size()
    rank = ps.get_ring_rank()
    cp_global_ranks = ps.get_ring_device_mesh().mesh.tolist()

    cp_para = dict()

    cp_para['causal'] = is_caual
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank

    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = None
    cp_para['megatron_cp_in_bnsd'] = fa_layout.upper() == "BNSD"

    if is_caual or fa_layout.upper() == "SBH" or fa_layout.upper() == "BNSD":
        # 输入shapes是一维list
        if seq_split_lens is not None:
            seq_split_lens = seq_split_lens.cpu().tolist()
        output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens)
    elif fa_layout.upper() == "TND":
        # 输入shapes是二维tensor
        output = ringattn_context_parallel_tnd_general(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, shapes=seq_split_lens)
    else:
        raise ValueError(f"Ring Attention only support fa layout: `SBH`、`SBND` and `TND`, bug got {fa_layout.upper()}.")

    return output


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
    skip_ulysses: bool = False, # Skip ulysses for some ViT cases like internvl3.5
    ring_fa_layout: str = None, # activated only when using Ring Attention, supported layout: TND/SBH/BNSD
    total_seq_len: int = None, # unaligned cp need this
    seq_split_lens: torch.Tensor = None, # unaligned cp need this
    **kwargs,
) -> tuple[torch.Tensor, None]:
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`."
            " Please set your attention to `eager` if you want any of these features."
        )

    # This is before the transpose
    seq_len = query.shape[2]

    if any(dim == 0 for dim in query.shape):
        raise ValueError(
            "Tensor query has shape  with a zero dimension.\n"
            "FlashAttention does not support inputs with dim=0.\n"
            "Please check your input shapes or use SDPA instead."
        )
    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)# bsnd or 1tnd
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
    is_ulysses_enabled = ps.is_ulysses_enable()
    is_ring_enabled = ps.is_ring_enable()
    q_head_num = query.shape[2]
    kv_head_num = key.shape[2]
    
    # ulysses validation
    if is_ulysses_enabled:
        ulysses_size = ps.get_ulysses_group_size()
        if q_head_num % ulysses_size != 0:
            raise ValueError(f"num_query_heads ({q_head_num}) must be divisible by ulysses_size ({ulysses_size})")
        if ulysses_size > kv_head_num:
            if ulysses_size % kv_head_num != 0:
                raise ValueError(f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({kv_head_num})")
            n_repeat = ulysses_size // kv_head_num
            # Shape before: (batch_size, seq_len, kv_head_num, head_dim)
            # This repeats the K/V heads (dim 2) to match the ulysses_size (SP world size)
            # Shape after: (batch_size, seq_len, kv_head_num * n_repeat, head_dim) where (kv_head_num * n_repeat) == ulysses_size
            key = torch.repeat_interleave(key, dim=2, repeats=n_repeat)
            value = torch.repeat_interleave(value, dim=2, repeats=n_repeat)
    
    if seq_split_lens is not None:
        if not isinstance(seq_split_lens, torch.Tensor):
            raise ValueError(f"seq_split_lens should be instance of torch.Tensor, bug got {type(seq_split_lens)}")
        if seq_split_lens.ndim != 1 and seq_split_lens.ndim != 2:
            raise ValueError(f"seq_split_lens should be a 1-dimensional tensor or a 2-dimensional tensor, bug got {seq_split_lens.shape}")

    if is_ring_enabled:
        if not IS_NPU_AVAILABLE:
            raise ValueError(f"Ring Attention now only support in NPU.")
        # Validate tensor layout for Ring Attention
        # For TND format, ensure input is [1, n, t, d] where t = seq_len * batch_size
        if ring_fa_layout.upper() == "TND" and query.shape[0] != 1:
            raise ValueError(f"When Ring Attention's fa layout is `TND`, input format should be [1, n, t, d], which t equals seq_len * batch_size.")
        
        # For causal attention, Ring Attention doesn't need mask
        if is_causal:
            attention_mask = None

        # Split attention mask across ring groups
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:# [S_q, S_k]
                seq_dim = 0
            elif len(attention_mask.shape) == 3:# [B, S_q, S_k]
                seq_dim = 1
            else:# [B, 1, S_q, S_k]
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
            query = all_to_all(query, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=seq_len_this_ring_rank)
            key = all_to_all(key, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=seq_len_this_ring_rank)
            value = all_to_all(value, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=seq_len_this_ring_rank)

            # Update number of query heads after all-to-all
            q_head_num = q_head_num // ps.get_ulysses_group_size()
        
        # ring attention only support input layout: TND or SBH
        if ring_fa_layout.upper() == "TND":
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            query = rearrange(query, "B S N D -> S B (N D)")
            key = rearrange(key, "B S N D -> S B (N D)")
            value = rearrange(value, "B S N D -> S B (N D)")

        # ring attention calculate
        attn_output = do_ring_attention(
            query,
            key,
            value,
            q_head_num,
            softmax_scale=scaling,
            is_caual=is_causal,
            fa_layout=ring_fa_layout,
            attn_mask=attention_mask,
            dropout_p=dropout,
            seq_split_lens=seq_split_lens
        )# Output in sbh or tnd layout

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
        if is_ulysses_enabled and not skip_ulysses:
            # ulysses a2a
            query = all_to_all(query, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=total_seq_len)
            key = all_to_all(key, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=total_seq_len)
            value = all_to_all(value, ps.get_ulysses_group(), scatter_dim=2, gather_dim=1, gather_size=total_seq_len)

            # Only after all_to_all we got the full seq_len
            seq_len = query.shape[1]

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
            **kwargs,
        )

        # Ulysses: attention a2a
        if is_ulysses_enabled and not skip_ulysses:
            attn_output = all_to_all(attn_output, ps.get_ulysses_group(), scatter_dim=1, gather_dim=2)

        return attn_output, None


def apply_transformers_attention_patch():
    # ============= REGISTER CP-ENABLED FLASH ATTENTION IMPLEMENTATIONS =============
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)
    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", flash_attention_forward)

    global _flash_attention_forward
    _flash_attention_forward = transformers_flash_attention_forward
