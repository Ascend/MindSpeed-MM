from typing import Optional, List

import torch
from torch.nn import functional as F
from torch_npu import npu_fusion_attention


def get_attention_func(name: str):
    if name in ALL_ATTENTION_FUNCTIONS:
        return ALL_ATTENTION_FUNCTIONS[name]
    else:
        raise NotImplementedError(f"Unrecognized attention function: {name}")


def verify_attn_layout(name: str, layout: str):
    if name not in ALL_ATTENTION_LAYOUT:
        raise NotImplementedError(f"Unrecognized attention function: {name}")
    if layout not in ALL_ATTENTION_LAYOUT[name]:
        raise NotImplementedError(f"Unsupported layout: {layout}, {name} attention only support {ALL_ATTENTION_LAYOUT[name]}")
    return ALL_ATTENTION_LAYOUT[name]


def pad_out(hidden_states, indices, batch_size, seqlen):
    dim = hidden_states.shape[1:]
    output = torch.zeros((batch_size * seqlen), *dim, device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output


ATTN_MASK_NPU_CACHE = {}


def get_attn_mask_npu(device, seq_len=2048):
    """Get or create NPU attention mask"""
    if device not in ATTN_MASK_NPU_CACHE:
        ATTN_MASK_NPU_CACHE[device] = torch.triu(torch.ones([seq_len, seq_len], device=device), diagonal=1).bool()
    return ATTN_MASK_NPU_CACHE[device]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
    **kwargs,
    ):
    if enable_gqa:
        k = repeat_kv(k, q.shape[1] // k.shape[1])
        v = repeat_kv(v, q.shape[1] // v.shape[1])

    is_causal = q.shape[2] > 1 and attention_mask is None and is_causal
    if is_causal:
        batch_size, _, seq_len, _ = q.shape
        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bool, device=q.device).tril(diagonal=0)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * scale
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : k.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output


def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str,
    attention_mask: Optional[torch.Tensor] = None,
    dropout: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    **kwargs,
    ):
    if layout == "TND":
        return varlen_fa_forward(q, k, v, scale=scale, dropout=dropout, is_causal=is_causal, **kwargs)

    keep_prob = 1.0 - dropout
    head_num = q.shape[1] if layout == "BNSD" else q.shape[2]
    if not is_causal:
        attn_output = npu_fusion_attention(q, k, v, head_num, input_layout=layout, atten_mask=attention_mask, keep_prob=keep_prob, scale=scale)[0]
    else:
        attn_mask_npu = get_attn_mask_npu(q.device)
        attn_output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            input_layout=layout,
            atten_mask=attn_mask_npu,
            keep_prob=keep_prob,
            scale=scale,
            sparse_mode=3,
        )[0]

    return attn_output


def varlen_fa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    actual_seq_qlen: Optional[List] = None,
    actual_seq_kvlen: Optional[List] = None,
    scale: float = None,
    dropout: float = 0.0,
    is_causal: bool = False,
    **kwargs,
    ):
    keep_prob = 1.0 - dropout
    head_num = q.shape[1]

    if not is_causal:
        attn_output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            atten_mask=None,
            scale=scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_kvlen,
        )[0]
    else:
        attn_mask_npu = get_attn_mask_npu(q.device)
        attn_output = npu_fusion_attention(
            q,
            k,
            v,
            head_num,
            pse=None,
            padding_mask=None,
            atten_mask=attn_mask_npu,
            scale=scale,
            keep_prob=keep_prob,
            input_layout="TND",
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_kvlen=actual_seq_qlen,
            sparse_mode=3,
        )[0]

    return attn_output


def sdpa_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scale: Optional[float] = None,
    is_causal: bool = False,
    enable_gqa: bool = False,
    **kwargs,
    ):
    enable_gqa = enable_gqa and attention_mask is None
    if not enable_gqa and q.shape[1] != k.shape[1]:
        k = repeat_kv(k, q.shape[1] // k.shape[1])
        v = repeat_kv(v, q.shape[1] // v.shape[1])

    is_causal = q.shape[2] > 1 and attention_mask is None and is_causal
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : k.shape[-2]]

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scale,
        is_causal=is_causal,
        enable_gqa=enable_gqa,
    )

    return attn_output


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_forward,  # support BNSD
    "flash_attention_2": flash_attention_forward,  # support BNSD, BSND, SBH, TND(need cu_seqs)
    "sdpa": sdpa_attention_forward,  # support BNSD
    "varlen_fa": varlen_fa_forward  # support TND
}

ALL_ATTENTION_LAYOUT = {
    "eager": ["BNSD"],
    "flash_attention_2": ["BNSD", "BSND", "SBH", "TND"],
    "sdpa": ["BNSD"],
    "varlen_fa": ["TND"]
}