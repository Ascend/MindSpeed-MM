import torch
import torch_npu

from .sfa_wrapper import (
    SFA_ATTENTION_MODE,
    SFA_QK_HEAD_DIM,
    SFA_SPARSE_BLOCK_SIZE,
    SFA_SPARSE_MODE_CAUSAL,
    make_rope_pair,
)


class KimiSparseFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sparse_indices, query_rope, key_rope,
                valid_query, scale, actual_seq_qlen, actual_seq_kvlen, sparse_mode):
        out, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
            query,
            key,
            value,
            sparse_indices=sparse_indices,
            block_table=None,
            actual_seq_lengths_query=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            query_rope=query_rope,
            key_rope=key_rope,
            scale_value=float(scale),
            sparse_block_size=SFA_SPARSE_BLOCK_SIZE,
            layout_query="BSND",
            layout_kv="BSND",
            sparse_mode=int(sparse_mode),
            attention_mode=SFA_ATTENTION_MODE,
            return_softmax_lse=True,
        )
        out.mul_(valid_query)
        ctx.save_for_backward(
            query, key, value, sparse_indices, query_rope, key_rope,
            out, softmax_max, softmax_sum,
        )
        ctx.valid_query = valid_query
        ctx.scale = scale
        ctx.sparse_mode = sparse_mode
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        return out

    @staticmethod
    def backward(ctx, d_out):
        (query, key, value, sparse_indices, query_rope, key_rope,
         out, softmax_max, softmax_sum) = ctx.saved_tensors
        d_out = (d_out * ctx.valid_query).contiguous()
        dq, dk, dv, dq_rope, dk_rope = torch_npu.npu_sparse_flash_attention_grad(
            query,
            key,
            value,
            sparse_indices,
            d_out,
            out,
            softmax_max,
            softmax_sum,
            float(ctx.scale),
            SFA_SPARSE_BLOCK_SIZE,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
            layout="BSND",
            sparse_mode=int(ctx.sparse_mode),
            attention_mode=SFA_ATTENTION_MODE,
        )
        return (
            dq.to(query), dk.to(key), dv.to(value), None,
            dq_rope.to(query_rope), dk_rope.to(key_rope),
            None, None, None, None, None,
        )


def sparse_flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    topk_indices: torch.Tensor,
    scale: float,
    rope_head_dim: int = 0,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    sparse_mode: int = SFA_SPARSE_MODE_CAUSAL,
) -> torch.Tensor:
    """Run absorbed SFA with zero output and gradients for fully invalid query rows."""
    if query.shape[-1] != SFA_QK_HEAD_DIM:
        raise NotImplementedError(
            f"Kimi SFA requires qk_head_dim={SFA_QK_HEAD_DIM}, got {query.shape[-1]}."
        )
    if key.shape[2] != 1 or value.shape[2] != 1:
        raise NotImplementedError(
            f"Kimi SFA requires one shared latent head, got {key.shape[2]} / {value.shape[2]}."
        )

    query, key, value = (tensor.contiguous() for tensor in (query, key, value))
    query_rope, key_rope = make_rope_pair(
        query, key, rope_head_dim, query_rope, key_rope
    )
    batch, seq_len = query.shape[:2]
    actual_seq_lengths = torch.full(
        (batch,), seq_len, dtype=torch.int32, device=query.device
    )
    sparse_indices = topk_indices.unsqueeze(2).to(torch.int32).contiguous()
    valid_query = (sparse_indices >= 0).any(dim=-1, keepdim=True)
    query_positions = torch.arange(
        seq_len, dtype=torch.int32, device=query.device
    ).view(1, seq_len, 1, 1)
    sparse_indices = torch.where(valid_query, sparse_indices, query_positions)
    return KimiSparseFlashAttentionFunction.apply(
        query,
        key,
        value,
        sparse_indices,
        query_rope,
        key_rope,
        valid_query,
        scale,
        actual_seq_lengths,
        actual_seq_lengths,
        sparse_mode,
    )
