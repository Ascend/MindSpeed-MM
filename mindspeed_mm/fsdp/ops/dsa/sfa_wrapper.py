# Copyright © 2026 Huawei Technologies Co., Ltd.
"""Sparse flash attention over the indexer's selected keys, for MLA-absorb DSA.

The dense path expands the top-k indices into a [B, 1, Q_S, KV_S] boolean mask
and hands that to FlashAttentionScore. Building the mask costs more than the
attention it feeds: measured at B=1 S=8192, 209.6 ms of scatter/gather/concat/
reduce against 131 ms of FA. npu_sparse_flash_attention consumes the indices.

The operator only accepts the absorbed MLA layout, verified against the wheel:
  qk_head_dim   must equal 512 (the kv_lora_rank), so the query has to carry
                W_UK already folded in and the key is the compressed latent
  kv_head_num   must be 1, i.e. one shared latent rather than expanded per-head
                K/V -- which also keeps sparse_indices at 67 MB instead of 4.3 GB
  rope          mandatory, and its head dim must be exactly 64
  attention_mode must be 2; the schema's default of 0 is rejected
  sparse_block_size must be 1 for token-level indices; larger values return
                inf/nan
Index order within a row does not matter (checked: 2.8e-07 between a permuted
and an unpermuted call), so the indices are passed through untouched.
"""

import torch
import torch_npu

# Fixed by the operator, not by this model.
SFA_ROPE_DIM = 64
SFA_QK_HEAD_DIM = 512
SFA_SPARSE_BLOCK_SIZE = 1
SFA_ATTENTION_MODE = 2
# rightDownCausal: the causal triangle anchored at the bottom right.
SFA_SPARSE_MODE_CAUSAL = 3

try:
    _SFA = torch_npu.npu_sparse_flash_attention
    _SFA_GRAD = torch_npu.npu_sparse_flash_attention_grad
except AttributeError as exc:  # pragma: no cover - depends on the wheel
    raise ImportError(
        "torch_npu does not expose npu_sparse_flash_attention(_grad). Keep "
        "dsa_implementation on the dense path."
    ) from exc


def make_rope_pair(query, key, rope_head_dim, query_rope=None, key_rope=None):
    """Return the (query_rope, key_rope) pair the operator requires.

    A model with real positional Q/K channels passes them through; one without
    gets zeros, which contribute nothing to the q.k score and so leave the
    attention unchanged. Zeroing is only ever applied to the no-rope case -- a
    model that has rope of an unsupported width must fail rather than lose it.
    """
    if rope_head_dim == 0:
        return (query.new_zeros((*query.shape[:-1], SFA_ROPE_DIM)),
                key.new_zeros((*key.shape[:-1], SFA_ROPE_DIM)))
    if query_rope is None or key_rope is None:
        raise ValueError(f"rope_head_dim={rope_head_dim} but no rope tensors were given.")
    if query_rope.shape[-1] != SFA_ROPE_DIM or key_rope.shape[-1] != SFA_ROPE_DIM:
        raise NotImplementedError(
            f"npu_sparse_flash_attention requires rope head dim {SFA_ROPE_DIM}, got "
            f"query {query_rope.shape[-1]} / key {key_rope.shape[-1]}."
        )
    return query_rope.contiguous(), key_rope.contiguous()


class SparseFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, sparse_indices, query_rope, key_rope,
                scale, actual_seq_qlen, actual_seq_kvlen, sparse_mode):
        out, softmax_max, softmax_sum, *_ = _SFA(
            query, key, value,
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
        ctx.save_for_backward(query, key, value, sparse_indices, query_rope,
                              key_rope, out, softmax_max, softmax_sum)
        ctx.scale = scale
        ctx.sparse_mode = sparse_mode
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        return out

    @staticmethod
    def backward(ctx, d_out):
        (query, key, value, sparse_indices, query_rope, key_rope,
         out, softmax_max, softmax_sum) = ctx.saved_tensors
        dq, dk, dv, dq_rope, dk_rope = _SFA_GRAD(
            query, key, value, sparse_indices, d_out.contiguous(), out,
            softmax_max, softmax_sum, float(ctx.scale), SFA_SPARSE_BLOCK_SIZE,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=ctx.actual_seq_qlen,
            actual_seq_kvlen=ctx.actual_seq_kvlen,
            layout="BSND",
            sparse_mode=int(ctx.sparse_mode),
            attention_mode=SFA_ATTENTION_MODE,
        )
        # One gradient per positional argument of apply(). The rope gradients are
        # dropped for a no-rope model because its rope tensors are constants.
        return (dq.to(query), dk.to(key), dv.to(value), None,
                dq_rope.to(query_rope), dk_rope.to(key_rope),
                None, None, None, None)


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
    """Attention over the indexer's selected keys, with no dense mask.

    query is the absorbed query [B, S, N, kv_lora_rank]; key and value are the
    shared compressed latent [B, S, 1, kv_lora_rank]. topk_indices is the
    indexer's [B, S, topk], with -1 marking unused slots.
    """
    if query.shape[-1] != SFA_QK_HEAD_DIM:
        raise NotImplementedError(
            f"The operator only accepts qk_head_dim={SFA_QK_HEAD_DIM} (the absorbed "
            f"MLA layout), got {query.shape[-1]}. W_UK must be folded into the query."
        )
    if key.shape[2] != 1 or value.shape[2] != 1:
        raise NotImplementedError(
            f"The operator only accepts kv_head_num=1, got key {key.shape[2]} / "
            f"value {value.shape[2]}. Pass the shared latent, not expanded K/V."
        )

    query, key, value = (t.contiguous() for t in (query, key, value))
    query_rope, key_rope = make_rope_pair(query, key, rope_head_dim, query_rope, key_rope)

    batch, seq_len = query.shape[0], query.shape[1]
    # Always supplied: the reference adaptation passes these even for a dense
    # batch, and leaving them None makes the operator return something that
    # matches no reference formula.
    actual_seq_qlen = torch.full((batch,), seq_len, dtype=torch.int32, device=query.device)

    out = SparseFlashAttentionFunction.apply(
        query, key, value,
        topk_indices.unsqueeze(2).to(torch.int32).contiguous(),
        query_rope, key_rope, scale,
        actual_seq_qlen, actual_seq_qlen, sparse_mode,
    )
    return out
