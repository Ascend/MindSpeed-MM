import math
import torch
import torch_npu

from mindspeed_mm.models.predictor.dits.hunyuanvideo15.ssta_attention import ssta_3d_attention
from tests.ut.utils import judge_expression


def _build_sparse_block_mask(batch_size, num_heads, ceil_q, ceil_kv, device):
    q_block_idx = torch.arange(ceil_q, device=device).view(ceil_q, 1)
    kv_block_idx = torch.arange(ceil_kv, device=device).view(1, ceil_kv)
    base_mask = (q_block_idx - kv_block_idx).abs() <= 1
    block_sparse_mask = base_mask.view(1, 1, ceil_q, ceil_kv)
    block_sparse_mask = block_sparse_mask.expand(batch_size, num_heads, -1, -1).contiguous()

    return block_sparse_mask


def _npu_fusion_attention_with_expanded_block_mask(
    query, key, value, num_heads, seq_len, block_m, block_n, scale_value, block_sparse_mask
):
    block_full_mask = torch.tile(
        block_sparse_mask[:, :, :, None, :, None],
        (1, 1, 1, block_m, 1, block_n),
    )
    block_full_mask = block_full_mask.reshape(
        block_full_mask.shape[0],
        block_full_mask.shape[1],
        block_full_mask.shape[2] * block_m,
        block_full_mask.shape[4] * block_n,
    )[:, :, :seq_len, :seq_len]
    inverted_block_full_mask = ~block_full_mask

    fusion_out = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        num_heads,
        padding_mask=None,
        atten_mask=inverted_block_full_mask,
        scale=scale_value,
        keep_prob=1.0,
        input_layout="BNSD",
        pre_tockens=65535,
        next_tockens=65535,
        sparse_mode=0,
    )[0]
    return fusion_out


def test_ssta_3d_attention_shape():
    """test HunyuanVideo 1.5 SSTA function and output shape"""

    if not hasattr(torch_npu, "npu_block_sparse_attention"):
        return

    batch_size = 1
    num_heads = 2
    head_dim = 128
    canvas_thw = (4, 16, 16)
    tile_thw = (8, 4, 4)
    seq_len = math.prod(canvas_thw)

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")

    output, sparse_ratio = ssta_3d_attention(
        query,
        key,
        value,
        canvas_thw=canvas_thw,
        topk=2,
        tile_thw=tile_thw,
        kernel_thw=(1, 1, 1),
        sparse_type="ssta",
        sampling_type="similarity",
        mask_share_within_head=False,
    )

    judge_expression(output.shape == query.shape)
    judge_expression(output.dtype == query.dtype)
    judge_expression(0.0 < sparse_ratio < 1.0)


def test_npu_block_sparse_attention_bnsd():
    """test npu_block_sparse_attention BNSD forward against token-mask npu_fusion_attention"""

    if not hasattr(torch_npu, "npu_block_sparse_attention"):
        return

    batch_size = 1
    num_heads = 2
    seq_len = 1024
    head_dim = 128
    block_shape = [128, 128]
    scale_value = 1.0 / math.sqrt(head_dim)
    ceil_q = (seq_len + block_shape[0] - 1) // block_shape[0]
    ceil_kv = (seq_len + block_shape[1] - 1) // block_shape[1]

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")
    key = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")
    value = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device="npu")
    block_sparse_mask = _build_sparse_block_mask(batch_size, num_heads, ceil_q, ceil_kv, query.device)
    sparse_ratio = block_sparse_mask.float().mean().item()

    fa_out = _npu_fusion_attention_with_expanded_block_mask(
        query, key, value, num_heads, seq_len, block_shape[0], block_shape[1], scale_value, block_sparse_mask
    )

    npu_bsa_out, npu_lse = torch_npu.npu_block_sparse_attention(
        query,
        key,
        value,
        block_sparse_mask,
        block_shape,
        q_input_layout="BNSD",
        kv_input_layout="BNSD",
        num_key_value_heads=num_heads,
        scale_value=scale_value,
        inner_precise=0,
        actual_seq_lengths=[seq_len] * batch_size,
        actual_seq_lengths_kv=[seq_len] * batch_size,
        softmax_lse_flag=1,
    )

    judge_expression(torch.allclose(npu_bsa_out.float(), fa_out.float(), rtol=0.005, atol=0.001))
    judge_expression(npu_lse is not None)
    judge_expression(0.0 < sparse_ratio < 1.0)
