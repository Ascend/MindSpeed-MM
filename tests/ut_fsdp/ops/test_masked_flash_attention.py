# Copyright © 2026 Huawei Technologies Co., Ltd.
"""CPU unit tests for masked_flash_attention pool-ID mask expansion and Glm5NextTextIndexer."""

import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mindspeed_mm.fsdp.ops.dsa.masked_flash_attention import (  # noqa: E402
    PoolSelection,
    build_mask,
    build_mask_from_pools,
    masked_flash_attention,
)


def _reference_build_mask_from_pools(
    selected_pools: torch.Tensor,
    selected_valid: torch.Tensor,
    kv_length: int,
    num_pools: int,
    always_select_tail: bool,
    valid_lengths: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len, k = selected_pools.shape
    device = selected_pools.device
    q = torch.arange(seq_len, dtype=torch.int64, device=device)[None, :]
    q_valid = (
        q < valid_lengths.to(torch.int64)[:, None]
        if valid_lengths is not None
        else torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    )
    visible = torch.where(q_valid, q + 1, 0)
    full_pools = (visible // 4).clamp(max=num_pools)
    pool_ok = (
        q_valid.unsqueeze(-1) & selected_valid & (selected_pools >= 0) & (selected_pools < full_pools.unsqueeze(-1))
    )
    expanded = selected_pools.long().unsqueeze(-1) * 4 + torch.arange(4, dtype=torch.int64, device=device)
    expanded = torch.where(pool_ok.unsqueeze(-1), expanded, -1).flatten(-2)

    if always_select_tail:
        tail_count = visible % 4
        tail_off = torch.arange(3, dtype=torch.int64, device=device)
        tail_idx = (q - tail_count + 1).unsqueeze(-1) + tail_off
        tail_tok = torch.where(
            q_valid.unsqueeze(-1) & (tail_off[None, None, :] < tail_count.unsqueeze(-1)) & (tail_idx < kv_length),
            tail_idx,
            -1,
        )
        ref_topk = torch.cat([expanded, tail_tok], dim=-1) if k > 0 else tail_tok
    else:
        ref_topk = expanded

    ref_topk = torch.where(q_valid.unsqueeze(-1), ref_topk, -1)
    return build_mask(ref_topk, kv_length)


def _assert_matches_reference(
    selected_pools: torch.Tensor,
    selected_valid: torch.Tensor,
    kv_length: int,
    num_pools: int,
    always_select_tail: bool,
    valid_lengths: torch.Tensor | None = None,
):
    got_mask, got_nonempty = build_mask_from_pools(
        selected_pools, selected_valid, kv_length, num_pools, always_select_tail, valid_lengths=valid_lengths
    )
    ref_mask, ref_nonempty = _reference_build_mask_from_pools(
        selected_pools, selected_valid, kv_length, num_pools, always_select_tail, valid_lengths=valid_lengths
    )
    assert got_mask.dtype == torch.uint8 and got_nonempty.dtype == torch.bool
    assert got_mask.shape == (selected_pools.shape[0], 1, selected_pools.shape[1], kv_length)
    assert torch.equal(got_nonempty, ref_nonempty)
    assert torch.equal(got_mask, ref_mask)


@pytest.mark.parametrize(
    "batch,seq_len,k,always_select_tail",
    [(1, 4, 0, True), (2, 4, 1, True), (2, 16, 4, True), (2, 32, 8, False)],
)
@pytest.mark.parametrize("idx_dtype", [torch.int32, torch.int64])
def test_pool_mask_unpadded_and_tail_variants(batch, seq_len, k, always_select_tail, idx_dtype):
    num_pools = seq_len // 4
    gen = torch.Generator().manual_seed(100 + seq_len + k)
    selected_pools = torch.randint(-2, num_pools + 3, (batch, seq_len, k), generator=gen, dtype=idx_dtype)
    selected_valid = torch.randint(0, 4, (batch, seq_len, k), generator=gen) > 0
    if k > 0:
        selected_pools[0, seq_len - 1, 0] = 0
        if k > 1:
            selected_pools[0, seq_len - 1, 1] = 0
        selected_pools[0, 0, :] = 0
        selected_valid[0, 0, :] = False
    _assert_matches_reference(selected_pools, selected_valid, seq_len, num_pools, always_select_tail)


@pytest.mark.parametrize("always_select_tail", [True, False])
def test_pool_mask_right_padding_and_empty_rows(always_select_tail):
    seq_len = 16
    num_pools = seq_len // 4
    gen = torch.Generator().manual_seed(777)
    sel = torch.randint(-1, num_pools + 2, (3, seq_len, 4), generator=gen, dtype=torch.int32)
    val = torch.randint(0, 3, (3, seq_len, 4), generator=gen) > 0
    _assert_matches_reference(sel, val, seq_len, num_pools, always_select_tail, torch.tensor([7, 0, 13], dtype=torch.int32))

    empty_vl = torch.tensor([0, 0], dtype=torch.int32)
    for k_val in (0, 2):
        s_empty = torch.zeros((2, seq_len, k_val), dtype=torch.int32)
        v_empty = torch.ones((2, seq_len, k_val), dtype=torch.bool)
        _assert_matches_reference(s_empty, v_empty, seq_len, 0, always_select_tail, empty_vl)


def test_pool_mask_non_contiguous_inputs():
    seq_len = 16
    raw_sel = torch.randint(-1, 5, (2, seq_len, 8), dtype=torch.int32)
    raw_val = torch.randint(0, 2, (2, seq_len, 8)).bool()
    vl = torch.tensor([11, 0, 5, 0], dtype=torch.int32)[::2]
    assert not raw_sel[:, :, ::2].is_contiguous() and not vl.is_contiguous()
    _assert_matches_reference(raw_sel[:, :, ::2], raw_val[:, :, 1::2], seq_len, 4, True, vl)


def test_pool_mask_validation():
    sel = torch.zeros((2, 8, 2), dtype=torch.int32)
    val = torch.ones((2, 8, 2), dtype=torch.bool)
    with pytest.raises(ValueError, match="divisible by 4"):
        build_mask_from_pools(sel[:, :6], val[:, :6], 6, 1, True)
    with pytest.raises(ValueError, match="kv_length"):
        build_mask_from_pools(sel, val, 12, 2, True)
    with pytest.raises(ValueError, match="valid_lengths"):
        build_mask_from_pools(sel, val, 8, 2, True, valid_lengths=torch.zeros((3,), dtype=torch.int32))
    with pytest.raises(TypeError, match="selected_pools"):
        build_mask_from_pools(sel.float(), val, 8, 2, True)
    with pytest.raises(ValueError, match="same device"):
        build_mask_from_pools(sel, torch.ones((2, 8, 2), dtype=torch.bool, device="meta"), 8, 2, True)


def _install_fake_npu_fa(monkeypatch):
    def fake_npu_fusion_attention(q, k, v, _head_num, layout, *, atten_mask, scale, **_kwargs):
        assert layout == "BSND" and atten_mask.dtype == torch.uint8
        qh, kh, vh = q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        probs = torch.softmax(scores.masked_fill(atten_mask.bool(), float("-inf")), dim=-1)
        return (torch.matmul(probs, vh).transpose(1, 2).to(q.dtype),)

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(npu_fusion_attention=fake_npu_fusion_attention))
    if hasattr(torch, "npu"):
        monkeypatch.setattr(torch.npu, "is_bf16_supported", lambda: False)


def test_masked_flash_attention_pool_selection_and_grad(monkeypatch):
    _install_fake_npu_fa(monkeypatch)

    batch, seq_len, heads, d_head, d_latent, num_pools = 2, 16, 2, 8, 6, 4
    gen = torch.Generator().manual_seed(999)
    q0 = torch.randn((batch, seq_len, heads, d_head), generator=gen, requires_grad=True)
    lat0 = torch.randn((batch, seq_len, 1, d_latent), generator=gen, requires_grad=True)
    w_uk0 = torch.randn((heads, d_head, d_latent), generator=gen, requires_grad=True)
    w_uv0 = torch.randn((heads, d_head, d_latent), generator=gen, requires_grad=True)
    q1, lat1, w_uk1, w_uv1 = [t.detach().clone().requires_grad_() for t in (q0, lat0, w_uk0, w_uv0)]

    sel = torch.randint(0, num_pools, (batch, seq_len, 2), generator=gen, dtype=torch.int32)
    val = torch.ones((batch, seq_len, 2), dtype=torch.bool)
    vl = torch.tensor([11, 0], dtype=torch.int32)
    pool_sel = PoolSelection(sel, val, num_pools, True, vl)

    q_idx = torch.arange(seq_len)[None, :]
    qv = q_idx < vl[:, None]
    exp = sel.long().unsqueeze(-1) * 4 + torch.arange(4)
    exp = torch.where((qv.unsqueeze(-1) & val & (sel < ((q_idx + 1) // 4).unsqueeze(-1))).unsqueeze(-1), exp, -1).flatten(-2)
    t_cnt = torch.where(qv, q_idx + 1, 0) % 4
    t_off = torch.arange(3)
    t_tok = torch.where(
        qv.unsqueeze(-1) & (t_off[None, None, :] < t_cnt.unsqueeze(-1)),
        (q_idx - t_cnt + 1).unsqueeze(-1) + t_off,
        -1,
    )
    ref_topk = torch.where(qv.unsqueeze(-1), torch.cat([exp, t_tok], dim=-1), -1)

    from torch.utils.checkpoint import checkpoint

    out_pool = checkpoint(
        lambda q, l, uk, uv: masked_flash_attention(q, l, uk, uv, pool_sel, d_head**-0.5),
        q0, lat0, w_uk0, w_uv0, use_reentrant=False,
    )
    out_ref = masked_flash_attention(q1, lat1, w_uk1, w_uv1, ref_topk, d_head**-0.5)
    torch.testing.assert_close(out_pool, out_ref, rtol=0, atol=0)

    grad_out = torch.randn_like(out_pool)
    out_pool.backward(grad_out)
    out_ref.backward(grad_out)
    for g0, g1 in ((q0.grad, q1.grad), (lat0.grad, lat1.grad), (w_uk0.grad, w_uk1.grad), (w_uv0.grad, w_uv1.grad)):
        torch.testing.assert_close(g0, g1, rtol=0, atol=0)


def test_masked_flash_attention_empty_boundary_row_zeros_output_and_grad(monkeypatch):
    _install_fake_npu_fa(monkeypatch)

    batch, seq_len, heads, d_head, d_latent, num_pools = 1, 8, 2, 8, 6, 2
    gen = torch.Generator().manual_seed(123)
    q = torch.randn((batch, seq_len, heads, d_head), generator=gen, requires_grad=True)
    lat = torch.randn((batch, seq_len, 1, d_latent), generator=gen, requires_grad=True)
    w_uk = torch.randn((heads, d_head, d_latent), generator=gen, requires_grad=True)
    w_uv = torch.randn((heads, d_head, d_latent), generator=gen, requires_grad=True)

    sel = torch.zeros((batch, seq_len, 1), dtype=torch.int32)
    val = torch.ones((batch, seq_len, 1), dtype=torch.bool)
    val[:, 3, :] = False  # (3 + 1) % 4 == 0: empty tail and invalid pool -> empty query row
    out = masked_flash_attention(q, lat, w_uk, w_uv, PoolSelection(sel, val, num_pools, True, None), d_head**-0.5)
    assert torch.equal(out[:, 3], torch.zeros_like(out[:, 3]))
    out.mul(torch.randn_like(out)).sum().backward()
    assert torch.equal(q.grad[:, 3], torch.zeros_like(q.grad[:, 3]))


def test_glm5_next_indexer_return_pools_parity_and_fallback():
    pytest.importorskip("transformers", minversion="5.16.1", reason="GLM-5.3-Flash example dependency")
    from mindspeed_mm.fsdp.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from mindspeed_mm.fsdp.models.glm5_next.modeling_glm5_next import Glm5NextTextIndexer

    cfg = Glm5NextTextConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, q_lora_rank=8, kv_lora_rank=8,
        qk_nope_head_dim=128, qk_rope_head_dim=0, v_head_dim=128,
        index_n_heads=2, index_head_dim=8, index_topk=8, index_kpool=4,
        index_kpool_always_select_tail=True, dsa_implementation="masked_fa", indexer_implementation="eager",
    )
    indexer = Glm5NextTextIndexer(cfg, layer_idx=0).eval()
    torch.manual_seed(1234)
    for p in indexer.parameters():
        p.data.normal_(0, 0.5)

    cases = [
        (torch.ones((2, 16), dtype=torch.bool), True),
        (torch.tensor([[True] * 11 + [False] * 5, [False] * 16], dtype=torch.bool), True),
        (torch.ones((2, 17), dtype=torch.bool), False),
        (torch.ones((2, 3), dtype=torch.bool), False),
        (torch.tensor([[False] * 4 + [True] * 8 + [False] * 4, [True] * 16], dtype=torch.bool), False),
        (torch.tensor([[True, True, False, True] * 4, [True] * 16], dtype=torch.bool), False),
    ]
    for mask, expect_pool in cases:
        batch, seq_len = mask.shape
        hidden = torch.randn((batch, seq_len, cfg.hidden_size))
        q_resid = torch.randn((batch, seq_len, cfg.q_lora_rank))
        ctx = Glm5NextTextIndexer.prepare_indexer_context(mask)

        topk_ref = indexer(hidden, q_resid, mask, past_key_values=None, indexer_context=ctx)
        out = indexer(hidden, q_resid, mask, past_key_values=None, indexer_context=ctx, return_pools=True)
        assert isinstance(out, PoolSelection) == expect_pool
        if expect_pool:
            got_mask, got_ne = build_mask_from_pools(
                out.pools, out.valid, seq_len, out.num_pools, out.always_select_tail, out.valid_lengths
            )
            ref_mask, ref_ne = build_mask(topk_ref, seq_len)
            assert torch.equal(got_ne, ref_ne) and torch.equal(got_mask, ref_mask)
        else:
            assert torch.equal(out, topk_ref)
