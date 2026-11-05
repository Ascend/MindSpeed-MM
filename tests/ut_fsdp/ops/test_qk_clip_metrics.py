"""Per-head attention maxima exclude packed padding."""
from types import SimpleNamespace

import pytest
import torch

from mindspeed_mm.fsdp.ops.flash_attn import flash_attn
from mindspeed_mm.fsdp.ops.flash_attn.flash_attn import _per_head_softmax_max


@pytest.mark.parametrize("padding", [1000., torch.nan])
def test_packed_maxima_ignore_padding(padding):
    maxima = torch.full((2, 5, 8), padding)
    maxima[0, :3] = 7.
    maxima[1, :3] = 11.
    maxima = maxima.reshape(5, 2, 8)
    actual = _per_head_softmax_max(maxima, valid_tokens=3)
    torch.testing.assert_close(actual, torch.tensor([7., 11.]))


def test_empty_packed_rank_contributes_no_maximum():
    maxima = torch.full((4, 2, 8), torch.nan)
    assert torch.isneginf(_per_head_softmax_max(maxima, valid_tokens=0)).all()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("padding", [1000., torch.nan])
def test_padding_mask_drops_padded_query_rows(packed, padding):
    mask = torch.tensor([[False, True, True, False], [True, False, True, False]])
    maxima = torch.full((2, 2, 4, 8), padding)
    maxima.permute(0, 2, 1, 3)[mask] = torch.tensor([7., 11.])[:, None]
    if packed:
        maxima = maxima.reshape(8, 2, 8)
    bounds = [0, 4, 8] if packed else None
    torch.testing.assert_close(_per_head_softmax_max(maxima, query_mask=mask, cu_seq_lens=bounds), torch.tensor([7., 11.]))
    assert torch.isneginf(_per_head_softmax_max(maxima, query_mask=torch.zeros_like(mask), cu_seq_lens=bounds)).all()



@pytest.mark.parametrize("padded", [False, True])
def test_packed_statistics_keep_each_sequences_head_order(padded):
    # Two sequences of lengths 2 and 3, each stored as [heads, sequence, 8].
    last = [1000., 2000.] if padded else [7., 2.]
    raw = torch.tensor([1., 3., 2., 4., 5., 9., last[0], 6., 8., last[1]])
    maxima = raw[:, None].expand(-1, 8).contiguous().reshape(5, 2, 8)
    mask = torch.tensor([[True, True, True, True, not padded]])
    actual = _per_head_softmax_max(maxima, query_mask=mask, cu_seq_lens=[0, 2, 5])
    torch.testing.assert_close(actual, torch.tensor([9., 8.]))


def test_padding_mask_must_match_local_query_rows():
    with pytest.raises(ValueError, match="padding mask"):
        _per_head_softmax_max(torch.ones(2, 1, 4, 8), query_mask=torch.ones(2, 8, dtype=torch.bool))


@pytest.mark.parametrize("padding", [1000., torch.nan])
def test_chunked_metrics_use_chunk_local_rows(monkeypatch, padding):
    # Packed padding tokens restart position_ids, so they sit inside cu[-1].
    maxima = torch.full((6, 2, 8), padding)
    maxima[:3] = torch.tensor([7., 11.])[:, None]
    offset = 0

    def fusion_attention(query, *args, **kwargs):
        nonlocal offset
        result = maxima[offset:offset + query.size(0)]
        offset += query.size(0)
        # Each chunk here is one sequence; the real kernel stores heads first.
        return query, result.transpose(0, 1).contiguous().reshape_as(result)

    monkeypatch.setattr(flash_attn, "torch_npu", SimpleNamespace(npu_fusion_attention=fusion_attention), raising=False)
    recorded = []
    module = SimpleNamespace(training=True, _qk_clip_enabled=True, _publish_qk_clip_max_logits=recorded.append)
    sink = flash_attn._make_qk_clip_metric_sink(
        module, valid_tokens=6, padding_mask=torch.tensor([[1, 1, 1, 0, 0, 0]]),
        cu_seq_lens_q=[0, 2, 4, 6],
    )
    query = torch.zeros(6, 2, 1)
    flash_attn._npu_fa_chunked_varlen_attention(
        query, query, query, 2, "TND", None, [0, 2, 4, 6], 1., 0., 2, 3, metric_sink=sink,
    )
    torch.testing.assert_close(recorded[:2], [torch.tensor([7., 11.])] * 2)
    assert torch.isneginf(recorded[2]).all()


def test_only_right_padding_is_accepted():
    # Neat packing numbers samples from 1; zeros are padding.
    for mask in ([[1, 1, 0]], [[1, 2, 2, 0]], [[1, 1, 1]]):
        flash_attn._require_right_padding(torch.tensor(mask))
    for mask in ([[0, 1, 1]], [[1, 0, 1]], [[1, 1, 1], [0, 0, 1]]):
        with pytest.raises(ValueError, match="right padding"):
            flash_attn._require_right_padding(torch.tensor(mask))


def _flash_qk_clip_call(monkeypatch, maxima, ring=False):
    ps = SimpleNamespace(is_ulysses_enable=lambda: False, is_ring_enable=lambda: ring,
                         is_kvallgather_enable=lambda: False)
    monkeypatch.setattr(flash_attn, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(flash_attn, "IS_NPU_AVAILABLE", True)
    monkeypatch.setattr(flash_attn.dist, "is_initialized", lambda: ring)
    monkeypatch.setattr(flash_attn, "torch_npu", SimpleNamespace(
        npu_fusion_attention=lambda query, *args, **kwargs: (query, maxima)), raising=False)
    recorded = []
    module = SimpleNamespace(training=True, is_causal=True, _qk_clip_enabled=True,
                             _publish_qk_clip_max_logits=recorded.append,
                             config=SimpleNamespace(_attn_implementation="flash_attention_2"))
    return module, recorded


@pytest.mark.parametrize("query_len,key_len", [(1, 3), (2, 2)])
def test_flash_metric_uses_the_query_mask_columns(monkeypatch, query_len, key_len):
    # (1, 3): one query after two cached keys. (2, 2): batch == seq, a square padding mask.
    # Sample 0's last query is padding in both.
    mask = torch.ones(2, key_len, dtype=torch.long)
    mask[0, -1] = 0
    maxima = torch.full((2, 1, query_len, 8), 7.)
    maxima[0, :, -1] = 1000.
    module, recorded = _flash_qk_clip_call(monkeypatch, maxima)
    query = torch.ones(2, 1, query_len, 4, dtype=torch.float16)
    key = torch.ones(2, 1, key_len, 4, dtype=torch.float16)
    flash_attn.flash_attention_forward(module, query, key, key, mask, scaling=1., input_layout="BNSD")
    torch.testing.assert_close(recorded, [torch.tensor([7.])])


def test_ring_rejects_padded_batches(monkeypatch):
    module, _ = _flash_qk_clip_call(monkeypatch, None, ring=True)
    query = torch.ones(2, 1, 3, 4, dtype=torch.float16)
    with pytest.raises(NotImplementedError, match="ring"):
        flash_attn.flash_attention_forward(module, query, query, query, torch.tensor([[1, 1, 0], [1, 1, 1]]),
                                           scaling=1., input_layout="BNSD")



def test_packed_local_query_padding_is_rejected(monkeypatch):
    module, _ = _flash_qk_clip_call(monkeypatch, None)
    monkeypatch.setattr(flash_attn.dist, "is_initialized", lambda: True)
    ps = SimpleNamespace(is_ulysses_enable=lambda: False, is_ring_enable=lambda: False,
                         is_kvallgather_enable=lambda: True)
    monkeypatch.setattr(flash_attn, "get_parallel_state", lambda: ps)
    monkeypatch.setattr(flash_attn, "kv_all_gather_attention_qkv", lambda query, key, value, **kwargs:
                        (query, key, value, 128, 1, None, None, False, [0, 128], [0, 256]))
    query = torch.ones(1, 128, 1, 4, dtype=torch.float16)
    mask = torch.ones(1, 256, dtype=torch.long)
    mask[:, 192:] = 0
    with pytest.raises(NotImplementedError, match="local-query"):
        flash_attn.flash_attention_forward(module, query, query, query, mask,
            scaling=1., input_layout="1TND", total_seq_len=256,
            cu_seq_lens_q=[0, 256], cu_seq_lens_k=[0, 256])


def test_kimi_forward_rejects_left_padding_with_qk_clip(monkeypatch):
    # configuration_kimi_k3 ships with the model checkpoint, not this repository.
    kimi = pytest.importorskip("mindspeed_mm.fsdp.models.kimi_k3.modeling_kimi_linear")

    class ReachedMask(Exception):
        pass

    def create_causal_mask(**kwargs):
        raise ReachedMask

    monkeypatch.setattr(kimi, "create_causal_mask", create_causal_mask)
    model = SimpleNamespace(
        training=True, gradient_checkpointing=False,
        config=SimpleNamespace(use_cache=False, output_attentions=False, output_hidden_states=False,
                               use_return_dict=True, return_dict=True),
        layers=[SimpleNamespace(self_attn=SimpleNamespace(_qk_clip_enabled=True))],
    )
    embeds = torch.ones(1, 3, 1)
    for mask in ([[0, 1, 1]], [[1, 0, 1]]):
        with pytest.raises(ValueError, match="right padding"):
            kimi.KimiLinearModel.forward(model, inputs_embeds=embeds, attention_mask=torch.tensor(mask), use_cache=False)
    with pytest.raises(ReachedMask):
        kimi.KimiLinearModel.forward(model, inputs_embeds=embeds, attention_mask=torch.tensor([[1, 1, 0]]), use_cache=False)
