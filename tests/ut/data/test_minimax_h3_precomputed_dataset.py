# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the MiniMax-H3 precomputed dataset cache-format compatibility.

DiffSynth-Studio v2.1.0 caches store ``update_mask``, ``cond_rows`` and the
packed metadata ints; v2.1.5+ caches omit them and the dataset must derive
them exactly like DiffSynth's ``model_fn`` does. Fixtures satisfy the real
packed geometry enforced by the dataset (video rows = latent_t * h * w,
audio rows = audio_channel * audio_t, condition rows prepended to the
img/audio row ranges), matching the reshapes in ``unpatchify_video`` /
``unpack_audio``.
"""

import pytest
import torch

from mindspeed_mm.fsdp.data.datasets.minimax_h3.minimax_h3_precomputed_dataset import (
    MiniMaxH3PrecomputedDataset,
    _PACKED_INT_KEYS,
    _PACKED_TENSOR_KEYS,
)

# Geometry shared by every fixture below.
LATENT_T, LATENT_H_PATCHED, LATENT_W_PATCHED = 2, 4, 4
VIDEO_ROWS = LATENT_T * LATENT_H_PATCHED * LATENT_W_PATCHED  # 32
AUDIO_CHANNEL, AUDIO_T = 2, 4
AUDIO_ROWS = AUDIO_CHANNEL * AUDIO_T  # 8
TEXT_ROWS = 3


def _base_shared():
    return {
        # [B, C, T, H, W] -> VIDEO_ROWS video rows after patchify
        "input_latents": torch.randn(1, 24, LATENT_T, 8, 8),
        # [audio_channel, latent_dim, T] -> AUDIO_ROWS audio rows after pack
        "audio_input_latents": torch.randn(AUDIO_CHANNEL, 32, AUDIO_T),
    }


def _base_packed(cond_rows=0, ref_audio_rows=0):
    img_rows = cond_rows + VIDEO_ROWS
    audio_rows = ref_audio_rows + AUDIO_ROWS
    seq_len = TEXT_ROWS + img_rows + audio_rows
    return {
        "img_pos": torch.arange(TEXT_ROWS, TEXT_ROWS + img_rows),
        "audio_pos": torch.arange(TEXT_ROWS + img_rows, seq_len),
        "text_pos": torch.arange(TEXT_ROWS),
        "token_tags": torch.zeros(seq_len, dtype=torch.int64),
        "cu_seqlens": torch.tensor([0, seq_len, seq_len], dtype=torch.int32),
        "seq_len": seq_len,
        "img_position_ids": torch.randn(1, seq_len, 3, dtype=torch.float64),
    }


def _make_dataset(tmp_path, shared, positive):
    cache_dir = tmp_path / "cache" / "0"
    cache_dir.mkdir(parents=True)
    torch.save((shared, positive, None), cache_dir / "0.pth")
    return MiniMaxH3PrecomputedDataset({"dataset_dir": str(tmp_path / "cache")})


def test_v210_cache_fields_are_preserved(tmp_path):
    packed = _base_packed(cond_rows=4, ref_audio_rows=2)
    packed["update_mask"] = torch.zeros(packed["img_pos"].shape[0], dtype=torch.bool)
    packed["update_mask"][4:] = True
    packed["cond_rows"] = 4
    packed["ref_audio_rows"] = 2
    packed["text_len"] = TEXT_ROWS
    packed["audio_channel"] = AUDIO_CHANNEL
    packed["audio_t"] = AUDIO_T
    packed["latent_t"] = LATENT_T
    packed["latent_h_patched"] = LATENT_H_PATCHED
    packed["latent_w_patched"] = LATENT_W_PATCHED
    dataset = _make_dataset(
        tmp_path, _base_shared(), {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    batch = dataset[0]
    # The explicit v2.1.0 mask is passed through untouched.
    assert torch.equal(batch["packed_update_mask"], packed["update_mask"])
    assert batch["packed_cond_rows"] == 4
    assert batch["packed_ref_audio_rows"] == 2
    for key in _PACKED_INT_KEYS:
        assert f"packed_{key}" in batch
    for key in _PACKED_TENSOR_KEYS:
        assert f"packed_{key}" in batch


def test_v215_cache_fields_are_derived(tmp_path):
    shared = _base_shared()
    shared["keyframe_cond_anchor"] = torch.randn(4, 96)
    packed = _base_packed(cond_rows=4)
    dataset = _make_dataset(
        tmp_path, shared, {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    batch = dataset[0]
    assert batch["packed_cond_rows"] == 4
    assert batch["packed_ref_audio_rows"] == 0
    assert batch["packed_text_len"] == TEXT_ROWS
    assert batch["packed_audio_channel"] == AUDIO_CHANNEL
    assert batch["packed_audio_t"] == AUDIO_T
    assert batch["packed_latent_t"] == LATENT_T
    assert batch["packed_latent_h_patched"] == LATENT_H_PATCHED
    assert batch["packed_latent_w_patched"] == LATENT_W_PATCHED
    # Condition (keyframe) rows at the front of img_pos are excluded from the
    # loss, so the derived mask clears them; video rows stay True.
    mask = batch["packed_update_mask"]
    assert mask.dtype == torch.bool
    assert mask.shape == (packed["img_pos"].shape[0],)
    assert not mask[:4].any()
    assert mask[4:].all()
    # position ids are transported as raw bits and restored by the model
    restored = batch["packed_img_position_ids_bits"].view(torch.float64)
    assert torch.equal(restored, packed["img_position_ids"])


def test_v215_t2va_mask_is_all_true(tmp_path):
    packed = _base_packed()
    dataset = _make_dataset(
        tmp_path, _base_shared(), {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    batch = dataset[0]
    assert batch["packed_cond_rows"] == 0
    assert batch["packed_update_mask"].all()


def test_seq_len_fallback_from_cu_seqlens(tmp_path):
    packed = _base_packed()
    del packed["seq_len"]
    dataset = _make_dataset(
        tmp_path, _base_shared(), {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    batch = dataset[0]
    assert batch["packed_seq_len"] == int(packed["cu_seqlens"][-1])


def test_v215_ref2va_anchors_take_precedence(tmp_path):
    shared = _base_shared()
    shared["keyframe_cond_anchor"] = torch.randn(4, 96)
    shared["ref_blocks"] = [
        {"visual_rows": torch.randn(6, 96), "audio_rows": torch.randn(2, 32)},
    ]
    packed = _base_packed(cond_rows=6, ref_audio_rows=2)
    dataset = _make_dataset(
        tmp_path, shared, {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    batch = dataset[0]
    assert batch["packed_cond_rows"] == 6
    assert batch["packed_ref_audio_rows"] == 2
    assert batch["ref_visual_anchor"].shape == (6, 96)
    assert batch["ref_audio_anchor"].shape == (2, 32)
    mask = batch["packed_update_mask"]
    assert not mask[:6].any()
    assert mask[6:].all()


def test_v210_update_mask_length_mismatch_raises(tmp_path):
    packed = _base_packed(cond_rows=4)
    packed["update_mask"] = torch.zeros(7, dtype=torch.bool)
    packed["cond_rows"] = 4
    dataset = _make_dataset(
        tmp_path, _base_shared(), {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    with pytest.raises(ValueError, match="update_mask length"):
        dataset[0]


def test_v215_audio_geometry_mismatch_raises(tmp_path):
    shared = _base_shared()
    # audio rows = 3 * 4 = 12 != audio_channel * audio_t (8) derived from a
    # [2, 32, 4] tensor: the cache layout is inconsistent, fail at load time.
    shared["audio_input_latents"] = torch.randn(3, 32, 4)
    dataset = _make_dataset(
        tmp_path, shared, {"packed": _base_packed(), "prompt_embeds": torch.randn(3, 8)}
    )

    with pytest.raises(ValueError, match="audio_channel\\*audio_t"):
        dataset[0]


def test_v215_video_geometry_mismatch_raises(tmp_path):
    shared = _base_shared()
    # Truncate img_pos so video rows no longer match t*h*w.
    packed = _base_packed()
    packed["img_pos"] = packed["img_pos"][:-2]
    packed["seq_len"] = int(packed["seq_len"]) - 2
    dataset = _make_dataset(
        tmp_path, shared, {"packed": packed, "prompt_embeds": torch.randn(3, 8)}
    )

    with pytest.raises(ValueError, match="video rows"):
        dataset[0]


def test_v215_audio_latents_with_batch_dim_raises(tmp_path):
    shared = _base_shared()
    # A leading batch dim on audio latents would silently mis-derive
    # audio_channel (batch size instead of channels); reject it explicitly.
    shared["audio_input_latents"] = torch.randn(1, AUDIO_CHANNEL, 32, AUDIO_T)
    dataset = _make_dataset(
        tmp_path, shared, {"packed": _base_packed(), "prompt_embeds": torch.randn(3, 8)}
    )

    with pytest.raises(ValueError, match="audio_input_latents must be"):
        dataset[0]
