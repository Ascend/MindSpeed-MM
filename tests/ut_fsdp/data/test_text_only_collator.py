# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Unit tests for the FSDP2 text-only collator."""

import copy
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
import torch

from mindspeed_mm.fsdp.data.data_utils.func_utils.collator import (
    IGNORE_INDEX,
    MultiModalDataCollatorForSeq2Seq,
    _has_modal_content,
)
from mindspeed_mm.fsdp.data.dataloader import data_collator as data_collator_module
from mindspeed_mm.fsdp.data.dataloader.data_collator import (
    DATA_COLLATOR,
    DataCollatorForTextOnly,
    resolve_data_collator,
)


class _DummyTokenizer:
    pad_token_id = 0
    eos_token = "</s>"
    padding_side = "right"
    model_input_names: ClassVar[list[str]] = ["input_ids", "attention_mask"]

    def pad(self, features, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors="pt"):
        del padding, max_length, return_tensors
        input_ids = [feature["input_ids"] for feature in features]
        attention_masks = [feature["attention_mask"] for feature in features]

        target_length = max(len(ids) for ids in input_ids)
        if pad_to_multiple_of and target_length % pad_to_multiple_of != 0:
            target_length = ((target_length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        padded_ids = []
        padded_masks = []
        for ids, attention_mask in zip(input_ids, attention_masks):
            pad_length = target_length - len(ids)
            padded_ids.append(list(ids) + [self.pad_token_id] * pad_length)
            padded_masks.append(list(attention_mask) + [0] * pad_length)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_masks, dtype=torch.long),
        }

    def encode(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return [1, 2, 3]


class _DummyMMPlugin:
    image_token = "<image>"
    audio_token = "<audio>"

    def get_mm_inputs(self, *args, **kwargs):
        del args, kwargs
        return {}

    def process_messages(self, messages, images, videos, audios, processor):
        del images, videos, audios, processor
        return messages

    def process_token_ids(self, input_ids, labels, images, videos, audios, tokenizer, processor):
        del images, videos, audios, tokenizer, processor
        return input_ids, labels


class _DummyTemplate:
    def __init__(self):
        self.mm_plugin = _DummyMMPlugin()


class _DummyConfig:
    model_type = "qwen3_5"
    image_token_id = None
    video_token_id = None


class _DummyModel:
    config = _DummyConfig()

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        del image_grid_thw, video_grid_thw, attention_mask
        batch_size, sequence_length = input_ids.shape
        position = torch.arange(sequence_length).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.stack([position, position, position])
        rope_deltas = torch.zeros(batch_size, 1)
        return position_ids, rope_deltas


def _make_collator(text_only=True, model=None, pad_to_multiple_of=8, template=None):
    return MultiModalDataCollatorForSeq2Seq(
        tokenizer=_DummyTokenizer(),
        template=template or _DummyTemplate(),
        model=model,
        pad_to_multiple_of=pad_to_multiple_of,
        label_pad_token_id=IGNORE_INDEX,
        text_only=text_only,
    )


def _make_text_features():
    return [
        {
            "input_ids": [10, 11, 12, 13],
            "attention_mask": [1, 1, 1, 1],
            "labels": [IGNORE_INDEX, 11, 12, 13],
        },
        {
            "input_ids": [20, 21],
            "attention_mask": [1, 1],
            "labels": [IGNORE_INDEX, 21],
        },
    ]


def test_text_only_output_matches_expected_text_batch():
    batch = _make_collator(model=_DummyModel())(_make_text_features())

    expected_input_ids = torch.tensor(
        [[10, 11, 12, 13, 0, 0, 0, 0], [20, 21, 0, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    expected_attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]],
        dtype=torch.long,
    )
    expected_labels = torch.tensor(
        [
            [IGNORE_INDEX, 11, 12, 13, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
            [IGNORE_INDEX, 21, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX],
        ],
        dtype=torch.long,
    )

    assert torch.equal(batch["input_ids"], expected_input_ids)
    assert torch.equal(batch["attention_mask"], expected_attention_mask)
    assert torch.equal(batch["labels"], expected_labels)


def test_text_only_output_has_no_modal_keys():
    batch = _make_collator(model=_DummyModel())(_make_text_features())

    modal_keys = (
        "images",
        "videos",
        "audios",
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    )
    assert all(key not in batch for key in modal_keys)


def test_text_only_does_not_mutate_source_features():
    features = _make_text_features()
    snapshot = copy.deepcopy(features)

    _make_collator(model=_DummyModel())(features)

    assert features == snapshot


@pytest.mark.parametrize("key", ["images", "videos", "audios"])
def test_text_only_rejects_non_empty_modal_content(key):
    features = _make_text_features()
    features[0][key] = ["modal-value"]

    with pytest.raises(ValueError, match=rf"text_only.*{key}"):
        _make_collator(model=_DummyModel())(features)


def test_text_only_accepts_empty_modal_fields():
    features = _make_text_features()
    for feature in features:
        feature.update({"images": [], "videos": [], "audios": []})

    batch = _make_collator(model=_DummyModel())(features)

    assert batch["input_ids"].shape == (2, 8)
    assert all(key not in batch for key in ("images", "videos", "audios"))


def test_text_only_skips_all_multimodal_processing():
    template = _DummyTemplate()
    template.mm_plugin.get_mm_inputs = MagicMock(return_value={})
    template.mm_plugin.process_messages = MagicMock(wraps=template.mm_plugin.process_messages)
    template.mm_plugin.process_token_ids = MagicMock(wraps=template.mm_plugin.process_token_ids)

    _make_collator(model=_DummyModel(), template=template)(_make_text_features())

    template.mm_plugin.get_mm_inputs.assert_not_called()
    template.mm_plugin.process_messages.assert_not_called()
    template.mm_plugin.process_token_ids.assert_not_called()


def test_text_only_preserves_qwen_mrope_outputs():
    batch = _make_collator(model=_DummyModel())(_make_text_features())

    assert batch["position_ids"].shape == (3, 2, 8)
    assert batch["rope_deltas"].shape == (2, 1)


def test_text_only_calls_rope_without_modal_grids():
    model = _DummyModel()
    model.get_rope_index = MagicMock(wraps=model.get_rope_index)

    _make_collator(model=model)(_make_text_features())

    rope_kwargs = model.get_rope_index.call_args.kwargs
    assert rope_kwargs["image_grid_thw"] is None
    assert rope_kwargs["video_grid_thw"] is None


def test_default_multimodal_path_still_injects_and_processes_fake_modal_data():
    template = _DummyTemplate()
    template.mm_plugin.get_mm_inputs = MagicMock(return_value={})
    template.mm_plugin.process_messages = MagicMock(wraps=template.mm_plugin.process_messages)
    template.mm_plugin.process_token_ids = MagicMock(wraps=template.mm_plugin.process_token_ids)

    batch = _make_collator(text_only=False, model=_DummyModel(), template=template)(_make_text_features())

    template.mm_plugin.get_mm_inputs.assert_called_once()
    assert template.mm_plugin.process_messages.call_count == 2
    assert template.mm_plugin.process_token_ids.call_count == 2
    assert "input_ids" in batch
    assert "labels" in batch


def test_registry_contains_text_only_collator():
    assert DATA_COLLATOR["text_only"] is DataCollatorForTextOnly


def test_resolve_data_collator_returns_text_only_class():
    collator_cls, kwargs = resolve_data_collator(
        {"collator_id": "text_only", "ignore_pad_token_for_loss": True},
        None,
    )

    assert collator_cls is DataCollatorForTextOnly
    assert kwargs["ignore_pad_token_for_loss"] is True


def test_text_only_wrapper_wires_the_underlying_collator():
    tokenizer = _DummyTokenizer()
    template = _DummyTemplate()
    model = _DummyModel()
    dataset_param = SimpleNamespace(
        preprocess_parameters=SimpleNamespace(to_dict=lambda: {"model_name_or_path": "unused"}),
        basic_parameters=SimpleNamespace(chat_template=None, template="dummy"),
    )
    parallel_state = MagicMock()
    parallel_state.is_cp_enable.return_value = False
    underlying_collator = MagicMock()
    underlying_collator.pad_to_multiple_of = 8

    with (
        patch.object(data_collator_module, "ProcessorArguments", return_value=MagicMock()),
        patch.object(
            data_collator_module,
            "load_tokenizer",
            return_value={"tokenizer": tokenizer, "processor": None},
        ),
        patch.object(data_collator_module, "get_template_and_fix_tokenizer", return_value=template),
        patch.object(data_collator_module, "get_parallel_state", return_value=parallel_state),
        patch.object(
            data_collator_module,
            "MultiModalDataCollatorForSeq2Seq",
            return_value=underlying_collator,
        ) as collator_cls,
    ):
        wrapper = DataCollatorForTextOnly(
            ignore_pad_token_for_loss=True,
            dataset_param=dataset_param,
            model=model,
            pad_to_multiple_of=8,
        )

    collator_kwargs = collator_cls.call_args.kwargs
    assert collator_kwargs["text_only"] is True
    assert collator_kwargs["model"] is model
    assert collator_kwargs["pad_to_multiple_of"] == 8
    assert collator_kwargs["label_pad_token_id"] == IGNORE_INDEX
    assert wrapper.data_collator is underlying_collator


class _NoLength:
    pass


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ([], False),
        ((), False),
        ([1], True),
        ("image.jpg", True),
        (_NoLength(), True),
    ],
)
def test_has_modal_content(value, expected):
    assert _has_modal_content(value) is expected
