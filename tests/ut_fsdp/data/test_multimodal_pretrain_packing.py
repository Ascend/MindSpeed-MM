# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Tests for complete-document multimodal pre-training packing."""

from types import SimpleNamespace

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import (
    IGNORE_INDEX,
    PackedMultimodalPretrainDatasetProcessor,
    Role,
)


class _Tokenizer:
    """Minimal whitespace tokenizer that avoids loading real model weights."""

    eos_token = " <eos>"
    add_bos_token = False

    def __init__(self):
        self.vocab = {}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [self._token_id(token) for token in text.split()]

    def _token_id(self, token):
        # Assign a stable token id so tests can recover image order from the token stream.
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

    def decode(self, input_ids, skip_special_tokens=False):
        del skip_special_tokens
        inverse_vocab = {token_id: token for token, token_id in self.vocab.items()}
        return " ".join(inverse_vocab[token_id] for token_id in input_ids)

    def convert_tokens_to_ids(self, token):
        return self._token_id(token)


class _MultimodalPlugin:
    """Expand ``<image>`` into traceable vision tokens for order assertions."""

    def process_messages(self, messages, images, videos, audios, processor):
        del processor
        assert not videos
        assert not audios
        result = [message.copy() for message in messages]
        image_iter = iter(images)
        for message in result:
            while "<image>" in message["content"]:
                image = next(image_iter)
                # Encode the image name in the test token. A real plugin expands
                # the placeholder according to the image grid instead.
                replacement = f"<vision> <img:{image}> </vision>"
                message["content"] = message["content"].replace("<image>", replacement, 1)
        try:
            next(image_iter)
        except StopIteration:
            pass
        else:
            raise AssertionError("Unused image in test sample")
        return result

    def process_token_ids(
            self, input_ids, labels, images, videos, audios, tokenizer, processor
    ):
        del images, videos, audios, tokenizer, processor
        return input_ids, labels


def _make_processor(processor_cls=PackedMultimodalPretrainDatasetProcessor):
    """Build a processor with only the dependencies required by PT preprocessing."""

    tokenizer = _Tokenizer()
    processor = processor_cls(
        template=SimpleNamespace(mm_plugin=_MultimodalPlugin()),
        tokenizer=tokenizer,
        processor=object(),
        data_args=SimpleNamespace(
            cutoff_len=64,
            packing=True,
            template="qwen3_vl_nothink",
        ),
    )
    return processor, tokenizer


def _examples():
    """Mimic the column-oriented batch passed by ``Dataset.map(batched=True)``."""

    # Each document owns one image. Packing must preserve the same order in the
    # token stream and the concatenated media list.
    return {
        "_prompt": [
            [{"role": Role.ASSISTANT.value, "content": "<image> red cat"}],
            [{"role": Role.ASSISTANT.value, "content": "<image> blue bird flies"}],
            [{"role": Role.ASSISTANT.value, "content": "<image> green tree"}],
        ],
        "_response": [[], [], []],
        "_system": ["", "", ""],
        "_images": [["a.png"], ["b.png"], ["c.png"]],
        "_videos": [None, None, None],
        "_audios": [None, None, None],
    }


def test_multimodal_pretrain_packing_preserves_atomic_documents_and_boundaries():
    """Verify atomic packing, media order, and all document boundary signals."""

    processor, tokenizer = _make_processor()

    # The cutoff is large enough for all three short documents to share one row.
    result = processor.preprocess_dataset(_examples())

    assert len(result["input_ids"]) == 1
    input_ids = result["input_ids"][0]
    labels = result["labels"][0]
    position_ids = result["position_ids"][0]
    cu_seqlens = result["cu_seqlens"][0]
    packed_images = result["images"][0]

    # Dataset preprocessing has not padded the row, so every mask value is valid.
    assert set(result["attention_mask"][0]) == {1}
    # N documents require N+1 cumulative boundaries, ending at physical length.
    assert cu_seqlens[0] == 0
    assert cu_seqlens[-1] == len(input_ids)
    assert len(cu_seqlens) == 4

    # Positions restart at zero, while the first ignored label prevents a
    # next-token loss from crossing the preceding document boundary.
    for start in cu_seqlens[:-1]:
        assert position_ids[start] == 0
        assert labels[start] == IGNORE_INDEX

    # Recover image names from vision tokens and compare them with the packed
    # media list to detect any token/media reordering.
    token_id_to_image = {
        tokenizer.vocab[f"<img:{image}>"]: image for image in ("a.png", "b.png", "c.png")
    }
    images_from_tokens = [
        token_id_to_image[token_id] for token_id in input_ids if token_id in token_id_to_image
    ]
    assert images_from_tokens == packed_images

    # Every cu_seqlens interval owns exactly one image, proving that packing did
    # not split a multimodal document across knapsacks.
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        assert sum(token_id in token_id_to_image for token_id in input_ids[start:end]) == 1
