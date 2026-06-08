# Copyright 2026 OpenMOSS team. All rights reserved.
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
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoProcessor, AutoTokenizer

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DataArguments
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.data.datasets.moss_tts_delay.common import (
    load_jsonl,
    normalize_audio_path_list,
    resolve_jsonl_paths,
)
from mindspeed_mm.fsdp.models.moss_tts_delay.processing_moss_tts import (
    MossTTSDelayProcessor,
)
from mindspeed_mm.fsdp.utils.register import data_register


USER_MESSAGE_KEYS = ("text", "instruction", "tokens", "quality", "sound_event", "ambient_sound", "language")


def normalize_audio_codes(value: Any, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, n_vq), got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


def normalize_audio_code_list(
    value: Any,
    field_name: str,
    allow_none: bool = False,
) -> Optional[List[Optional[torch.Tensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [normalize_audio_codes(value, field_name)]
    if isinstance(value, list):
        if not value:
            return None
        if allow_none and any(item is None for item in value):
            return [
                None if item is None else normalize_audio_codes(item, f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
        first_item = value[0]
        if torch.is_tensor(first_item):
            return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
        if isinstance(first_item, list):
            if first_item and isinstance(first_item[0], list):
                return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
            return [normalize_audio_codes(value, field_name)]
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


def load_jsonl_for_rank(
    spec: str,
    world_size: int,
    rank: int,
) -> tuple[List[Path], List[Dict[str, Any]], List[Path], bool]:
    all_paths = resolve_jsonl_paths(spec)
    rank_paths, using_pre_sharded_files = shard_paths_for_rank(
        all_paths, world_size=world_size, rank=rank
    )
    records: List[Dict[str, Any]] = []
    for path in rank_paths:
        records.extend(load_jsonl(path))
    return all_paths, records, rank_paths, using_pre_sharded_files


def shard_paths_for_rank(paths: List[Path], world_size: int, rank: int) -> tuple[List[Path], bool]:
    if world_size <= 1:
        return paths, False

    shard_pattern = re.compile(r"\.rank(\d+)-of-(\d+)\.jsonl$")
    parsed: List[tuple[Path, int, int]] = []
    for path in paths:
        match = shard_pattern.search(path.name)
        if match is None:
            return paths, False
        shard_rank = int(match.group(1))
        shard_world_size = int(match.group(2))
        parsed.append((path, shard_rank, shard_world_size))

    shard_world_sizes = {item[2] for item in parsed}
    if len(shard_world_sizes) != 1:
        return paths, False

    selected = [path for path, shard_rank, _ in parsed if shard_rank % world_size == rank]
    if not selected:
        raise ValueError(
            f"No shard assigned for rank={rank} world_size={world_size}. "
            "Please check --train-jsonl shard files and distributed config."
        )
    return selected, True


def processor_needs_audio_tokenizer(records: List[Dict[str, Any]]) -> bool:
    for record in records:
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if record.get("ref_audio_codes") is None and ref_audio is not None:
            return True
        if record.get("reference_audio_codes") is None:
            reference = normalize_audio_path_list(record.get("reference"), "reference", allow_none=True)
            if reference is not None and any(item is not None for item in reference):
                return True
            reference_audio = normalize_audio_path_list(record.get("reference_audio"), "reference_audio")
            if reference_audio is not None:
                return True
    return False


def build_processor(
    model_path: str,
    codec_path: str,
    need_audio_tokenizer: bool,
    audio_tokenizer_device: Optional[str],
    default_audio_tokenizer_device: str,
):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )

    if need_audio_tokenizer:
        processor = MossTTSDelayProcessor.from_pretrained(
            model_path,
            codec_path=codec_path,
        )
        device = audio_tokenizer_device or default_audio_tokenizer_device
        processor.audio_tokenizer = processor.audio_tokenizer.to(device)

    return processor


@data_register.register("mosstts_delay")
class MossTTSSFTDataset(Dataset):
    def __init__(self, basic_param, preprocess_param, **kwargs):
        data_args = DataArguments(**basic_param)
        process_args = ProcessorArguments(**preprocess_param)
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        train_paths, records, local_train_paths, using_pre_sharded_files = load_jsonl_for_rank(
            data_args.dataset_dir,
            world_size=world_size,
            rank=rank
        )
        if not records:
            raise ValueError(f"No records found in {args.train_jsonl}.")

        need_audio_tokenizer = processor_needs_audio_tokenizer(records)

        processor = build_processor(
            model_path=process_args.model_name_or_path,
            codec_path="OpenMOSS-Team/MOSS-Audio-Tokenizer",
            need_audio_tokenizer=need_audio_tokenizer,
            audio_tokenizer_device=None,
            default_audio_tokenizer_device='cpu'
        )

        self.records = records
        self.processor = processor
        self.n_vq = None
        self._audio_cache: Dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._pack_record(self.records[index])

    def _encode_audio_paths(self, paths: List[str], target_n_vq: int) -> List[torch.Tensor]:
        uncached_paths = [path for path in paths if path not in self._audio_cache]
        if uncached_paths:
            if getattr(self.processor, "audio_tokenizer", None) is None:
                raise ValueError(
                    "Found audio path fields but processor has no audio_tokenizer. "
                    "Either keep audio_tokenizer available or precompute the corresponding *_codes field."
                )
            encoded = self.processor.encode_audios_from_path(uncached_paths, n_vq=target_n_vq)
            for path, codes in zip(uncached_paths, encoded):
                self._audio_cache[path] = codes.cpu()
        return [self._audio_cache[path] for path in paths]

    def _validate_code_list(
        self,
        codes_list: Optional[List[Optional[torch.Tensor]]],
        target_n_vq: int,
        field_name: str,
    ) -> Optional[List[Optional[torch.Tensor]]]:
        if codes_list is None:
            return None
        for codes in codes_list:
            if codes is None:
                continue
            if codes.shape[1] != target_n_vq:
                raise ValueError(
                    f"`{field_name}` n_vq={codes.shape[1]} does not match target n_vq={target_n_vq}."
                )
        return codes_list

    def _resolve_reference_codes(self, record: Dict[str, Any], target_n_vq: int) -> Optional[List[Optional[torch.Tensor]]]:
        for code_field in ("reference_audio_codes", "ref_audio_codes"):
            if record.get(code_field) is not None:
                return self._validate_code_list(
                    normalize_audio_code_list(record[code_field], code_field, allow_none=(code_field == "reference_audio_codes")),
                    target_n_vq,
                    code_field,
                )

        for path_field in ("reference", "reference_audio", "ref_audio"):
            paths = normalize_audio_path_list(record.get(path_field), path_field, allow_none=(path_field == "reference"))
            if paths is not None:
                encoded_paths = self._encode_audio_paths([path for path in paths if path is not None], target_n_vq)
                encoded_iter = iter(encoded_paths)
                return [None if path is None else next(encoded_iter) for path in paths]

        return None

    def _pack_record(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        if "audio_codes" not in record:
            raise ValueError("Each record must contain `audio_codes`. Run prepare_data.py first.")

        target_codes = normalize_audio_codes(record["audio_codes"], "audio_codes")
        target_n_vq = int(target_codes.shape[1])
        if self.n_vq is not None and target_n_vq != self.n_vq:
            raise ValueError(f"Expected n_vq={self.n_vq}, but got {target_n_vq}.")

        reference_codes = self._resolve_reference_codes(record, target_n_vq)

        user_kwargs: Dict[str, Any] = {
            "reference": reference_codes,
        }
        for key in USER_MESSAGE_KEYS:
            if record.get(key) is not None:
                user_kwargs[key] = record[key]

        user_message = self.processor.build_user_message(**user_kwargs)
        prompt = self.processor([[user_message]], mode="generation", n_vq=target_n_vq)
        full_assistant_codes = target_codes
        assistant_message = self.processor.build_assistant_message(audio_codes_list=[full_assistant_codes])
        conversation = self.processor([[user_message, assistant_message]], mode="computing_loss", n_vq=target_n_vq)

        full_input_ids = conversation["input_ids"][0].cpu()
        prompt_length = int(prompt["input_ids"][0].shape[0])
        if prompt_length >= full_input_ids.shape[0]:
            raise ValueError("Prompt length must be shorter than the packed teacher-forcing sequence.")

        loss_mask = torch.zeros(full_input_ids.shape[0] - 1, dtype=torch.bool)
        loss_mask[prompt_length - 1 :] = True

        return {
            "input_ids": full_input_ids,
            "loss_mask": loss_mask,
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids_list = [item["input_ids"] for item in batch]
        padded = self.processor._pad(input_ids_list)

        full_input_ids = padded["input_ids"].to(torch.long)
        full_attention_mask = padded["attention_mask"].bool()

        loss_masks = pad_sequence(
            [item["loss_mask"] for item in batch],
            batch_first=True,
            padding_value=False,
            padding_side="left",
        )

        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_masks.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        # Audio pad code is a structural placeholder from the delay pattern, not a trainable target.
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == self.processor.model_config.audio_pad_code,
            -100,
        )

        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }
