import os
import warnings
from typing import Any, Dict, Optional

from datasets import load_dataset
from transformers.training_args import TrainingArguments

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DataArguments, DatasetAttr, load_tokenizer
from mindspeed_mm.fsdp.data.data_utils.func_utils.log import get_logger
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.utils.register import data_register


logger = get_logger(__name__)


def _log_dataset_lengths(train_dataset, eval_dataset=None):
    try:
        logger.info_rank0("train dataset len: %s", len(train_dataset))
        if eval_dataset is not None:
            logger.info_rank0("eval dataset len: %s", len(eval_dataset))
    except TypeError:
        logger.info_rank0("dataset len is unavailable for streaming dataset")


def _resolve_audio_path(audio_path: Optional[str], dataset_dir: str) -> Optional[str]:
    if audio_path and not os.path.isabs(audio_path):
        candidate = os.path.join(dataset_dir, audio_path)
        if os.path.isfile(candidate):
            return candidate
    return audio_path


def _extract_qwen3_asr_fields(example: Dict[str, Any], dataset_attr: DatasetAttr, data_args: DataArguments):
    prompt = example.get("prompt", "")
    audio = example.get(dataset_attr.audios or "audio", example.get("audio"))
    target = example.get("text", example.get("target", ""))

    for message in example.get(dataset_attr.messages or "messages", []) or []:
        role = message.get(dataset_attr.role_tag or "role")
        content = message.get(dataset_attr.content_tag or "content", "")
        if role == (dataset_attr.system_tag or "system") and not prompt:
            prompt = content
        elif role == (dataset_attr.assistant_tag or "assistant"):
            target = content

    return prompt or "", _resolve_audio_path(audio, data_args.dataset_dir), target


def _build_prefix_text(processor, prompt: str):
    prefix_messages = [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": None}]},
    ]
    return processor.apply_chat_template(
        [prefix_messages],
        add_generation_prompt=True,
        tokenize=False,
    )[0]


@data_register.register("qwen3_asr")
def get_qwen3_asr_dataset(basic_param, preprocess_param, dataset_param, **kwargs):
    data_args = DataArguments(**basic_param)
    process_args = ProcessorArguments(**preprocess_param)
    dataset_attr = DatasetAttr(**dataset_param["attr"])
    processor = load_tokenizer(process_args)["processor"]

    def preprocess(example):
        prompt, audio, target = _extract_qwen3_asr_fields(example, dataset_attr, data_args)
        return {
            "prompt": prompt,
            "audio": audio,
            "target": target,
            "prefix_text": _build_prefix_text(processor, prompt),
        }

    map_kwargs = {}
    if data_args.preprocessing_num_workers is not None:
        map_kwargs["num_proc"] = data_args.preprocessing_num_workers
    map_kwargs["load_from_cache_file"] = not data_args.overwrite_cache

    with TrainingArguments(output_dir="./").main_process_first(desc="pre-process qwen3_asr dataset"):
        train_dataset = load_dataset(
            path="json",
            data_files=data_args.dataset,
            split="train",
            cache_dir=data_args.cache_dir,
            streaming=data_args.streaming,
        )
        if data_args.max_samples and not data_args.streaming:
            train_dataset = train_dataset.select(range(data_args.max_samples))
        train_dataset = train_dataset.map(
            preprocess,
            remove_columns=list(train_dataset.column_names),
            desc="running qwen3_asr preprocess on train_dataset",
            **map_kwargs,
        )

        eval_dataset = None
        if data_args.val_dataset:
            eval_dataset = load_dataset(
                path="json",
                data_files=data_args.val_dataset,
                split="train",
                cache_dir=data_args.cache_dir,
                streaming=data_args.streaming,
            )
            if data_args.val_max_samples:
                eval_dataset = eval_dataset.select(range(data_args.val_max_samples))
            if data_args.val_rate is not None and data_args.val_rate > 0.0:
                warnings.warn(
                    "Both val_dataset and val_rate have been provided. The val_dataset will take priority.",
                    UserWarning,
                )
            eval_dataset = eval_dataset.map(
                preprocess,
                remove_columns=list(eval_dataset.column_names),
                desc="running qwen3_asr preprocess on eval_dataset",
                **map_kwargs,
            )

    _log_dataset_lengths(train_dataset, eval_dataset)
    if eval_dataset is not None:
        return train_dataset, eval_dataset
    return train_dataset
