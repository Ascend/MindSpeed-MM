from dataclasses import dataclass
from typing import Dict, Sequence, Any
import logging

import torch
from transformers import DataCollatorForLanguageModeling

from mindspeed_mm.fsdp.data.data_utils.func_utils.collator import MultiModalDataCollatorForSeq2Seq
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import load_tokenizer, update_tokenizer_with_chat_template, IGNORE_INDEX
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.data.data_utils.func_utils.template import get_template_and_fix_tokenizer
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state

logger = logging.getLogger(__name__)

class DataCollatorForQwen2vl:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')

        # chat_template的优先级高于template。如果chat_template和template同时为None，使用模型自带的chat_template
        chat_template_path = dataset_param.basic_parameters.chat_template
        if chat_template_path is not None:
            tokenizer = update_tokenizer_with_chat_template(tokenizer, chat_template_path)
            template = get_template_and_fix_tokenizer(tokenizer, None)
        else:
            template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)

        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            model=kwargs.get("model", None),
            pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8),  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )
        ps = get_parallel_state()
        cp_size = ps.get_cp_group_size()
        pad_to_multiple_of = self.data_collator.pad_to_multiple_of
        if pad_to_multiple_of % cp_size != 0:
            raise ValueError(f"pad_to_multiple_of {pad_to_multiple_of} must be divisible by context parallel size {cp_size}.")
        if ps.ring_attention_size > 1 and pad_to_multiple_of % (2 * cp_size) != 0:
            raise ValueError(f"pad_to_multiple_of {pad_to_multiple_of} must be divisible by context parallel size {cp_size} * 2 when using ring CP.")

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


@dataclass
class OmniModalDataCollatorForSeq2Seq(MultiModalDataCollatorForSeq2Seq):
    r"""Omni data collator that adds audio feature before the __call__ method returns
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        features["use_audio_in_video"] = getattr(self.processor, "use_audio_in_video", False)
        return features


class DataCollatorForQwen3Omni:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)
        self.data_collator = OmniModalDataCollatorForSeq2Seq(
            template=template,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


class DataCollatorForStep3VL:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')

        chat_template_path = dataset_param.basic_parameters.chat_template
        if chat_template_path is not None:
            tokenizer = update_tokenizer_with_chat_template(tokenizer, chat_template_path)
            template = get_template_and_fix_tokenizer(tokenizer, None)
        else:
            template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)

        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            model=kwargs.get("model", None),
            pad_to_multiple_of=kwargs.get("pad_to_multiple_of", 8),
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


class DataCollatorForLLMPretrain:
    def __init__(self, dataset_param=None, **kwargs):
        if dataset_param is None:
            raise ValueError("dataset_param is required for DataCollatorForLLM Pretrain")
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)


DATA_COLLATOR = {}


def _register_data_collator(name: str, collator_cls) -> None:
    if name in DATA_COLLATOR:
        raise ValueError(f"Data collator with name {name} is already registered.")
    DATA_COLLATOR[name] = collator_cls


def resolve_data_collator(collate_param, dataset_param):
    collate_kwargs = dict(collate_param or {})
    collator_id = collate_kwargs.pop("collator_id", None)
    model_name = collate_kwargs.pop("model_name", None)

    if model_name is not None:
        logger.warning(
            "model_name is deprecated and will be removed in a future version. "
            "Please use collator_id instead. "
            f"Available collator_ids: {', '.join(sorted(DATA_COLLATOR))}."
        )

    if collator_id is not None and model_name is not None and collator_id != model_name:
        raise ValueError(
            "collator_id and deprecated model_name refer to different data collators: "
            f"{collator_id} != {model_name}."
        )

    data_collate_type = collator_id or model_name

    if data_collate_type is None:
        raise ValueError(
            "No data collator is specified. Please set "
            "data.dataloader_param.collate_param.collator_id "
            "(or deprecated model_name)."
        )

    if data_collate_type not in DATA_COLLATOR:
        raise ValueError(
            f"No data collator is registered as {data_collate_type}. "
            f"Available collators: {', '.join(sorted(DATA_COLLATOR))}."
        )
    return DATA_COLLATOR[data_collate_type], collate_kwargs


_register_data_collator(
    "qwen3vl",
    DataCollatorForQwen2vl,
)
_register_data_collator(
    "qwen3omni",
    DataCollatorForQwen3Omni,
)
_register_data_collator(
    "llm_pretrain",
    DataCollatorForLLMPretrain,
)
_register_data_collator(
    "step3_vl",
    DataCollatorForStep3VL,
)
