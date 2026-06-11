# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
import os
os.environ["USE_TF"] = "FALSE"
from copy import deepcopy
from functools import partial
from typing import Dict, Any
import importlib

from datasets import Dataset
import torch

spec = importlib.util.spec_from_file_location("config_loader", "mindspeed_mm/configs/read_yaml_config.py")
spec.loader.exec_module(importlib.util.module_from_spec(spec))

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.training import pretrain
from mindspeed_mm.mcore.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
from mindspeed_mm.mcore.models.qwen3_5.qwen35_config import get_qwen3_5_llm_config, get_qwen3_5_vision_config
from mindspeed_mm.mcore.models.qwen3_5.spec import get_qwen3_5_text_block_spec


def model_provider(pre_process=None, post_process=None, vp_stage=None):
    """Builds the model."""
    args = get_args()
    print_rank_0("building Qwen3.5Model ...")
    model_config = deepcopy(args.mm.model)

    vision_transformer_config = get_qwen3_5_vision_config(model_config.image_encoder.to_dict())
    language_transformer_config = get_qwen3_5_llm_config(model_config.text_decoder.to_dict())
    block_spec = get_qwen3_5_text_block_spec(language_transformer_config, vp_stage=vp_stage)

    model = Qwen3_5Model(
        language_transformer_config=language_transformer_config,
        language_transformer_layer_spec=block_spec,
        vision_transformer_config=vision_transformer_config,
        pre_process=pre_process,
        post_process=post_process,
        pg_collection=None,
        vp_stage=vp_stage,
    )
    model.freeze(
        freeze_language_model=model_config.freeze_language_model,
        freeze_vision_model=model_config.freeze_vision_model,
        freeze_vision_projection=model_config.freeze_vision_projection
    )

    print_rank_0(model)
    return model


def move_to_device(batch, device, dtype):
    """Recursively move tensors in nested dicts to device."""
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device, dtype=dtype if torch.is_floating_point(v) else v.dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=device, dtype=dtype if torch.is_floating_point(t) else t.dtype) for t in v]


def get_batch(data_iterator, is_vit_last_stage=False):
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    dtype = torch.bfloat16 if get_args().bf16 else torch.float32
    move_to_device(batch, device=torch.cuda.current_device(), dtype=dtype)
    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    loss = output_tensor
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)

    report_loss = loss.clone().detach().view(1)
    torch.distributed.all_reduce(
        report_loss,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
        op=torch.distributed.ReduceOp.AVG
    )
    torch.distributed.all_reduce(
        num_tokens,
        group=mpu.get_data_parallel_group(with_context_parallel=True),
        op=torch.distributed.ReduceOp.AVG
    )

    report  = {"loss": report_loss, "num_tokens": num_tokens.view(1)}
    return loss, report


def forward_step(data_iterator, model):
    """Forward step."""
    batch = get_batch(data_iterator)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    pixel_values = batch['pixel_values']
    image_grid_thw = batch['image_grid_thw']

    output_tensor = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw
    )

    return output_tensor, partial(loss_func, attention_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data

    train_dataset = build_mm_dataset(data_config.dataset_param)
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )

    train_dataloader = build_dataloader(train_dataset)
    train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    import gc
    # set gc threshold to mitigate performance fluctuation
    gc.set_threshold(700, 10, 1000)
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
