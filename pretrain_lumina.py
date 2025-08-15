# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain Lumina."""
from typing import Dict

import torch

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    unwrap_model,
)

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.lumina_model import Lumina
from mindspeed_mm.utils.utils import get_device


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building Lumina model ...")
    model = Lumina(args.mm.model)
    return model


def get_batch(data_iterator):
    batch = None
    if data_iterator is not None:
        data_item = next(data_iterator, None)
        input_ids, labels = data_item
        batch = {
            "input_ids": input_ids,
            "labels": labels,
        }
    return batch


def loss_func(output_tensor):
    """Loss function."""
    closs, additional_loss_dict = output_tensor
    averaged_closs = average_losses_across_data_parallel_group([closs])
    loss = closs
    averaged_additional_loss_dict = {"closs": averaged_closs}
    for name, (add_loss, weight) in additional_loss_dict.items():
        loss = loss + add_loss * weight
        averaged_additional_loss_dict[name] = average_losses_across_data_parallel_group([add_loss])
    
    averaged_additional_loss_dict["total_loss"] = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)

    return loss, averaged_additional_loss_dict


def forward_step(data_iterator, model):
    """Forward step."""
    batch = get_batch(data_iterator)
    output = model(**batch)
    return output, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)

    process_group = mpu.get_data_parallel_group()
    # Build dataloader
    train_dataloader = build_mm_dataloader(
        train_dataset,
        data_config.dataloader_param,
        process_group=process_group,
        dataset_param=data_config.dataset_param,
    )
    data_iterator, _, _ = build_iterations(train_dl=train_dataloader)
    return data_iterator, None, None


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False, "curr_forward_iteration": 0},
    )
