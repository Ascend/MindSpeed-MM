# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
"""Posttrain QWEN2VL DPO."""
import os
os.environ["USE_TF"] = "FALSE"

import mindspeed.megatron_adaptor
import torch

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.patchs import dummy_optimizer_patch
from mindspeed_mm.tasks.rl.dpo.qwen2vl_dpo_trainer import Qwen2VLDPOTrainer


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)
    train_dataloader = build_mm_dataloader(train_dataset, data_config.dataloader_param,
                                           process_group=mpu.get_data_parallel_group(),
                                           dataset_param=data_config.dataset_param,
                                           consumed_samples=args.consumed_train_samples,)
    train_dataloader, val_dataloader, test_dataloader = build_iterations(train_dataloader)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    trainer = Qwen2VLDPOTrainer(
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_type=ModelType.encoder_or_decoder,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
    trainer.train()
