# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
"""Pretrain VideoAlign."""

import mindspeed.megatron_adaptor
import torch

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args
from megatron.training.global_vars import set_args
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.patchs import dummy_optimizer_patch
from mindspeed_mm.tasks.rl.dpo.reward_trainer import VideoVLMRewardTrainer


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    datasets = build_mm_dataset(data_config.dataset_param)
    if isinstance(datasets, dict) and "model_args" in datasets.keys():
        model_args = datasets['model_args']
        args.mm.model.special_token_ids = model_args['special_token_ids']
        args.mm.model.token_embedding_length = model_args['token_embedding_length']
        args.mm.model.tokenizer_padding_side = model_args['tokenizer_padding_side']
        args.mm.model.pad_token_id = model_args['pad_token_id']
        set_args(args)
        datasets = datasets['dataset']
    train_dataset, val_dataset = datasets
    train_dataloader = build_mm_dataloader(train_dataset, data_config.dataloader_param,
                                           process_group=mpu.get_data_parallel_group(),
                                           dataset_param=data_config.dataset_param,
                                           consumed_samples=args.consumed_train_samples,)
    if val_dataset:
        val_dataloader = build_mm_dataloader(val_dataset, data_config.dataloader_param,
                                           process_group=mpu.get_data_parallel_group(),
                                           dataset_param=data_config.dataset_param,
                                           consumed_samples=args.consumed_valid_samples,)
    else:
        val_dataloader = None
    train_dataloader, val_dataloader, test_dataloader = build_iterations(train_dataloader, val_dataloader)
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    trainer = VideoVLMRewardTrainer(
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
        model_type=ModelType.encoder_or_decoder,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
    trainer.train()
