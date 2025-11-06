# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy
from functools import partial
from typing import Dict, Any

from datasets import Dataset
import torch

import mindspeed.megatron_adaptor
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.training import pretrain
from mindspeed_mm.models.transformers_model import TransformersModel
from mindspeed_mm.utils.utils import compute_token_level_loss


def model_provider(*args, **kwargs):
    """Builds the model."""
    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)
    model = TransformersModel(vlm_config)

    return model


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    return batch


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dir = {}

    if args.log_tps:
        loss_mask = output_tensor['loss_mask'].view(-1).float()
        total_tokens = loss_mask.sum()
        dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        tokens_per_sample = torch.tensor(total_tokens, device=output_tensor['loss_mask'].device) / dp_size
        torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
        loss_dir["tokens per sample"] = tokens_per_sample

    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(output_tensor)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),
            local_num_tokens,
            loss_dir
        )

    loss = output_tensor['loss']
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss / mpu.get_context_parallel_world_size(), loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    batch_data = get_batch(data_iterator)
    output_tensor = model(**batch_data)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data

    datasets = build_mm_dataset(data_config.dataset_param)
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )
    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            valid_dataloader = build_dataloader(valid_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
