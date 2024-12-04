# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain SoRA."""
from copy import deepcopy

import mindspeed.megatron_adaptor  # noqa
import torch
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.qwen2vl_model import Qwen2VLModel
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building QWen2VL model ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)

    model = Qwen2VLModel(vlm_config)
    for name, parameters in model.named_parameters():
        if name.startswith('image_encoder'):
            parameters.requires_grad = False
    return model


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    input_ids = batch['input_ids'].to(torch.cuda.current_device())
    labels = batch['labels'].to(torch.cuda.current_device())
    attention_mask = batch['attention_mask'].to(torch.cuda.current_device())
    pixel_values = batch['pixel_values'].to(torch.cuda.current_device())
    image_grid_thw = batch['image_grid_thw'].to(torch.cuda.current_device())
    batch = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask,
        'pixel_values': pixel_values,
        'image_grid_thw': image_grid_thw
    }
    return batch['input_ids'], batch['labels'], batch['attention_mask'], batch['pixel_values'], batch['image_grid_thw']


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor['loss'].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""

    input_ids, labels, attention_mask, pixel_values, image_grid_thw = get_batch(data_iterator)

    output_tensor = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                          attention_mask=attention_mask, labels=labels)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    data_config = get_args().mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)
    train_dataloader = build_mm_dataloader(train_dataset, data_config.dataloader_param,
                                           process_group=mpu.get_data_parallel_group(),
                                           dataset_param=data_config.dataset_param)
    train_dataloader, val_dataloader, test_dataloader = build_iterations(train_dataloader)
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external", "vision_pretraining": False},
    )
