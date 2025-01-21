# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain InternVL."""

from copy import deepcopy
import torch
import torch.distributed
import mindspeed.megatron_adaptor  # noqa

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.training import pretrain
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.internvl_model import InternVLModel
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.patchs import vpp_patches, dummy_optimizer_patch  # noqa


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    print_rank_0("building InternVL model ...")
    model_config = deepcopy(args.mm.model)
    model_config.image_encoder.vision_encoder = get_model_config(
        model_config.image_encoder.vision_encoder
    )
    model_config.text_decoder = get_model_config(model_config.text_decoder)

    model = InternVLModel(model_config)
    if model_config.image_encoder.vision_encoder.is_freeze:
        model.freeze(freeze_image_encoder=True)

    return model


def get_batch_on_this_tp_rank(data_iterator):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            batch = None

        input_ids = batch["input_ids"].to(torch.cuda.current_device())
        labels = batch["labels"].to(torch.cuda.current_device())
        attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
        image = batch["pixel_values"].to(torch.cuda.current_device())
        image_flags = batch["image_flags"].to(torch.cuda.current_device())
        _broadcast(input_ids)
        _broadcast(labels)
        _broadcast(attention_mask)
        _broadcast(image)
        _broadcast(image_flags)

    else:
        raise NotImplementedError

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "image": image,
        "image_flags": image_flags,
    }

    return batch


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    input_ids = batch["input_ids"].to(torch.cuda.current_device())
    labels = batch["labels"].to(torch.cuda.current_device())
    attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
    image = batch["pixel_values"].to(torch.cuda.current_device())
    image_flags = batch["image_flags"].to(torch.cuda.current_device())
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "image": image,
        "image_flags": image_flags,
    }
    return (
        batch["input_ids"],
        batch["labels"],
        batch["attention_mask"],
        batch["image"],
        batch["image_flags"],
    )


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss = output_tensor["loss"].mean()
    loss_dir = {}
    if args.log_tps:
        B, S, _ = output_tensor["logits"].shape
        dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        tokens_per_sample = torch.tensor(S, device=output_tensor["logits"].device) / dp_size
        torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
        loss_dir["tokens per sample"] = tokens_per_sample
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0)
    return loss, loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    input_ids, labels, attention_mask, image, image_flags = get_batch(data_iterator)
    if image is not None:
        image = image.to(args.params_dtype)
    output = model(
        image=image,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        image_flags=image_flags,
    )
    return output, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        consumed_samples=args.consumed_train_samples,
    )
    train_dataloader, val_dataloader, test_dataloader = build_iterations(
        train_dataloader
    )
    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={
            "dataloader_type": "external",
            "vision_pretraining": False,
        },
    )
