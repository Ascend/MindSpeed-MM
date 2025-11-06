# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy

import mindspeed.megatron_adaptor
import torch
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.global_vars import set_args
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.common.transformer.multi_token_prediction import apply_mtp_patch
from mindspeed_mm.models.deepseekvl_model import VLMModel
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.patchs import dummy_optimizer_patch
mindspeed_args = mindspeed.megatron_adaptor.get_mindspeed_args()
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    if hasattr(args.mm.model.text_decoder, "mtp_num_layers") and args.mm.model.text_decoder.mtp_num_layers:
        args.mtp_num_layers = args.mm.model.text_decoder.mtp_num_layers
        set_args(args)
        apply_mtp_patch()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)

    # distinguish model construct stage when pipeline parallel
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process

    if vlm_config.image_encoder:
        vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)

        model = VLMModel(vlm_config)
        model.freeze(freeze_image_encoder=getattr(vlm_config.image_encoder.vision_encoder, 'freeze', True), \
            freeze_image_projection=getattr(vlm_config.image_encoder.vision_projector, 'freeze', False))
    else:
        vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)
        model = VLMModel(vlm_config)

    return model


def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    input_ids = batch["input_ids"].to(torch.cuda.current_device())
    labels = batch["labels"].to(torch.cuda.current_device())
    attention_mask = batch["attention_mask"].to(torch.cuda.current_device())
    images = batch["images"].to(torch.cuda.current_device()).to(args.params_dtype)
    images_seq_mask = batch["images_seq_mask"].to(torch.cuda.current_device())
    images_spatial_crop = batch["images_spatial_crop"].to(torch.cuda.current_device())

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "images": images,
        "images_seq_mask": images_seq_mask,
        "images_spatial_crop": images_spatial_crop
    }
    return batch


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss = output_tensor['loss'].mean()
    loss_dir = {}
    if args.log_tps:
        B, S, _ = output_tensor['logits'].shape
        dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        tokens_per_sample = torch.tensor(S, device=output_tensor['logits'].device) / dp_size
        torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
        loss_dir["tokens per sample"] = tokens_per_sample
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss, loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    output_tensor = model(**get_batch(data_iterator))
    return output_tensor, loss_func


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
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
