import os
from copy import deepcopy
from typing import Any, Dict

import torch

from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group


from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.models.omni_model import OmniModel
from mindspeed_mm.training import pretrain
from mindspeed_mm.utils.transformer_model_config import get_model_config
mindspeed_args = get_mindspeed_args()


def model_provider(pre_process=True, post_process=True, modules=None):
    """Builds the model."""
    args = get_args()
    print_rank_0("building OmniModel ...")
    omni_config = deepcopy(args.mm.model)

    model = OmniModel(omni_config)

    _apply_freezing(model, omni_config)

    return model


def _apply_freezing(model, omni_config):
    """Apply freezing settings to the model."""
    has_image = hasattr(omni_config, 'image_encoder') and omni_config.image_encoder is not None
    freeze_image_encoder = has_image and getattr(omni_config.image_encoder.vision_encoder, 'freeze', False)
    freeze_image_projection = has_image and getattr(omni_config.image_encoder.vision_projector, 'freeze', False)

    has_ae = hasattr(omni_config, 'ae') and omni_config.ae is not None
    freeze_ae = has_ae and getattr(omni_config.ae, 'freeze', True)

    model.freeze(
        freeze_image_encoder=freeze_image_encoder,
        freeze_image_projection=freeze_image_projection,
        freeze_ae=freeze_ae
    )


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
    if not isinstance(batch, dict):
        batch = batch.to_dict()
    move_to_device(batch, get_args().params_dtype)
    return batch


def loss_func(output_tensor):
    """Loss function."""
    loss = output_tensor[0].mean()
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss = loss.unsqueeze(0)
    return loss / mpu.get_context_parallel_world_size(), {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    batch = get_batch(data_iterator)
    output_tensor = model(batch)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data
    train_dataset = build_mm_dataset(data_config.dataset_param)
    train_dataloader = build_mm_dataloader(train_dataset, data_config.dataloader_param,
                                           process_group=mpu.get_data_parallel_group(),
                                           dataset_param=data_config.dataset_param,
                                           consumed_samples=args.consumed_train_samples, )
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