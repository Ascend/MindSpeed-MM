# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain AE."""
import os
os.environ["USE_TF"] = "FALSE"

import torch

import mindspeed.megatron_adaptor

from mindspeed_mm.models.ae.training.training import pretrain_ae
from mindspeed_mm.data import build_ae_dataloader, build_ae_dataset
from mindspeed_mm.models.ae.training.global_vars import get_ae_args
from mindspeed_mm.models.ae.base import AEModel
from mindspeed_mm.models.ae.losses import DISCRIMINATOR_MODEL_MAPPINGS


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_ae_args()
    ae_model = AEModel(args.model.ae).model
    discrim_config = args.model.discriminator
    discrim_model = DISCRIMINATOR_MODEL_MAPPINGS[discrim_config.model_id](**discrim_config.to_dict())

    return ae_model, discrim_model


def forward_step(batch, ae_model, discrim_model):
    """Forward step."""
    args = get_ae_args()
    inputs = batch["video"].to(torch.cuda.current_device())

    with torch.cuda.amp.autocast(dtype=args.mix_precision):
        outputs = ae_model(inputs)
        recon = outputs[0]
        posterior = outputs[1]
    
    gen_loss, discrim_loss = None, None
    # Generator Step
    if args.step_gen:
        with torch.cuda.amp.autocast(dtype=args.mix_precision):
            gen_loss, gen_log = discrim_model(
                inputs,
                recon,
                posterior,
                optimizer_idx=0,
                global_step=args.current_step,
                last_layer=ae_model.module.get_last_layer(),
            )
    # Discriminator Step
    if args.step_disc:
        with torch.cuda.amp.autocast(dtype=args.mix_precision):
            discrim_loss, discrim_log = discrim_model(
                inputs,
                recon,
                posterior,
                optimizer_idx=1,
                global_step=args.current_step,
                last_layer=None,
            )
    
    return gen_loss, discrim_loss


def train_valid_test_datasets_provider():
    """Build train, valid, and test datasets."""
    args = get_ae_args()
    train_dataset = build_ae_dataset(args.data.dataset_param)
    train_dataloader = build_ae_dataloader(
        train_dataset,
        args.data.dataloader_param
    )
    return train_dataloader, None, None


if __name__ == "__main__":
    pretrain_ae(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
    )
