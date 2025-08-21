# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import mindspeed.megatron_adaptor
from mindspeed_mm.tasks.rl.soragrpo.dataset.latent_flux_rl_datasets import LatentDataset
from mindspeed_mm.tasks.rl.soragrpo.flux_grpo_trainer import FluxGRPOTrainer


def train_valid_test_datasets_provider(args):
    """Build train, valid, and test datasets."""
    train_dataset = LatentDataset(args.mm.data.dataset_param.basic_parameters.data_path, args.num_latent_t, args.cfg)
    return train_dataset


if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True

    trainer = FluxGRPOTrainer(
        train_valid_test_dataset_provider=train_valid_test_datasets_provider,
    )
    trainer.train()
