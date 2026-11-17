# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from torchdata.stateful_dataloader import StatefulDataLoader

from mindspeed_mm.fsdp.data import build_mm_dataset
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.config.config_manager import ConfigManager
from mindspeed_mm.fsdp.log import print_rank


def get_cosyvoice_dataloader(args):
    """Build training dataloader with proper parallel partitioning."""
    print_rank(logging.info, "Prepare data")
    data_config = args.data

    datasets = build_mm_dataset(data_config.dataset_param)
    dataloader_param = data_config.dataloader_param.to_dict()
    train_dataloader = StatefulDataLoader(
        datasets,
        batch_size=None,
        pin_memory=dataloader_param.get('pin_memory'),
        pin_memory_device=get_device_type(),
        num_workers=dataloader_param.get('num_workers'),
        prefetch_factor=dataloader_param.get('prefetch_factor')
    )

    return train_dataloader


if __name__ == "__main__":
    # Entry point for training script
    args = ConfigManager(config_class=Arguments).load_and_parse()
    trainer = Trainer(args=args, dataloader_provider=get_cosyvoice_dataloader)
    trainer.train()
