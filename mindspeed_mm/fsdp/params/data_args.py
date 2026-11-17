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

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import logging

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DatasetAttr
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import DataArguments as BasicDataAruments
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments

logger = logging.getLogger(__name__)


class DataSetArguments(BaseArguments):
    dataset_type: str = field(
        metadata={"help": "Type of dataset to use."}
    )
    basic_parameters: BasicDataAruments = field(default_factory=BasicDataAruments)
    preprocess_parameters: Optional[ProcessorArguments] = field(default_factory=ProcessorArguments)
    attr: DatasetAttr = field(default_factory=DatasetAttr)


class CollateArguments(BaseArguments):
    collator_id: Optional[str] = field(
        default=None,
        metadata={"help": "Registered data collator class name."}
    )
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "This argument is deprecated. Use collator_id to select a registered data collator."},
    )
    ignore_pad_token_for_loss: bool = field(
        default=False,
        metadata={"help": ""}
    )
    pad_to_multiple_of: int = field(
        default=8,
        metadata={"help": "Pad sequences to a multiple of this value for efficient processing."}
    )


class DataloaderArguments(BaseArguments):
    dataloader_mode: str = field(metadata={"help": "Mode of dataloader."})
    sampler_type: str = field(metadata={"help": "Type of sampler to use."})
    shuffle: Optional[bool] = field(metadata={"help": "Whether to shuffle the data during training."})
    drop_last: bool = field(metadata={"help": "Whether to drop the last incomplete batch if dataset size is not divisible by batch size."})
    pin_memory: bool = field(metadata={"help": "Whether to pin memory for faster data transfer to GPU."})
    collate_param: CollateArguments = field(default_factory=CollateArguments)
    num_workers: int = field(default=2, metadata={"help": "Number of worker processes for data loading."})
    enable_preload: bool = field(
        default=False,
        metadata={"help": "Whether to enable async data preloading to overlap CPU-H2D transfer with training."},
    )
    infinite_sampler: bool = field(
        default=False,
        metadata={
            "help": "Whether the batch sampler keeps yielding across epoch boundaries so the "
            "DataLoader iterator never exhausts. The next epoch's batch tasks are dispatched "
            "while the last batch of the current epoch is still being consumed, removing the "
            "epoch‑boundary data stall. Only applies to the BaseRandomBatchSampler and SeedRandomBatchSampler train dataloaders. "
            "Note: Enabling this increases in‑flight batches to prefetch_factor * num_workers, "
            "which raises host memory usage. (When num_workers>0, prefetch_factor defaults to 2)."
        },
    )


class DataArguments(BaseArguments):
    dataset_param: DataSetArguments = field(default_factory=DataSetArguments)
    val_dataset_param: Optional[DataSetArguments] = field(default=None)
    dataloader_param: DataloaderArguments = field(default_factory=DataloaderArguments)
