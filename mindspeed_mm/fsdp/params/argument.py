# pylint: skip-file
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints
from dataclasses import MISSING, asdict, dataclass, field, fields
import sys
import os

import yaml

from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import has_media_columns
from mindspeed_mm.fsdp.params.data_args import DataArguments
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.fsdp.params.parallel_args import ParallelArguments
from mindspeed_mm.fsdp.params.tools_args import ToolsArguments
from mindspeed_mm.fsdp.params.utils import instantiate_dataclass
from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments


def _is_packed_pretrain_dataset(dataset_param):
    if dataset_param is None or dataset_param.dataset_type != "huggingface":
        return False

    # Text-only pre-training keeps the legacy fixed-block packing and has no batch size limit.
    if not has_media_columns(dataset_param.attr):
        return False

    basic_parameters = dataset_param.basic_parameters
    return basic_parameters.stage == "pretrain" and basic_parameters.packing is not False


def _validate_packed_pretrain_micro_batch_size(data_args, training_args):
    dataset_param = data_args.dataset_param
    if dataset_param is None:
        return

    if _is_packed_pretrain_dataset(dataset_param) and training_args.micro_batch_size != 1:
        raise ValueError(
            "Multimodal pre-training packing requires training.micro_batch_size=1, "
            f"but got {training_args.micro_batch_size}. "
            "Set training.micro_batch_size to 1 or disable packing."
        )

    if training_args.val_interval <= 0:
        return

    val_dataset_param = data_args.val_dataset_param
    if val_dataset_param is None and dataset_param.basic_parameters.val_dataset:
        val_dataset_param = dataset_param

    val_micro_batch_size = training_args.val_micro_batch_size or training_args.micro_batch_size
    if _is_packed_pretrain_dataset(val_dataset_param) and val_micro_batch_size != 1:
        raise ValueError(
            "Multimodal pre-training packing requires the effective validation micro batch size to be 1, "
            f"but got {val_micro_batch_size}. "
            "Set training.val_micro_batch_size to 1 or disable packing for the validation dataset."
        )


class Arguments(BaseArguments):
    """Root argument class: model/data/parallel/training four types of parameters"""
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    tools: ToolsArguments = field(default_factory=ToolsArguments)
    features: FeatureArguments = field(default_factory=FeatureArguments)

    def model_post_init(self, __context):
        _validate_packed_pretrain_micro_batch_size(self.data, self.training)
        self.training.compute_distributed_training(self.parallel)


def parse_args(dataclass_type: Arguments):
    """Parse YAML arguments into structured dataclasses."""
    if not issubclass(dataclass_type, Arguments):
        raise ValueError(f"Expected dataclass_type to be a subclass of `Arguments`, but got {dataclass_type}")

    # Parse command line arguments
    cmd_args = sys.argv[1:]

    # Validate that a configuration file was provided
    if not cmd_args:
        raise ValueError(
            "❌ No configuration file provided.\n"
        )

    # Handle config file input
    input_data = {}
    # Validate file extension to ensure it's a YAML configuration file
    if not (cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml")):
        raise ValueError(
            f"❌ Invalid configuration file: '{cmd_args[0]}'\n"
            f"Expected a YAML file with extension .yaml or .yml\n"
        )
    with open(os.path.abspath(cmd_args[0]), encoding="utf-8") as f:
        input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)

    # Instantiate the Arguments dataclass from YAML data
    args = instantiate_dataclass(dataclass_type, input_data)

    # Critical: Resolve dependencies between different configuration sections
    # and validate parameter consistency across the entire configuration
    args.training.compute_distributed_training(args.parallel)

    return args
