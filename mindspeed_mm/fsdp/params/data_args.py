from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import logging

from mindspeed_mm.data.data_utils.func_utils.convert import DatasetAttr
from mindspeed_mm.data.data_utils.func_utils.convert import DataArguments as BasicDataAruments
from mindspeed_mm.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.params.utils import allow_extra_fields

logger = logging.getLogger(__name__)


@allow_extra_fields
@dataclass
class DataSetArguments:
    dataset_type: str = field(
        metadata={"help": "Type of dataset to use."}
    )
    basic_parameters: BasicDataAruments = field(default_factory=BasicDataAruments)
    preprocess_parameters: ProcessorArguments = field(default_factory=ProcessorArguments)
    attr: DatasetAttr = field(default_factory=DatasetAttr)

    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


@allow_extra_fields
@dataclass
class CollateArguments:
    model_name: str = field(metadata={"help": "Name of the model for which collation is configured."})
    ignore_pad_token_for_loss: bool = field(
        default=False,
        metadata={"help": ""}
    )


@allow_extra_fields
@dataclass
class DataloaderArguments:
    dataloader_mode: str = field(metadata={"help": "Mode of dataloader."})
    sampler_type: str = field(metadata={"help": "Type of sampler to use."})
    shuffle: bool = field(metadata={"help": "Whether to shuffle the data during training."})
    drop_last: bool = field(metadata={"help": "Whether to drop the last incomplete batch if dataset size is not divisible by batch size."})
    pin_memory: bool = field(metadata={"help": "Whether to pin memory for faster data transfer to GPU."})
    collate_param: CollateArguments = field(default_factory=CollateArguments)
    num_workers: int = field(default=2, metadata={"help": "Number of worker processes for data loading."})

    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        result = asdict(self)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result


@allow_extra_fields
@dataclass
class DataArguments:
    dataset_param: DataSetArguments = field(default_factory=DataSetArguments)
    dataloader_param: DataloaderArguments = field(default_factory=DataloaderArguments)

