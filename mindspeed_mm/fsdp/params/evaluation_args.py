from dataclasses import field
from typing import Optional

from mindspeed_mm.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.params.inference_args import InferenceArguments


class EvaluationRuntimeArguments(BaseArguments):
    dataset_type: str = field(
        default="vqa2_val",
        metadata={"help": "Dataset type used for evaluation."},
    )
    dataset_path: str = field(
        metadata={"help": "Root path of the evaluation dataset."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of evaluation samples. None uses the full dataset."},
    )
    result_output_path: str = field(
        default="./evaluation_outputs",
        metadata={"help": "Directory used to save evaluation predictions and metrics."},
    )


class EvaluationArguments(InferenceArguments):
    evaluation: EvaluationRuntimeArguments = field(default_factory=EvaluationRuntimeArguments)
