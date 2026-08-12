from dataclasses import field
from typing import Any, Dict, List, Literal, Optional

from mindspeed_mm.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.params.feature_args import FeatureArguments
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.parallel_args import ParallelArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments


class GenerationArguments(BaseArguments):
    max_new_tokens: int = field(
        default=512,
        metadata={"help": "Maximum number of new tokens to generate, excluding input tokens."},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether to sample from the token distribution instead of greedy decoding."},
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={"help": "Sampling temperature. Only used when do_sample is enabled."},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={"help": "Nucleus sampling probability threshold. Only used when do_sample is enabled."},
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={"help": "Number of highest-probability tokens to sample from. Only used when do_sample is enabled."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Penalty applied to repeated tokens during generation. Default to 1.0."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use key-value cache to accelerate autoregressive generation."},
    )

    def to_generate_kwargs(self, distributed: bool = False) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "use_cache": self.use_cache,
        }
        if distributed:
            kwargs["synced_gpus"] = True
        if self.do_sample:
            for key in ("temperature", "top_p", "top_k"):
                value = getattr(self, key)
                if value is not None:
                    kwargs[key] = value
        return kwargs


class InferenceRuntimeArguments(BaseArguments):
    load: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the HF or distributed checkpoint containing the inference model weights."},
    )
    load_format: Literal["auto", "hf", "dcp"] = field(
        default="hf",
        metadata={"help": "Checkpoint format. 'auto' detects HF safetensors or DCP from the checkpoint path."},
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed used to initialize the inference runtime."},
    )
    use_deter_comp: bool = field(
        default=False,
        metadata={"help": "Whether to use deterministic computation for reproducible inference."},
    )
    init_model_with_meta_device: bool = field(
        default=False,
        metadata={"help": "Whether to initialize model weights on the meta device to reduce peak memory usage."},
    )
    plugin: List[str] = field(
        default_factory=list,
        metadata={"help": "Paths of model plugins to import before building the inference model."},
    )
    adapter: str = field(
        default="qwen3_5",
        metadata={"help": "Registered adapter used to preprocess inputs, generate outputs, and decode responses."},
    )
    processor_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the model processor assets used for inference."},
    )
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to enable the model processor's thinking mode when building the chat prompt."},
    )
    generation: GenerationArguments = field(
        default_factory=GenerationArguments,
        metadata={"help": "Text generation settings passed to the model's generate method."},
    )
    data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a JSON list of inference samples with text and optional image fields."},
    )

    def apply_to_training(self, training_args: TrainingArguments) -> None:
        training_args.seed = self.seed
        training_args.use_deter_comp = self.use_deter_comp
        training_args.init_model_with_meta_device = self.init_model_with_meta_device
        training_args.plugin = list(self.plugin)


class InferenceArguments(BaseArguments):
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    features: FeatureArguments = field(default_factory=FeatureArguments)
    inference: InferenceRuntimeArguments = field(default_factory=InferenceRuntimeArguments)

    def model_post_init(self, __context):
        self.inference.apply_to_training(self.training)
        self.training.compute_distributed_training(self.parallel)
