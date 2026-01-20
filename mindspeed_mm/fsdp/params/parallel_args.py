from dataclasses import dataclass, field
from typing import List, Literal, Optional
import logging
import os

from mindspeed_mm.fsdp.params.utils import allow_extra_fields


logger = logging.getLogger(__name__)


@dataclass
class FSDPPlanConfig:
    """Configuration for Fully Sharded Data Parallelism (FSDP) plan."""
    ignored_modules: List[str] = None
    apply_modules: List[str] = None
    enable_mixed_precision: bool = True


@dataclass
class TPPlanConfig:
    """Configuration for Tensor Parallelism (TP) plan."""
    colwise_parallel: List[str] = None
    rowwise_parallel: List[str] = None
    sequence_parallel: List[str] = None



@dataclass
class EPPlanConfig:
    """Configuration for Expert Parallelism (EP) plan for MoE models."""
    apply_modules: List[str] = None
    dispatcher: Literal["eager", "fused", "mc2"] = "eager"
    apply_efsdp_modules: List[str] = None
    _gradient_divide_factor: float = None


@allow_extra_fields
@dataclass
class ParallelArguments():
    data_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of data parallelism. If None, calculated automatically."}
    )

    fully_shard_parallel_size: str = field(
        default="auto",
        metadata={"help": "Fully Sharded Data Parallel size. (Sharding parameters)"}
    )

    fsdp_plan: FSDPPlanConfig = field(default_factory=FSDPPlanConfig)

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor Parallel size. (Cols/Rows splitting)"}
    )
    tp_plan: TPPlanConfig = field(default_factory=TPPlanConfig)

    context_parallel_size: int = 1 # Size for context parallelism(ring)
    ulysses_parallel_size: int = 1 # Size for Ulysses parallelism

    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert Parallel size for MoE models."}
    )
    expert_fully_shard_parallel_size: int = field(
        default=None,
        metadata={"help": "FSDP size inside Expert Parallel groups."}
    )
    ep_plan: EPPlanConfig = field(default_factory=EPPlanConfig)

    recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."}
    )
    recompute_plan: List[str] = None

    def __post_init__(self):
        self.local_rank = int(os.getenv("LOCAL_RANK"))
        self.global_rank = int(os.getenv("RANK"))
        self.world_size = int(os.getenv("WORLD_SIZE"))

        if self.fully_shard_parallel_size == "auto":
            # If -1, use all remaining processes after tensor parallelism for FSDP
            self.fully_shard_parallel_size = self.world_size // self.tensor_parallel_size
        else:
            self.fully_shard_parallel_size = int(self.fully_shard_parallel_size)

        if self.expert_fully_shard_parallel_size is None:
            self.expert_fully_shard_parallel_size = self.world_size // self.expert_parallel_size

        if (
            self.world_size
            % (
                self.tensor_parallel_size
                * self.context_parallel_size
                * self.ulysses_parallel_size
            )
            != 0
        ):
            raise ValueError(
                f"World size should be a multiple of tensor_parallel_size: {self.tensor_parallel_size}, ulysses_parallel_size: {self.ulysses_parallel_size}, context_parallel_size: {self.context_parallel_size}."
            )
        if (
            self.world_size
            % (
                self.tensor_parallel_size
                * self.fully_shard_parallel_size
            )
            != 0
        ):
            raise ValueError(
                f"World size should be a multiple of tensor_parallel_size: {self.tensor_parallel_size}, fully_shard_parallel_size: {self.fully_shard_parallel_size}."
            )

        dp_size = self.world_size // (
            self.tensor_parallel_size
            * self.context_parallel_size
            * self.ulysses_parallel_size
        )
        if self.data_parallel_size is None:
            self.data_parallel_size = dp_size

        if self.data_parallel_size != dp_size:
            raise ValueError(f"data_parallel_size should be equal to tensor_parallel_size: {self.tensor_parallel_size}, ulysses_parallel_size: {self.ulysses_parallel_size}, context_parallel_size: {self.context_parallel_size}.")

        if self.fully_shard_parallel_size < self.context_parallel_size * self.ulysses_parallel_size:
            raise ValueError("fully shard parallel size should be greater the context_parallel_size * ulysses_parallel_size.")
        if self.tensor_parallel_size != 1:
            raise ValueError("Tensor parallel size not supported yet.")
        if self.context_parallel_size != 1:
            raise ValueError("Context parallel size not supported yet.")
