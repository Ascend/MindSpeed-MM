from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union
import logging
import torch

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE
from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments

logger = logging.getLogger(__name__)


class FSDPPlanConfig(BaseArguments):
    """Configuration for Fully Sharded Data Parallelism (FSDP) plan."""
    ignored_modules: List[str] = field(default_factory=list)
    apply_modules: List[str] = field(default_factory=list)

    # mp_policy settings
    param_dtype: Optional[str] = None
    reduce_dtype: Optional[str] = None
    output_dtype: Optional[str] = None
    cast_forward_inputs: bool = True
    reshard_after_forward: bool = True

    # prefetch settings
    num_to_forward_prefetch: Optional[int] = 0
    num_to_backward_prefetch: Optional[int] = 0

    # pregather settings
    pregather: bool = False

    # fsdp2 hook manager
    hook_modules: Optional[List[str]] = None

    cpu_offload: bool = False


class TPPlanConfig(BaseArguments):
    """Configuration for Tensor Parallelism (TP) plan."""
    colwise_parallel: List[str] = field(default_factory=list)
    rowwise_parallel: List[str] = field(default_factory=list)
    sequence_parallel: List[str] = field(default_factory=list)


class EPPlanConfig(BaseArguments):
    """Configuration for Expert Parallelism (EP) plan for MoE models."""
    apply_modules: List[str] = field(default_factory=list)
    use_npu_fused_ops: bool = True
    dispatcher: Literal["alltoall", "allgather", "mc2", "chunkmoe"] = "alltoall"
    apply_efsdp_modules: List[str] = field(default_factory=list)
    _gradient_divide_factor: float = None

    moe_chunk_size: int = field(
        default=0,
        metadata={"help": "Chunk size for chunked MoE dispatch; 0 means no chunking (the whole token sequence is one chunk)."}
    )
    moe_num_chunks: int = field(
        default=0,
        metadata={"help": "Fixed number of chunks for chunked MoE dispatch; chunk length is derived as ceil(num_tokens / moe_num_chunks) and tail chunks are padded empty so every EP rank has the same chunk count. 0 means disabled; mutually exclusive with moe_chunk_size."}
    )
    chunk_moe_recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."}
    )

    def model_post_init(self, __context):
        if self.moe_chunk_size > 0 and self.moe_num_chunks > 0:
            logger.warning(
                "moe_chunk_size (%s) and moe_num_chunks (%s) are both set; "
                "moe_num_chunks takes precedence and moe_chunk_size is ignored.",
                self.moe_chunk_size, self.moe_num_chunks,
            )


class VocabPlanConfig(BaseArguments):
    """Configuration for Vocabulary Parallelism (VP) plan for Ngram embedding tables."""
    apply_modules: List[str] = field(default_factory=list)
    dispatcher: Literal["alltoall"] = field(
        default="alltoall",
        metadata={"help": "Routing backend of the lookup. Only the all-to-all dispatcher exists today; "
                          "the field stays so a plan keeps naming it explicitly."}
    )
    prefetch: bool = field(
        default=True,
        metadata={"help": "Look the table rows up before the layer that consumes them: the n-gram ids "
                          "depend only on input_ids, so the lookup is not repeated by recomputation "
                          "and can overlap with the preceding layers."}
    )
    prefetch_split: bool = field(
        default=False,
        metadata={"help": "Split that prefetch in two: route the ids and start the cpu-to-device copy "
                          "before the decoder layers, then cast and send the rows back inside the layer "
                          "that consumes them. The copy then runs under the layers in between, at the "
                          "cost of one extra embedding all-to-all in the recomputation pass."}
    )
    host_by_cpu: bool = field(
        default=True,
        metadata={"help": "True keeps the n-gram table in host (cpu) memory and ships the touched rows "
                          "per lookup: row-sparse gradient, a cpu-side sparse optimizer, dedup and "
                          "prefetch. False keeps the table on the accelerator, sharded over the "
                          "vocabulary group: dense gradient, the model's own optimizer, and none of the "
                          "cpu-side machinery. The two are numerically equivalent."}
    )
    dedup: bool = field(
        default=True,
        metadata={"help": "Collapse repeated n-gram ids before routing and expand them after the rows "
                          "come back, so both all-to-alls, the cpu fetch and the step-end merge carry "
                          "distinct rows only. The forward output is unchanged. Costs one unique() over "
                          "the lookup ids, which pays off whenever a step repeats an id; turn it off "
                          "for a corpus where almost every id is distinct."}
    )
    async_offload: bool = field(
        default=True,
        metadata={"help": "Move the row transfers between cpu and accelerator onto a side stream so "
                          "they overlap the compute stream. Covers the backward row-gradient copy and, "
                          "with prefetch_split, the forward copy. Off makes every transfer block."}
    )

    def model_post_init(self, __context):
        # Three renames landed together, and the base class accepts unknown keys, so a stale plan would
        # otherwise be ignored and silently take the cpu path.
        legacy = {
            "device": "host_by_cpu (true: table on cpu, false: table on the accelerator)",
            "async_transfer": "async_offload",
            "device_sort": "environment variable MM_VP_DEVICE_SORT",
            "device_merge": "environment variable MM_VP_DEVICE_MERGE",
        }
        stale = [f"{key} -> {hint}" for key, hint in legacy.items() if key in (self.model_extra or {})]
        if stale:
            raise ValueError("[Vocabulary Parallel] stale vocab_plan keys: " + "; ".join(stale))


class RecomputePlanConfig(BaseArguments):
    """Configuration for recompute plan."""
    apply_modules: List[str] = field(default_factory=list)
    use_reentrant: bool = False


class ParallelArguments(BaseArguments):
    data_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "Size of data parallelism. If None, calculated automatically."}
    )

    fully_shard_parallel_size: Union[str, int] = field(
        default="auto",
        metadata={"help": "Fully Sharded Data Parallel size. (Sharding parameters)"}
    )

    fsdp_plan: FSDPPlanConfig = field(default_factory=FSDPPlanConfig)

    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor Parallel size. (Cols/Rows splitting)"}
    )
    tp_plan: TPPlanConfig = field(default_factory=TPPlanConfig)

    ring_attention_size: int = 1 # Size for Ring Attention
    ulysses_parallel_size: int = 1 # Size for Ulysses parallelism
    # Size of the orthogonal CP group used to all-gather K/V after Ulysses.
    # Total context parallel size is ulysses_parallel_size * kvallgather_parallel_size.
    # Ring attention is intentionally not combined with kvallgather (mutually exclusive).
    kvallgather_parallel_size: int = 1
    expert_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert Parallel size for MoE models."}
    )
    expert_fully_shard_parallel_size: int = field(
        default=None,
        metadata={"help": "FSDP size inside Expert Parallel groups."}
    )
    expert_data_parallel_size: int = field(
        default=1,
        metadata={"help": "Expert Data Parallel size for MoE models."}
    )
    ep_plan: EPPlanConfig = field(default_factory=EPPlanConfig)

    recompute: bool = field(
        default=False,
        metadata={"help": "Whether to enable Gradient Checkpointing (Activation Recomputation)."}
    )
    recompute_plan: RecomputePlanConfig = field(default_factory=RecomputePlanConfig)

    def model_post_init(self, __context):
        self.local_rank = envs.get("LOCAL_RANK", required=True)
        self.global_rank = envs.get("RANK", required=True)
        self.world_size = envs.get("WORLD_SIZE", required=True)

        if self.fully_shard_parallel_size == "auto":
            # If -1, use all remaining processes after tensor parallelism for FSDP
            self.fully_shard_parallel_size = self.world_size // self.tensor_parallel_size
        else:
            self.fully_shard_parallel_size = int(self.fully_shard_parallel_size)

        if self.expert_fully_shard_parallel_size is None:
            self.expert_fully_shard_parallel_size = self.world_size // (self.expert_parallel_size * self.expert_data_parallel_size)

        if (
            self.world_size
            % (
                self.tensor_parallel_size
                * self.ring_attention_size
                * self.ulysses_parallel_size
                * self.kvallgather_parallel_size
            )
            != 0
        ):
            raise ValueError(
                f"World size should be a multiple of tensor_parallel_size: {self.tensor_parallel_size}, "
                f"ulysses_parallel_size: {self.ulysses_parallel_size}, "
                f"kvallgather_parallel_size: {self.kvallgather_parallel_size}, "
                f"ring_attention_size: {self.ring_attention_size}."
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
            * self.ring_attention_size
            * self.ulysses_parallel_size
            * self.kvallgather_parallel_size
        )
        if self.data_parallel_size is None:
            self.data_parallel_size = dp_size

        if self.data_parallel_size != dp_size:
            raise ValueError(
                f"data_parallel_size should match world_size / (tp * ulysses * kvallgather * ring): "
                f"tp={self.tensor_parallel_size}, ulysses={self.ulysses_parallel_size}, "
                f"kvallgather={self.kvallgather_parallel_size}, ring={self.ring_attention_size}."
            )

        cp_size = self.ring_attention_size * self.ulysses_parallel_size * self.kvallgather_parallel_size
        if self.fully_shard_parallel_size < cp_size:
            raise ValueError("fully_shard_parallel_size should be >= ring_attention_size * ulysses_parallel_size * kvallgather_parallel_size.")
        if self.fully_shard_parallel_size % cp_size != 0:
            raise ValueError("fully_shard_parallel_size should be divisible by ring_attention_size * ulysses_parallel_size * kvallgather_parallel_size.")
        if self.tensor_parallel_size != 1:
            raise ValueError("Tensor parallel size not supported yet.")
        if self.ring_attention_size != 1 and not IS_NPU_AVAILABLE:
            raise ValueError("Ring Attention only support on NPU.")
        if self.kvallgather_parallel_size != 1 and self.ring_attention_size != 1:
            raise ValueError("KV AllGather does not support Ring Attention; should set ring_attention_size=1.")

        # edp=1 → FSDP；edp>1 → HSDP
        if self.world_size % (self.expert_fully_shard_parallel_size * self.expert_parallel_size * self.expert_data_parallel_size) != 0:
            raise ValueError(
                f"world_size({self.world_size}) must be divisible by "
                f"expert_fully_shard_parallel_size({self.expert_fully_shard_parallel_size}) * "
                f"expert_parallel_size({self.expert_parallel_size}). "
                f"expert_data_parallel_size({self.expert_data_parallel_size}). "
                f"This is required for constructing the EP mesh ('edp', 'efsdp', 'ep')."
            )
        if self.expert_data_parallel_size > 1 and self.expert_fully_shard_parallel_size > 1:
            if self.expert_parallel_size == 1:
                raise ValueError(
                    f"HSDP mode requires expert_parallel_size > 1, "
                    f"got {self.expert_parallel_size}"
                )
