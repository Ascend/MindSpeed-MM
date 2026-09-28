# Copyright 2025 Bytedance Ltd. and/or its affiliates
from dataclasses import dataclass, field
from typing import List, Literal, Optional
import logging

from mindspeed_mm.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp import envs


logger = logging.getLogger(__name__)


class Tensorboard(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable TensorBoard recording of training scalars."},
    )
    dir: str = field(
        default="./tensorboard",
        metadata={"help": "Directory to write TensorBoard event files."},
    )
    interval: int = field(
        default=1,
        metadata={"help": "Reserved, not used. TensorBoard write frequency follows training.log_interval."},
    )


class Metrics(BaseArguments):
    """Backend-agnostic switches for which training metrics to collect.

    These control *what* is measured, independent of *where* it is written
    (TensorBoard today, possibly wandb etc. later). The output backends read
    these switches; they do not live under any single backend's config.
    """

    grad_norm_per_layer: bool = field(
        default=False,
        metadata={"help": "Record per-layer gradient norm (not zero-cost, disabled by default)."},
    )
    token_stats: bool = field(
        default=False,
        metadata={"help": "Record per-step token counts (valid/padding/total) and per-rank token distribution (min/max/ave/std)."},
    )
    token_stats_per_rank: bool = field(
        default=False,
        metadata={"help": "Record per-rank token counts: one curve per rank, the per-step histogram over every rank, the rank x step heatmap and the max/min/avg imbalance scalars. Implies token counting is enabled."},
    )
    token_stats_per_rank_list: Optional[List[int]] = field(
        default=None,
        metadata={"help": "Which ranks to record when token_stats_per_rank is on. None records all ranks; e.g. [1, 2, 3] records only ranks 1/2/3. It narrows the per-rank curves and the heatmap; the per-step histogram always covers every rank."},
    )

    def model_post_init(self, __context):
        # WORLD_SIZE is registered with a default of 1, so reading it without
        # `required=True` stays safe for single-process (non-torchrun) runs.
        self.world_size = envs.get("WORLD_SIZE")
        if self.token_stats_per_rank_list is not None:
            for rank in self.token_stats_per_rank_list:
                if not 0 <= rank < self.world_size:
                    raise ValueError(
                        f"metrics.token_stats_per_rank_list contains invalid rank {rank}, "
                        f"must be in [0, {self.world_size})."
                    )


class StaticParam(BaseArguments):
    level: str = field(
        default="level1",
        metadata={"help": "The info level of profiler."},
    )
    with_stack: bool = field(
        default=False,
        metadata={"help": "Whether to collect operator call stack info."},
    )
    with_memory: bool = field(
        default=False,
        metadata={"help": "Whether to collect the memory usage of the operator."},
    )
    record_shapes: bool = field(
        default=False,
        metadata={"help": "Whether to collect the innput shapes and input types of operators."},
    )
    with_cpu: bool = field(
        default=False,
        metadata={"help": "Whether to collect CPU events."},
    )
    save_path: str = field(
        default="./profiling",
        metadata={"help": "Direction to export the profiling result."},
    )
    start_step: int = field(
        default=10,
        metadata={"help": "Start step for profiling. `start_step = 0` means to start profiling from the beginning of training."},
    )
    end_step: int = field(
        default=11,
        metadata={"help": "End step for profiling."},
    )
    data_simplification: bool = field(
        default=False,
        metadata={"help": "Whether to enable the data simplification mode."},
    )
    aic_metrics_type: str = field(
        default="PipeUtilization",
        metadata={"help": "AI Core performance metric collection items."},
    )
    analyse_flag: bool = field(
        default=True,
        metadata={"help": "Whether to analyse profiling online."},
    )


class Profiler(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    profile_type: str = field(
        default="static",
        metadata={"help": "the type of profiling"},
    )
    ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={
            "help": "List of ranks to profile (default is rank 0 only)"
        },
    )
    static_param: StaticParam = field(default_factory=StaticParam)


class MemoryProfiler(BaseArguments):
    enable: bool = field(
        default=False,
        metadata={"help": "Enable memory profiling."},
    )
    start_step: int = field(
        default=1,
        metadata={"help": "Start step for memory profiling."},
    )
    end_step: int = field(
        default=2,
        metadata={"help": "End step for memory profiling."},
    )
    save_path: str = field(
        default="./memory_snapshot",
        metadata={"help": "Direction to export the memory profiling result."},
    )
    dump_ranks: List[int] = field(
        default_factory=lambda: [0],
        metadata={"help": "List of ranks to memory profile (default is rank 0 only)"},
    )
    stacks: Literal["python", "all"] = field(
        default="all",
        metadata={
            "help": "python, include Python, TorchScript, and inductor frames in tracebacks, all, additionally include C++ frames."},
    )
    max_entries: Optional[int] = field(
        default=None,
        metadata={"help": "Keep a maximum of `max_entries` alloc/free events in the recorded history recorded."},
    )
    mem_info: bool = field(
        default=False,
        metadata={"help": "Whether to print memory infos."},
    )


class ToolsArguments(BaseArguments):
    profile: Profiler = field(default_factory=Profiler)
    memory_profile: MemoryProfiler = field(default_factory=MemoryProfiler)
    tensorboard: Tensorboard = field(default_factory=Tensorboard)
    metrics: Metrics = field(
        default_factory=Metrics,
        metadata={"help": "Backend-agnostic switches for which training metrics to collect."},
    )
