# pylint: skip-file
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from pydantic import model_validator

from mindspeed_mm.config.arguments.base_args import BaseArguments


class SwapPlanConfig(BaseArguments):
    """Physical resources of the shared swap cache (HBM<->DDR async swap),
    owned by the SwapManager. Memory features (op replay, act stash, ...)
    obtain the shared cache from the manager as pure tenants.

    Execution is always asynchronous (swap stream + event ordering, no host
    block); hosts without a usable device fail fast at cache construction.
    capacity_mb selects the mode: <0 no management (pass-through), =0
    deterministic eviction (the compute stream waits the swap stream), >0
    asynchronous capacity management (default 1024)."""
    capacity_mb: float = field(
        default=1024.0,
        metadata={"help": "HBM capacity budget (MB) for swapped payloads, shared by all tenants. <0: no capacity management (pass-through, nothing is evicted); =0: every put evicts and the compute stream waits the swap stream (deterministic, no overlap); >0: asynchronous capacity management. Default 1024; tune to the per-step swap volume (a stash aiming to save HBM needs capacity below its evicted volume)."},
    )
    keep_rate: float = field(
        default=0.0,
        metadata={"help": "HBM retention ratio [0,1]: the soft limit is capacity x keep_rate, the line above which proactive eviction starts (projected reading, after in-flight D2H). 0 = evict right after put (max async D2H overlap); 1 = retain the full capacity budget, evict only when physically over capacity."},
    )
    enable_prefetch: bool = field(
        default=True,
        metadata={"help": "Whether enable transition-graph chain prefetch for the swap cache. Turn off (with fifo eviction) for structurally variable workloads: the graph is learned once from the first iteration's pop order."},
    )
    cpu_arena_slab_size_mb: Optional[float] = field(
        default=2048.0,
        metadata={"help": "CPU slab arena slab size (MB). None selects the on-demand arena (per-tensor exact allocation, released immediately)."},
    )
    cpu_arena_pool_policy: str = field(
        default="standard",
        metadata={"help": "CPU arena pool policy: 'all' (retain all empty slabs), 'standard' (retain standard slabs, release oversized on empty), 'none' (release immediately). Only effective with cpu_arena_slab_size_mb."},
    )
    pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether arena slabs use pinned (page-locked) memory. False is the escape hatch when host pinned memory is exhausted; non-pinned copies then execute synchronously (torch behavior)."},
    )
    eviction_policy: Literal["fifo", "belady"] = field(
        default="fifo",
        metadata={"help": "Swap cache eviction policy: 'fifo' or 'belady'. With keep_rate=0 (default) every put evicts everything and the two policies are equivalent; belady matters when keep_rate > 0 (evict the farthest-future use)."},
    )

    @model_validator(mode='after')
    def _check_keep_rate(self):
        if not (0.0 <= self.keep_rate <= 1.0):
            raise ValueError(
                f"swap_plan.keep_rate must be in [0, 1], got {self.keep_rate}")
        return self


class OpReplayScopeConfig(BaseArguments):
    """One op-replay scope: a region (apply_modules) with its own op whitelist
    and RNG policy. Scopes are independent; nested zones resolve to the
    innermost scope, and a module matched by several scopes is patched by the
    first one in the list (list order = priority)."""
    name: Optional[str] = field(
        default=None,
        metadata={"help": "Scope label, used in logs only."},
    )
    apply_modules: List[str] = field(
        default_factory=list,
        metadata={"help": "Modules marking this replay zone. Must lie within the subtree of modules "
                          "covered by the enclosing recompute_plan.apply_modules (non-reentrant "
                          "checkpoint): the dispatch mode only activates inside checkpoint "
                          "boundaries, so zones outside coverage never replay."}
    )
    save_rng: bool = field(
        default=True,
        metadata={"help": "Whether record/replay post-op RNG snapshots for this scope's cached ops "
                          "(keeps the recompute RNG stream aligned with forward op by op)"},
    )
    cache_ops: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Whitelist of aten op names to cache in this scope (e.g. "
                          "['aten.mm.default', 'aten.linear.default']). None falls back to the "
                          "built-in default whitelist."},
    )


class RecomputePlanConfig(BaseArguments):
    """Configuration for recompute plan"""
    apply_modules: List[str] = field(default_factory=list)
    use_reentrant: bool = False
    op_replay_scopes: List[OpReplayScopeConfig] = field(
        default_factory=list,
        metadata={"help": "Op replay (selective activation compression) scopes riding the recompute "
                          "boundary: each scope is (region x whitelist x save_rng). Non-empty "
                          "enables the feature; only effective with use_reentrant=false (it "
                          "activates via the non-reentrant checkpoint's context_fn). Nested zones "
                          "resolve to the innermost scope; overlapping scopes: first in the list wins."},
    )


class ChunkLossPlanConfig(BaseArguments):
    apply_module: str = field(
        default="lm_head",
        metadata={"help": "module that applied chunk loss"}
    )
    chunk_size: Optional[int] = field(
        default=None,
        metadata={"help": "Chunk size. Legacy (impl_type='legacy'): token-chunk size along seq, "
                          "None falls back to 1024. CCE (impl_type='cce'): reused as the outer "
                          "seq_chunk_size, None passes through to the kernel as single segment "
                          "(no seq chunking)."},
    )
    total_chunk_size: int = field(
        default=4096,
        metadata={"help": "Size of total chunk loss"},
    )
    impl_type: Literal["legacy", "cce"] = field(
        default="legacy",
        metadata={"help": "Chunk loss implementation: 'legacy' or 'cce'"},
    )
    vocab_tile_size: Optional[int] = field(
        default=4096,
        metadata={"help": "Vocab tile size for CCE loss"},
    )


class LossArguments(BaseArguments):
    loss_type: Optional[str] = field(
        default="raw",
        metadata={"help": "Type of loss function type, If ot provided, will be computed based on raw model loss function"},
    )
    router_aux_loss_coef: float = field(
        default=0.0,
        metadata={"help": "Router Auxiliary Loss Coefficient"},
    )
    router_aux_loss_offload: bool = field(
        default=False,
        metadata={"help": "Whether apply router auxiliary loss offload"},
    )
    router_aux_loss_type: Literal["local", "global"] = field(
        default="local",
        metadata={
            "help": "Router auxiliary loss scope. 'local' uses per-microbatch statistics; "
                    "'global' follows Megatron-LM global_aux_loss semantics by aggregating "
                    "expert token counts across the HSDP group. Global mode requires "
                    "loss_type='per_token_loss'; when recompute is enabled, "
                    "recompute_plan.use_reentrant must be false."
        },
    )
    router_aux_loss_use_attention_mask: bool = field(
        default=False,
        metadata={
            "help": "Whether global router auxiliary loss excludes masked tokens using a 2D "
                    "[batch, sequence] attention mask. When false, every router input row is "
                    "included."
        },
    )

    @model_validator(mode="after")
    def _validate_global_aux_loss(self):
        if (
            self.router_aux_loss_type == "global"
            and self.router_aux_loss_coef > 0.0
            and self.loss_type != "per_token_loss"
        ):
            raise ValueError(
                "Global router aux loss requires loss_type='per_token_loss'."
            )
        return self


class ActivationOffloadPlanConfig(BaseArguments):
    apply_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": "module that applied activation offload"}
    )
    impl: Literal["legacy", "stash"] = field(
        default="legacy",
        metadata={"help": "Activation offload implementation. 'legacy': per-layer hidden_states "
                          "swap with resize_(0), self-managed stream and prefetch heuristics. "
                          "'stash': saved_tensors_hooks tenant (act stash) on the shared swap "
                          "cache; all physical behavior (capacity, async, prefetch, eviction) is "
                          "governed by swap_plan."}
    )


class ChunkMbsPlanConfig(BaseArguments):
    apply_modules: List[str] = field(
        default=None,
        metadata={"help": "module that applied chunkmbs"}
    )

    chunk_mbs: int = field(
        default=1,
        metadata={"help": "chunk_mbs, chunked micro batch size"}
    )

    batch_dim: int = field(
        default=0,
        metadata={"help": "chunk_mbs, batchsize dim"}
    )

    chunk_arg_indexs: List[int] = field(
        default=[0],
        metadata={"help": "chunk_mbs, chunk args indexs"}
    )

    chunk_kwarg_names: List[str] = field(
        default=[],
        metadata={"help": "chunk_mbs, chunk kwarg names"}
    )


class EPBalancePlanConfig(BaseArguments):
    max_dup_experts_num: int = field(
        default=2,
        metadata={"help": "max dup experts num"},
    )


class FeatureArguments(BaseArguments):
    swap_plan: SwapPlanConfig = field(default_factory=SwapPlanConfig)

    recompute_plan: RecomputePlanConfig = field(default_factory=RecomputePlanConfig)

    loss_cfg: LossArguments = field(default_factory=LossArguments)

    enable_chunk_loss: bool = field(
        default=False,
        metadata={"help": "Whether apply chunkloss for loss compute"},
    )
    enable_dynamic_chunk_loss: bool = field(
        default=False,
        metadata={"help": "Whether apply dynamic chunkloss for loss compute"},
    )
    chunkloss_plan: ChunkLossPlanConfig = field(default_factory=ChunkLossPlanConfig)

    enable_activation_offload: bool = field(
        default=False,
        metadata={"help": "Whether apply activation offload"}
    )
    activation_offload_plan: ActivationOffloadPlanConfig = field(default_factory=ActivationOffloadPlanConfig)

    enable_chunk_mbs: bool = field(
        default=False,
        metadata={"help": "Whether apply chunk_mbs"}
    )
    chunkmbs_plan: ChunkMbsPlanConfig = field(default_factory=ChunkMbsPlanConfig)

    enable_ep_balance: bool = field(
        default=False,
        metadata={"help": "Whether apply ep balance strategy"}
    )
    ep_balance_plan: EPBalancePlanConfig = field(default_factory=EPBalancePlanConfig)

    skip_moe_pad_tokens: bool = field(
        default=False,
        metadata={"help": "Whether skip moe pad tokens"}
    )

    # Field normalization for the chunk loss plan. This is the single authority that
    # owns enable/disable and impl-specific normalization for chunkloss_plan:
    #
    #   * disabled        -> chunk_size / vocab_tile_size = None  (plain loss path)
    #   * enabled legacy  -> chunk_size None falls back to 1024; vocab tile stripped
    #   * enabled cce     -> vocab tile kept; chunk_size None stays None (opaque, kernel)
    @model_validator(mode="after")
    def _normalize_loss_plan(self):
        plan = self.chunkloss_plan
        impl = plan.impl_type

        if impl == "cce" and not self.enable_chunk_loss:
            raise ValueError(
                "chunkloss_plan.impl_type='cce' requires features.enable_chunk_loss=True, "
                "otherwise CCE loss would be silently disabled."
            )

        if not self.enable_chunk_loss:
            # chunk loss disabled: pass None through so loss build takes the plain path
            if plan.chunk_size is not None:
                plan.chunk_size = None
            if plan.vocab_tile_size is not None:
                plan.vocab_tile_size = None
            return self

        # enabled: impl-specific normalization
        if impl == "legacy":
            if plan.chunk_size is None:
                plan.chunk_size = 1024
        if impl != "cce" and plan.vocab_tile_size is not None:
            # vocab tile only applies to cce
            plan.vocab_tile_size = None
        return self
