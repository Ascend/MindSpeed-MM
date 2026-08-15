"""SwapManager: single owner of the shared swap cache and its lifecycle.

Holds the one SwapCache (swap stream + pinned arena + capacity budget) built
from swap_plan. Memory features (op replay, act stash, ...) obtain it via
get_cache(tenant) and use it as pure tenants: they never construct caches and
never delimit iterations. The FeaturesApplier registers step_end() as the
TrainEngine's on_step_end callback, fired once per training step after
optimizer.zero_grad(); a custom training loop must drive step_end() itself —
without it, prefetch never arms, stats are not archived and the put history
grows without bound (correctness is unaffected).
"""
import logging

from .slab_arena import SlabArena
from .swap_core import SwapCache
from ...params.feature_args import SwapPlanConfig


logger = logging.getLogger(__name__)


class SwapManager:
    """Owns the shared SwapCache and its iteration boundary.

    get_cache(tenant) returns the single shared instance (created lazily on
    the first call); the tenant name is a registration label for logging only.
    step_end() delimits one iteration (cache.clear(): transition-graph
    learning + stats archival + resource release) and must be called exactly
    once per training step — a second call within the same step archives empty
    stats over the real ones.
    """

    def __init__(self, plan):
        # plan is always present (FeatureArguments.swap_plan has a default
        # factory); None is only a defensive fallback for direct construction.
        self._plan = plan if plan is not None else SwapPlanConfig()
        self._cache = None
        self._tenants = []

    def get_cache(self, tenant: str = "default") -> SwapCache:
        if self._cache is None:
            # capacity_mb < 0 -> pass-through (None); 0 -> deterministic
            # eviction; >0 -> async capacity management (default 1024).
            capacity_mb = self._plan.capacity_mb
            capacity_bytes = None if capacity_mb < 0 else int(capacity_mb * 1024 * 1024)
            soft_limit_bytes = (int(capacity_bytes * self._plan.keep_rate)
                                if capacity_bytes is not None else 0)
            # The arena is composed here, not inside SwapCache: pooled when a
            # slab size is configured (empty slabs kept for reuse), otherwise
            # the on-demand arena (exact per-tensor allocation).
            slab_size_mb = self._plan.cpu_arena_slab_size_mb
            pin_memory = self._plan.pin_memory
            if slab_size_mb is not None:
                arena = SlabArena(
                    slab_size_bytes=max(1, int(slab_size_mb * 1024 * 1024)),
                    device='cpu',
                    pin_memory=pin_memory,
                    pool_policy=self._plan.cpu_arena_pool_policy,
                )
            else:
                arena = SlabArena.on_demand(pin_memory=pin_memory)
            self._cache = SwapCache(
                capacity_bytes=capacity_bytes,
                soft_limit_bytes=soft_limit_bytes,
                cpu_arena=arena,
                eviction_policy=self._plan.eviction_policy,
                enable_prefetch=self._plan.enable_prefetch,
            )
        if tenant not in self._tenants:
            self._tenants.append(tenant)
            logger.info(f"Swap cache shared with tenant: {tenant}")
        return self._cache

    def step_end(self):
        """Drive the per-step iteration boundary of the shared cache."""
        if self._cache is not None:
            self._cache.clear()
