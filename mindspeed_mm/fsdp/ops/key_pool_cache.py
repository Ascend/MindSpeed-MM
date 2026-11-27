"""Cache-owned KeyPool state for the GLM5-Next indexer.

The main KV cache is updated by attention before the indexer runs. Auxiliary
state follows the same reset and batch-reordering lifecycle. Rollback and
offloading are unsupported because KeyPool's persisted K/G state would need
the same transition.
"""

from dataclasses import dataclass, field

import torch
from transformers.cache_utils import DynamicCache


@dataclass
class PoolRow:
    length: int = 0
    first_slot: int | None = None
    state_cache: torch.Tensor | None = None
    pools: torch.Tensor | None = None

    def clone(self):
        return PoolRow(
            length=self.length,
            first_slot=self.first_slot,
            state_cache=None if self.state_cache is None else self.state_cache.clone(),
            pools=None if self.pools is None else self.pools.clone(),
        )


@dataclass
class PoolLayer:
    rows: list[PoolRow] = field(default_factory=list)
    raw_length: int = 0
    failed: bool = False

    def select(self, indices):
        # Repeated beams must not write into each other's partial-pool state.
        self.rows = [self.rows[index].clone() for index in indices]


class KeyPoolCacheMixin:
    @property
    def is_compileable(self):
        return False

    @property
    def is_croppable(self):
        return False

    def __init__(self, *args, **kwargs):
        if kwargs.get("offloading", False):
            raise NotImplementedError("KeyPool cache offloading is unsupported")
        super().__init__(*args, **kwargs)
        self.key_pool_states = {}

    def reset(self):
        super().reset()
        self.key_pool_states.clear()

    def reorder_cache(self, beam_idx):
        indices = beam_idx.detach().cpu().tolist()
        super().reorder_cache(beam_idx)
        for layer in self.key_pool_states.values():
            layer.select(indices)

    def batch_select_indices(self, indices):
        selected = indices.detach().cpu().tolist()
        super().batch_select_indices(indices)
        for layer in self.key_pool_states.values():
            layer.select(selected)

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        for layer in self.key_pool_states.values():
            layer.select([i for i in range(len(layer.rows)) for _ in range(repeats)])

    def crop(self, *args, **kwargs):
        if self.key_pool_states:
            raise NotImplementedError("KeyPool cache rollback is unsupported")
        return super().crop(*args, **kwargs)

    def offload(self, *args, **kwargs):
        raise NotImplementedError("KeyPool cache offloading is unsupported")


class KeyPoolDynamicCache(KeyPoolCacheMixin, DynamicCache):
    """Use for `past_key_values` whenever key_pool_v2 is selected."""
