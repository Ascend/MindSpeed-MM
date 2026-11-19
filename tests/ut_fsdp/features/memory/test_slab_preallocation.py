from types import SimpleNamespace

import pytest
import torch

from mindspeed_mm.fsdp.features.memory.slab_arena import SlabArena
from mindspeed_mm.fsdp.params.feature_args import SwapPlanConfig
from mindspeed_mm.fsdp.train import train_engine as train_engine_mod
from mindspeed_mm.fsdp.train.train_engine import TrainEngine


def test_preallocate_first_touches_and_reuses_standard_slabs():
    arena = SlabArena(slab_size_bytes=4096, pin_memory=False, pool_policy="standard")

    stats = arena.preallocate(3)

    assert stats["total_slabs"] == 3
    assert stats["free_slabs"] == 3
    assert stats["total_capacity"] == 3 * 4096
    assert all(torch.count_nonzero(slab.buf).item() == 0 for slab in arena._free)

    data_ptrs = {slab.buf.data_ptr() for slab in arena._free}
    arena.preallocate(2)
    assert arena.stats()["total_slabs"] == 3
    assert {slab.buf.data_ptr() for slab in arena._free} == data_ptrs

    arena.preallocate(4)
    assert arena.stats()["total_slabs"] == 4
    assert arena.stats()["total_capacity"] == 4 * 4096


def test_preallocate_rejects_non_retained_pool():
    arena = SlabArena(slab_size_bytes=4096, pin_memory=False, pool_policy="none")
    with pytest.raises(ValueError, match="retained pool policy"):
        arena.preallocate(1)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"cpu_arena_preallocate_slabs": -1}, "non-negative"),
        ({"cpu_arena_preallocate_slabs": 1, "cpu_arena_slab_size_mb": None}, "pooled arena"),
        ({"cpu_arena_preallocate_slabs": 1, "cpu_arena_pool_policy": "none"}, "pool policy"),
    ],
)
def test_preallocate_config_validation(overrides, match):
    with pytest.raises(ValueError, match=match):
        SwapPlanConfig(**overrides)


def test_train_start_preallocates_after_cpu_affinity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        train_engine_mod, "prime_cpu_affinity_binding", lambda: calls.append("affinity"))
    monkeypatch.setattr(train_engine_mod.profiler, "stop", lambda: None)
    monkeypatch.setattr(train_engine_mod.memory_profiler, "stop", lambda: None)
    monkeypatch.setattr(train_engine_mod.metrics, "close", lambda: None)

    engine = TrainEngine.__new__(TrainEngine)
    engine.args = SimpleNamespace(
        training=SimpleNamespace(
            val_interval=0,
            train_iters=0,
            manual_gc_interval=0,
            save=False,
        ),
        parallel=SimpleNamespace(fsdp_plan=SimpleNamespace(param_dtype=None)),
        data=SimpleNamespace(dataloader_param=SimpleNamespace(enable_preload=False)),
    )
    engine.train_dataloader = []
    engine.val_dataloader = None
    engine.model = SimpleNamespace(train=lambda: None)
    engine.lr_scheduler = SimpleNamespace(get_last_lr=lambda: [0.0])
    engine.iteration = 0
    engine.on_train_start = lambda: calls.append("preallocate")

    engine.train()

    assert calls == ["affinity", "preallocate"]


def test_preallocate_without_tenant_does_not_create_cache(monkeypatch):
    from mindspeed_mm.fsdp.features.memory.swap_manager import SwapManager

    manager = SwapManager(SwapPlanConfig(cpu_arena_preallocate_slabs=2))

    def fail_get_cache():
        raise AssertionError("preallocation must not create a cache without a tenant")

    monkeypatch.setattr(manager, "get_cache", fail_get_cache)
    manager.preallocate()

    assert manager._cache is None
    assert manager._tenants == []
