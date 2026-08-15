import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from mindspeed_mm.fsdp.features.memory import swap_core as swap_core_mod
from mindspeed_mm.fsdp.features.memory.act_stash import (
    ActStashHooks,
    _stashable,
    apply_act_stash_modules,
    get_act_stash_modules,
)
from mindspeed_mm.fsdp.features.memory.slab_arena import SlabArena
from mindspeed_mm.fsdp.features.memory.swap_core import SwapCache, SwapHandle
from mindspeed_mm.fsdp.utils.device import get_device_type


pytestmark = pytest.mark.skipif(
    not swap_core_mod.IS_DEVICE_AVAILABLE,
    reason="swap requires a usable device (fast fail on CPU-only hosts)")


def _get_device():
    return torch.device(get_device_type())


def _skip_if_cpu():
    if _get_device().type == 'cpu':
        pytest.skip("act stash requires cuda/npu tensors")


def _make_cache(*args, **kwargs):
    """SwapCache with a default on-demand arena; pass cpu_arena to override."""
    kwargs.setdefault("cpu_arena", SlabArena.on_demand())
    return SwapCache(*args, **kwargs)


def _make_evicting_cache(**kwargs):
    """capacity=1B (soft limit defaults to 0): every put is evicted to DDR
    immediately — the pop path must read back from DDR."""
    kwargs.setdefault("capacity_bytes", 1)
    return _make_cache(**kwargs)


class TinyModel(torch.nn.Module):
    def __init__(self, depth=2, width=64):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(width, width) for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _run_fwd_bwd(model, x):
    model.zero_grad(set_to_none=True)
    if x.requires_grad:
        x.grad = None
    model(x).sum().backward()


def _grads_of(model, x):
    grads = [p.grad.clone() for p in model.parameters()]
    if x.requires_grad:
        grads.append(x.grad.clone())
    return grads


class TestStashable:
    """Eligibility predicate: device tensor worth moving, nothing else"""

    def test_device_tensor_is_stashable(self):
        _skip_if_cpu()
        assert _stashable(torch.randn(4, 4, device=_get_device())) is True

    def test_cpu_tensor_passes_through(self):
        assert _stashable(torch.randn(4, 4)) is False

    def test_parameter_and_its_view_are_skipped(self):
        _skip_if_cpu()
        param = torch.nn.Parameter(torch.randn(4, 4, device=_get_device()))
        assert _stashable(param) is False
        assert _stashable(param.t()) is False  # view: _base is the Parameter

    def test_empty_storage_is_skipped(self):
        _skip_if_cpu()
        assert _stashable(torch.empty(0, device=_get_device())) is False

    def test_sparse_tensor_passes_through(self):
        _skip_if_cpu()
        # non-strided layouts have no flat storage; storage() would raise
        sparse = torch.sparse_coo_tensor([[0]], [1.0], (4,), device=_get_device())
        assert _stashable(sparse) is False


class TestPackUnpack:
    """Token discipline of the hooks: handle token vs pass-through token"""

    def test_device_tensor_roundtrip_bitwise(self):
        _skip_if_cpu()
        cache = _make_evicting_cache()  # forced real D2H + H2D
        hooks = ActStashHooks(cache)
        x = torch.randn(64, 64, dtype=torch.bfloat16, device=_get_device())
        x_copy = x.clone()

        token = hooks._pack(x)
        assert isinstance(token, SwapHandle)
        del x  # drop the last strong ref so eviction frees HBM for real
        out = hooks._unpack(token)
        assert torch.equal(out, x_copy)  # D2H/H2D copies are byte-exact

    def test_ineligible_tensor_is_its_own_token(self):
        cache = _make_evicting_cache()
        hooks = ActStashHooks(cache)
        cpu_t = torch.randn(4, 4)
        assert hooks._pack(cpu_t) is cpu_t
        assert hooks._unpack(cpu_t) is cpu_t
        assert cache._iter_stats['put_cnt'] == 0

    def test_unpack_missing_token_raises(self):
        _skip_if_cpu()
        cache = _make_evicting_cache()
        hooks = ActStashHooks(cache)
        token = hooks._pack(torch.randn(4, 4, device=_get_device()))
        cache.pop(token)  # consume it; the handle leaves the storage
        with pytest.raises(RuntimeError, match="missing from the swap cache"):
            hooks._unpack(token)


class TestForwardBackward:
    """Install point: put on (grad-enabled) forward, pop on backward;
    gradients bitwise-identical to the no-hooks reference"""

    def _make_pair(self, depth=2):
        device = _get_device()
        torch.manual_seed(42)
        model = TinyModel(depth=depth).to(device)
        reference = copy.deepcopy(model)
        return model, reference

    def test_grads_bitwise_with_forced_eviction(self):
        _skip_if_cpu()
        model, reference = self._make_pair()
        cache = _make_evicting_cache()
        apply_act_stash_modules(model, ["layers.{*}"], cache)

        x = torch.randn(8, 64, device=_get_device(), requires_grad=True)
        _run_fwd_bwd(model, x)
        _run_fwd_bwd(reference, x)

        for g, g_ref in zip(_grads_of(model, x), _grads_of(reference, x)):
            assert torch.equal(g, g_ref)

    def test_put_on_forward_pop_on_backward(self):
        _skip_if_cpu()
        model, _ = self._make_pair()
        cache = _make_evicting_cache()
        apply_act_stash_modules(model, ["layers.{*}"], cache)

        # eval / no-grad forward: no autograd graph, the hooks never fire
        with torch.no_grad():
            model(torch.randn(8, 64, device=_get_device()))
        assert cache._iter_stats['put_cnt'] == 0

        x = torch.randn(8, 64, device=_get_device(), requires_grad=True)
        model(x).sum().backward()
        # each Linear saves exactly one stashable tensor (its input; the
        # weight/bias Parameters are filtered) and unpacks it once
        assert cache._iter_stats['put_cnt'] == 2
        assert cache._iter_stats['pop_cnt'] == 2

    def test_no_match_raises(self):
        _skip_if_cpu()
        model, _ = self._make_pair()
        with pytest.raises(RuntimeError, match="No module named"):
            get_act_stash_modules(model, ["nonexistent.{*}"])


class TestCheckpointJurisdiction:
    """Stash hooks outside a non-reentrant checkpoint (the production wiring
    order): coverage = checkpoint boundary inputs only; internal saved tensors
    belong to the checkpoint frame hooks; kwargs tensors are a blind spot"""

    def _run(self, x, extra_kwargs=None, with_hooks=True):
        cache = _make_evicting_cache()
        torch.manual_seed(42)
        layer = torch.nn.Linear(64, 64).to(x.device)

        def fn(inp, pos=None):
            out = layer(inp)
            return out if pos is None else out + pos

        kwargs = extra_kwargs or {}
        if with_hooks:
            with ActStashHooks(cache):
                y = checkpoint(fn, x, use_reentrant=False, **kwargs)
        else:
            y = checkpoint(fn, x, use_reentrant=False, **kwargs)
        y.sum().backward()
        return layer, cache

    def test_coverage_is_boundary_inputs_only(self):
        _skip_if_cpu()
        device = _get_device()
        x = torch.randn(8, 64, device=device, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()

        layer, cache = self._run(x)
        # exactly one put: the boundary input x (checkpoint's initial forward
        # runs under no_grad -> nothing packed; internal saved tensors are
        # taken over by the checkpoint frame hooks at recompute)
        assert cache._iter_stats['put_cnt'] == 1
        assert cache._iter_stats['pop_cnt'] == 1

        layer_ref, _ = self._run(x_ref, with_hooks=False)
        assert torch.equal(x.grad, x_ref.grad)
        assert torch.equal(layer.weight.grad, layer_ref.weight.grad)

    def test_kwargs_tensors_never_packed(self):
        _skip_if_cpu()
        device = _get_device()
        x = torch.randn(8, 64, device=device, requires_grad=True)
        pos = torch.randn(64, device=device)
        _, cache = self._run(x, extra_kwargs={"pos": pos})
        # documented blind spot: tensors inside kwargs stay outside the hooks
        assert cache._iter_stats['put_cnt'] == 1


class TestPrefetchArmsOnReversePops:
    """Tenant self-driven prefetch: the backward pop order (reverse of the
    forward put order) is learned in the first iteration; the second
    iteration's pops trigger chain prefetch, and prefetched tensors hit"""

    def test_second_iteration_prefetch(self):
        _skip_if_cpu()
        device = _get_device()
        torch.manual_seed(42)
        model = TinyModel(depth=2).to(device)
        # capacity=0: deterministic eviction (put returns with the tensor
        # already in DDR); pops read back from DDR, and the second
        # iteration's chain prefetch issues real H2D loads
        cache = _make_cache(capacity_bytes=0)
        cache.enable_prefetch = True
        apply_act_stash_modules(model, ["layers.{*}"], cache)

        x = torch.randn(8, 64, device=device, requires_grad=True)
        _run_fwd_bwd(model, x)  # iteration 1: learns the pop order
        cache.clear()           # step boundary (SwapManager.step_end)
        _run_fwd_bwd(model, x)  # iteration 2: prefetch armed

        stats = cache._iter_stats
        assert stats['prefetch_issued'] == 1
        assert stats['hbm_hit'] >= 1  # the prefetched tensor is consumed from HBM


class TestWiring:
    """FeaturesApplier: impl='legacy' (default, untouched) vs impl='stash'
    (shared cache tenant); invalid impl rejected at config validation"""

    def _make_config(self, **plan_kwargs):
        from mindspeed_mm.fsdp.params.feature_args import (
            ActivationOffloadPlanConfig, FeatureArguments)
        return FeatureArguments(
            enable_activation_offload=True,
            activation_offload_plan=ActivationOffloadPlanConfig(
                apply_modules=["layers.{*}"], **plan_kwargs))

    def test_stash_impl_registers_tenant_on_shared_cache(self):
        _skip_if_cpu()
        from mindspeed_mm.fsdp.features.apply_features import FeaturesApplier
        applier = FeaturesApplier(self._make_config(impl="stash"))
        model = TinyModel(depth=2).to(_get_device())

        applier.apply_activation_offload_modules(model)

        assert applier.swap_manager is not None
        assert applier.swap_manager._tenants == ["actstash"]
        x = torch.randn(8, 64, device=_get_device(), requires_grad=True)
        model(x).sum().backward()
        cache = applier.swap_manager.get_cache("actstash")
        assert cache._iter_stats['put_cnt'] == 2
        assert cache._iter_stats['pop_cnt'] == 2

    def test_legacy_impl_untouched(self):
        _skip_if_cpu()
        from mindspeed_mm.fsdp.features.apply_features import FeaturesApplier
        applier = FeaturesApplier(self._make_config())  # impl defaults to legacy
        model = TinyModel(depth=2).to(_get_device())

        applier.apply_activation_offload_modules(model)

        assert applier.swap_manager is None  # legacy never touches the swap manager
        assert not hasattr(model.layers[0].forward, '__self__')  # wrapped, not a bound method

    def test_invalid_impl_rejected(self):
        from pydantic import ValidationError
        from mindspeed_mm.fsdp.params.feature_args import ActivationOffloadPlanConfig
        with pytest.raises(ValidationError):
            ActivationOffloadPlanConfig(impl="bogus")
