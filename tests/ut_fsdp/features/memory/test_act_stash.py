import copy
import inspect

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
from mindspeed_mm.fsdp.features.memory.recompute import recompute_wrapper
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
        model = TinyModel(depth=3).to(device)
        # capacity is a hard cap: 4096B < 3x2048B total, so the oldest put's
        # D2H is waited out during forward (DDR_ONLY), while the remaining
        # budget admits one in-flight prefetch in the second iteration
        cache = _make_cache(capacity_bytes=4096)
        cache.enable_prefetch = True
        apply_act_stash_modules(model, ["layers.{*}"], cache)

        x = torch.randn(8, 64, device=device, requires_grad=True)
        _run_fwd_bwd(model, x)  # iteration 1: learns the pop order
        cache.clear()           # step boundary (SwapManager.step_end)
        _run_fwd_bwd(model, x)  # iteration 2: prefetch armed

        stats = cache._iter_stats
        assert stats['prefetch_issued'] == 1
        assert stats['hbm_hit'] >= 1  # the prefetched tensor is consumed from HBM

    def test_zero_capacity_hard_cap_blocks_prefetch(self):
        _skip_if_cpu()
        device = _get_device()
        torch.manual_seed(42)
        model = TinyModel(depth=2).to(device)
        # capacity>=0 is a hard cap: 0 is zero budget, so prefetch admission
        # is refused for every handle; pops still swap in on demand (ddr_load)
        cache = _make_cache(capacity_bytes=0)
        cache.enable_prefetch = True
        apply_act_stash_modules(model, ["layers.{*}"], cache)

        x = torch.randn(8, 64, device=device, requires_grad=True)
        _run_fwd_bwd(model, x)  # iteration 1: learns the pop order
        cache.clear()
        _run_fwd_bwd(model, x)  # iteration 2: graph armed, budget zero

        stats = cache._iter_stats
        assert stats['prefetch_issued'] == 0
        assert stats['prefetch_skip_capacity'] == 1
        assert stats['ddr_load'] == 2  # both pops read back from DDR on demand


class TestFlattenInputs:
    """recompute 包装层的输入树展平（recompute_plan.flatten_inputs，默认关
    = 纯 PyTorch 调用形态）：开启后 (args, kwargs) 全树按 pytree 展平、按身份
    去重为唯一叶子，作位置参数过 checkpoint（ckpt 内部还原结构后调原函数），
    全部边界张量经 save_for_backward 进 stash pack、forward 末即释放 HBM、梯度
    逐位不变；keyword-only / **kwargs / 容器内张量 / *args 签名全覆盖（机制探针
    见 probe_pytree_flatten.py）"""

    class AttnLike(torch.nn.Module):
        def forward(self, hidden_states, aux=None):
            return hidden_states * aux if aux is not None else hidden_states * 2.0

    class AttnKwOnly(torch.nn.Module):
        def forward(self, hidden_states, *, aux):
            return hidden_states * aux

    class AttnVarKw(torch.nn.Module):
        def forward(self, hidden_states, **kw):
            return hidden_states * kw["aux"]

    class AttnGap(torch.nn.Module):
        def forward(self, hidden_states, scale=1.0, aux=None):
            return hidden_states * scale * aux

    class AttnVarArgs(torch.nn.Module):
        def forward(self, *args, **kw):
            return args[0] * kw["aux"]

    class AttnContainer(torch.nn.Module):
        def forward(self, hidden_states, extras):
            return hidden_states * extras["scale"] * (extras["gates"][0] + extras["gates"][1])

    class AttnAliasMut(torch.nn.Module):
        """b1/b2 同为调用方的同一对象 a：forward 内跨槽位原地改写累积。"""

        def forward(self, hidden_states, b1, b2):
            b1.mul_(2.0)
            b2.mul_(2.0)
            return hidden_states * b1 * b2

    def _run(self, inner, flatten, rows=8):
        """norm + ckpt(inner, kwargs 传入 hidden_states/aux)：stash hooks 包外层，
        与生产接线序一致。返回 (cache, out, x)。"""
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = inner.to(device)
        cache = _make_cache(capacity_bytes=0)
        wrapped = recompute_wrapper(inner.forward, use_reentrant=False,
                                    flatten_inputs=flatten)
        x = torch.randn(rows, 64, device=device, requires_grad=True)
        with ActStashHooks(cache):
            h = norm(x)
            out = wrapped(hidden_states=h, aux=torch.full_like(h, 0.5))
        out.sum().backward()
        return cache, out, x

    def _reference(self, inner):
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = inner.to(device)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        out = inner(norm(x), aux=torch.full_like(x, 0.5))
        out.sum().backward()
        return out.detach(), x.grad.clone()

    def test_default_off_keeps_kwargs_blind(self):
        _skip_if_cpu()
        cache, _, _ = self._run(self.AttnLike(), flatten=False)
        # 边界输入在 kwargs 里：只有 norm 的 saved tensors 进 pack，无边界 put
        assert cache._iter_stats['put_cnt'] == 3  # norm 的 input/mean/rstd
        # 不显式传 flag 时与 False 等价（默认关）
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = self.AttnLike().to(device)
        cache2 = _make_cache(capacity_bytes=0)
        wrapped = recompute_wrapper(inner.forward, use_reentrant=False)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        with ActStashHooks(cache2):
            h = norm(x)
            wrapped(hidden_states=h, aux=torch.full_like(h, 0.5))
        assert cache2._iter_stats['put_cnt'] == 3

    def test_on_packs_named_kwargs_bitwise(self):
        _skip_if_cpu()
        cache, out, x = self._run(self.AttnLike(), flatten=True)
        # 边界 hidden_states+aux 摊平进 save_for_backward：+2 且全部被消费
        assert cache._iter_stats['put_cnt'] == 5
        assert cache._iter_stats['pop_cnt'] == 5
        out_ref, g_ref = self._reference(self.AttnLike())
        assert torch.equal(out.detach(), out_ref)
        assert torch.equal(x.grad, g_ref)

    def test_on_releases_hbm_at_forward_end(self):
        _skip_if_cpu()
        # 32MB 边界 tensor：off 时被 ckpt frame 持有到 backward，on 时驱逐即释。
        # forward 收进函数作用域，让局部引用随返回释放（否则测的是测试自身）
        def _fwd_only(flatten):
            device = _get_device()
            torch.manual_seed(42)
            norm = torch.nn.LayerNorm(64).to(device)
            inner = self.AttnLike().to(device)
            cache = _make_cache(capacity_bytes=0)
            wrapped = recompute_wrapper(inner.forward, use_reentrant=False,
                                        flatten_inputs=flatten)
            x = torch.randn(131072, 64, device=device, requires_grad=True)
            with ActStashHooks(cache):
                h = norm(x)
                out = wrapped(hidden_states=h, aux=torch.full_like(h, 0.5))
            return out, x

        for flatten in (False, True):
            device = _get_device()
            mem = torch.npu if device.type == 'npu' else torch.cuda
            mem.synchronize()
            base = mem.memory_allocated()
            out, x = _fwd_only(flatten)
            mem.synchronize()
            alive = mem.memory_allocated() - base
            if flatten:
                assert alive < 96 * 2**20  # h/aux 已驱逐：仅剩 out+x(64MB)+少量
            else:
                assert alive > 120 * 2**20  # h+aux 被 frame 持有：out+x+h+aux ≈ 128MB
            del out, x

    def test_covers_signature_edge_shapes(self):
        _skip_if_cpu()
        # keyword-only / **kwargs / 前缀缺洞：全树展平不看签名，aux 照样进 pack（+2）
        for inner in (self.AttnKwOnly(), self.AttnVarKw(), self.AttnGap()):
            cache, out, x = self._run(inner, flatten=True)
            assert cache._iter_stats['put_cnt'] == 5  # norm 3 + hidden_states + aux
            assert cache._iter_stats['pop_cnt'] == 5
            _, g_ref = self._reference(inner)
            assert torch.equal(x.grad, g_ref)

    def test_covers_varargs_signature(self):
        _skip_if_cpu()
        # *args/**kw 签名：prefix 摊平整体退回的形态，展平不看签名照常覆盖
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = self.AttnVarArgs().to(device)
        cache = _make_cache(capacity_bytes=0)
        wrapped = recompute_wrapper(inner.forward, use_reentrant=False, flatten_inputs=True)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        with ActStashHooks(cache):
            h = norm(x)
            out = wrapped(h, aux=torch.full_like(h, 0.5))
        out.sum().backward()
        assert cache._iter_stats['put_cnt'] == 5  # norm 3 + h + aux
        assert cache._iter_stats['pop_cnt'] == 5
        # 参考：无 hooks 无 ckpt
        torch.manual_seed(42)
        norm2 = torch.nn.LayerNorm(64).to(device)
        inner2 = self.AttnVarArgs().to(device)
        x2 = torch.randn(8, 64, device=device, requires_grad=True)
        inner2(norm2(x2), aux=torch.full_like(x2, 0.5)).sum().backward()
        assert torch.equal(x.grad, x2.grad)

    def test_covers_nested_container_bitwise(self):
        _skip_if_cpu()
        # 嵌套容器（dict 套 list）内的张量随整树展平进 pack（+4）
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = self.AttnContainer().to(device)
        cache = _make_cache(capacity_bytes=0)
        wrapped = recompute_wrapper(inner.forward, use_reentrant=False, flatten_inputs=True)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        with ActStashHooks(cache):
            h = norm(x)
            extras = {"scale": torch.full_like(h, 0.5),
                      "gates": [torch.full_like(h, 0.25), torch.full_like(h, 0.25)]}
            out = wrapped(hidden_states=h, extras=extras)
        out.sum().backward()
        assert cache._iter_stats['put_cnt'] == 7  # norm 3 + h + scale + 2 gates
        assert cache._iter_stats['pop_cnt'] == 7
        torch.manual_seed(42)
        norm2 = torch.nn.LayerNorm(64).to(device)
        inner2 = self.AttnContainer().to(device)
        x2 = torch.randn(8, 64, device=device, requires_grad=True)
        h2 = norm2(x2)
        extras2 = {"scale": torch.full_like(h2, 0.5),
                   "gates": [torch.full_like(h2, 0.25), torch.full_like(h2, 0.25)]}
        inner2(h2, extras2).sum().backward()
        assert torch.equal(x.grad, x2.grad)

    def test_alias_dedup_bitwise(self):
        _skip_if_cpu()
        # b1/b2 同为 a：按身份去重后单份 pack（+2 = h + a），recompute 恢复共享
        # 对象保住别名语义，跨槽位原地改写累积复现 forward，梯度逐位 == 无 ckpt 参考
        device = _get_device()
        torch.manual_seed(42)
        norm = torch.nn.LayerNorm(64).to(device)
        inner = self.AttnAliasMut().to(device)
        cache = _make_cache(capacity_bytes=0)
        wrapped = recompute_wrapper(inner.forward, use_reentrant=False, flatten_inputs=True)
        x = torch.randn(8, 64, device=device, requires_grad=True)
        with ActStashHooks(cache):
            h = norm(x)
            a = torch.ones(64, device=device)
            out = wrapped(hidden_states=h, b1=a, b2=a)
        out.sum().backward()
        assert cache._iter_stats['put_cnt'] == 5  # norm 3 + h + a（a 单份）
        assert cache._iter_stats['pop_cnt'] == 5
        torch.manual_seed(42)
        norm2 = torch.nn.LayerNorm(64).to(device)
        inner2 = self.AttnAliasMut().to(device)
        x2 = torch.randn(8, 64, device=device, requires_grad=True)
        a2 = torch.ones(64, device=device)
        inner2(norm2(x2), a2, a2).sum().backward()
        assert torch.equal(x.grad, x2.grad)

    def test_generic_wrapper_flattens_without_signature(self):
        _skip_if_cpu()
        # 生产接线序（recompute.py:22-30）：op replay 先把 attn forward 换成
        # OpReplayPatcher（通用 __call__(*args, **kwargs)），recompute_wrapper 再包
        # checkpoint。展平不调被包函数的签名，通用包装不再构成盲区。

        class GenericWrapper:
            """模拟 OpReplayPatcher 的签名形态（不设 __signature__）。"""

            def __init__(self, fn):
                self._fn = fn

            def __call__(self, *args, **kwargs):
                return self._fn(*args, **kwargs)

        inner = self.AttnLike().to(_get_device())
        inner.forward = GenericWrapper(inner.forward)
        cache, out, x = self._run(inner, flatten=True)
        assert cache._iter_stats['put_cnt'] == 5
        _, g_ref = self._reference(self.AttnLike())
        assert torch.equal(x.grad, g_ref)

    def test_op_replay_patcher_exposes_wrapped_signature(self):
        # OpReplayPatcher.__call__ 是通用签名；透出被包 forward 的签名，
        # 供下游 inspect.signature 消费者（recompute 包装层的 past_key_values
        # 注入判断）看到真实签名
        from mindspeed_mm.fsdp.features.memory.op_replay import OpReplayPatcher
        patcher = OpReplayPatcher(self.AttnLike(), controller=None, scope=None)
        assert list(inspect.signature(patcher).parameters) == ['hidden_states', 'aux']

    def test_config_field_default_false(self):
        from mindspeed_mm.fsdp.params.feature_args import RecomputePlanConfig
        assert RecomputePlanConfig().flatten_inputs is False
        assert RecomputePlanConfig(flatten_inputs=True).flatten_inputs is True


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
