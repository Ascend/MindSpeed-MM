"""Recompute wiring tests (unified apply_modules).

CPU-only suite: checkpoint and the wiring are device-agnostic, so unlike
test_act_stash.py nothing here needs a usable accelerator.

apply_modules entries are dispatched by format: an entry that matches a module
in the model tree wraps that module's .forward; anything else is treated as
'module.pattern.method_name' (split at the last '.') and wraps the named method
on each matched module instance. Method entries therefore always need a module
prefix — the root module's name is '' and cannot be addressed.
"""
import torch
import pytest

from mindspeed_mm.fsdp.features.memory.recompute import recompute_modules
from mindspeed_mm.fsdp.params.feature_args import RecomputePlanConfig


def _input():
    # requires_grad so the non-reentrant checkpoint actually replays on backward.
    return torch.randn(2, 4, requires_grad=True)


class TestRecomputeModulesWiring:
    def test_empty_plan_is_noop(self):
        model = torch.nn.Linear(4, 4)
        recompute_modules(model, RecomputePlanConfig())  # empty apply_modules is a no-op

    def test_module_forward_wrapped_and_replayed(self):
        calls = []

        class Blk(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                calls.append(1)
                return self.linear(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blk = Blk()

            def forward(self, x):
                return self.blk(x)

        model = M()
        recompute_modules(model, RecomputePlanConfig(apply_modules=['blk']))
        model(_input()).sum().backward()
        assert len(calls) == 2  # initial forward + backward replay

    def test_method_wrapped_and_replayed(self):
        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.calls = 0

            def _custom_method(self, x):
                self.calls += 1
                return self.linear(x)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Inner()

            def forward(self, x):
                return self.sub._custom_method(x)

        model = M()
        recompute_modules(model, RecomputePlanConfig(apply_modules=['sub._custom_method']))
        model(_input()).sum().backward()
        assert model.sub.calls == 2

    def test_missing_method_on_matched_module_raises(self):
        # Well-formed 'module.method' entry whose method does not exist.
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = torch.nn.Linear(4, 4)

        with pytest.raises(RuntimeError):
            recompute_modules(M(), RecomputePlanConfig(apply_modules=['sub.no_such_method']))

    def test_malformed_method_entry_raises(self):
        # No module prefix: rpartition yields an empty module pattern.
        with pytest.raises(ValueError):
            recompute_modules(torch.nn.Linear(4, 4), RecomputePlanConfig(apply_modules=['no_dot']))

    def test_past_key_values_positional_not_duplicated(self):
        # Regression: forcing the kv cache to None must overwrite the bound slot,
        # not inject a kwarg that collides with a positional past_key_values
        # ("got multiple values for argument 'past_key_values'").
        seen = []

        class Attn(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def _attention_forward(self, hidden_states, position_embeddings,
                                   attention_mask=None, position_ids=None,
                                   past_key_values=None, cache_position=None):
                seen.append(past_key_values)
                return self.linear(hidden_states)

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attn()

            def forward(self, x):
                # past_key_values passed positionally, as the model does.
                return self.attn._attention_forward(x, None, None, None, 'cache', None)

        model = M()
        recompute_modules(model, RecomputePlanConfig(apply_modules=['attn._attention_forward']))
        model(_input()).sum().backward()
        assert len(seen) == 2                      # initial forward + backward replay
        assert all(pkv is None for pkv in seen)    # slot forced to None, no duplicate-arg error
