# Copyright (c) Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Muon learning-rate scaling, skipped updates, and checkpoint continuation."""
import copy

import pytest
import torch

from mindspeed_mm.optimizer.muon import (
    Muon,
    adjust_lr_wd_for_muon,
    zeropower_via_newtonschulz5,
)


class TestZeropowerViaNewtonSchulz5:
    """Properties of the NS5 orthogonalization used by Muon."""

    def test_output_shape_matches_input(self):
        torch.manual_seed(0)
        G = torch.randn(8, 4, dtype=torch.bfloat16)
        out = zeropower_via_newtonschulz5(G, steps=5)
        assert out.shape == G.shape

    def test_wide_input_tall_output(self):
        """Tall matrices (rows > cols) are transposed, orthogonalized, and transposed back."""
        torch.manual_seed(1)
        G = torch.randn(16, 4, dtype=torch.bfloat16)
        out = zeropower_via_newtonschulz5(G, steps=5)
        assert out.shape == G.shape

    def test_singular_values_bounded_around_one(self):
        """After NS5, singular values cluster around 1 within [0.5, 1.5]."""
        torch.manual_seed(2)
        G = torch.randn(16, 16, dtype=torch.bfloat16).float()
        out = zeropower_via_newtonschulz5(G.to(torch.bfloat16), steps=5).float()
        s = torch.linalg.svdvals(out)
        # The interval [0.5, 1.5] is the documented design range.
        assert s.min() > 0.4, f"min singular value {s.min()} below 0.4"
        assert s.max() < 1.6, f"max singular value {s.max()} above 1.6"

    def test_non_2d_input_raises(self):
        """1D and 3D inputs are programmer errors and must raise ValueError."""
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(8), steps=5)
        with pytest.raises(ValueError, match="expects a 2-D tensor"):
            zeropower_via_newtonschulz5(torch.randn(2, 3, 4), steps=5)

    def test_more_steps_closer_to_orthogonal(self):
        """More NS5 iterations produce tighter clustering around 1."""
        torch.manual_seed(3)
        G = torch.randn(32, 32, dtype=torch.bfloat16)
        s_few = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=3).float())
        s_many = torch.linalg.svdvals(zeropower_via_newtonschulz5(G, steps=8).float())
        # The spread of singular values shrinks with more iterations.
        spread_few = s_few.max() - s_few.min()
        spread_many = s_many.max() - s_many.min()
        assert spread_many < spread_few


class TestAdjustLrWdForMuon:

    def test_lr_scaled_by_sqrt_max_dim(self):
        lr = 0.02
        matched_rms = 0.2
        adjusted = adjust_lr_wd_for_muon(lr, matched_rms, (16, 8))
        assert adjusted == pytest.approx(0.02 * (16 ** 0.5) * 0.2)

    def test_lr_uses_max_of_two_dims(self):
        adjusted_1 = adjust_lr_wd_for_muon(0.02, 0.2, (16, 4))
        adjusted_2 = adjust_lr_wd_for_muon(0.02, 0.2, (4, 16))
        assert adjusted_1 == adjusted_2

    def test_non_matrix_shape_rejected(self):
        with pytest.raises(ValueError, match="shape of one matrix"):
            adjust_lr_wd_for_muon(0.02, 0.2, (16, 8, 64, 64))

    def test_zero_lr_stays_zero(self):
        assert adjust_lr_wd_for_muon(0.0, 0.5, (16, 16)) == 0.0


class TestMuonOptimizer:
    @pytest.mark.parametrize("use_muon", [False, True])
    def test_missing_gradient_preserves_parameter_and_state(self, use_muon):
        parameter = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = Muon([dict(params=[parameter], use_muon=use_muon)], weight_decay=.1)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.ones_like(parameter), rtol=0, atol=0)
        assert not optimizer.state

    def test_zero_gradient_applies_only_weight_decay(self):
        parameter = torch.nn.Parameter(torch.ones(8, 8))
        parameter.grad = torch.zeros_like(parameter)
        optimizer = Muon([dict(params=[parameter], use_muon=True)], lr=.01, weight_decay=.1)
        optimizer.step()
        torch.testing.assert_close(parameter, torch.full_like(parameter, .999))

    def test_state_dict_round_trip(self):
        param_muon = torch.nn.Parameter(torch.ones(8, 8))
        param_adamw = torch.nn.Parameter(torch.ones(8, 8))
        opt = Muon(
            [
                {"params": [param_muon], "use_muon": True},
                {"params": [param_adamw], "use_muon": False},
            ],
            lr=0.01,
            ns_steps=3,
        )
        param_muon.grad = torch.ones_like(param_muon)
        param_adamw.grad = torch.ones_like(param_adamw)
        opt.step()
        sd = opt.state_dict()
        # Restore both parameter groups and their state into a fresh optimizer.
        sd_copy = copy.deepcopy(sd)
        opt2 = Muon(
            [
                {"params": [torch.nn.Parameter(torch.zeros(8, 8))], "use_muon": True},
                {"params": [torch.nn.Parameter(torch.zeros(8, 8))], "use_muon": False},
            ],
            lr=0.01,
            ns_steps=3,
        )
        opt2.load_state_dict(sd_copy)
        for source_group, target_group in zip(opt.param_groups, opt2.param_groups):
            source, target = source_group["params"][0], target_group["params"][0]
            assert source_group["use_muon"] == target_group["use_muon"]
            assert opt.state[source].keys() == opt2.state[target].keys()
            for key, value in opt.state[source].items():
                torch.testing.assert_close(opt2.state[target][key], value)
            with torch.no_grad():
                target.copy_(source)
            target.grad = source.grad.clone()
        opt.step()
        opt2.step()
        for source_group, target_group in zip(opt.param_groups, opt2.param_groups):
            torch.testing.assert_close(source_group["params"][0], target_group["params"][0])


class TestLegacyMomentumBufferMigration:
    """Legacy ``muon_buffer`` (running sum) migration to ``momentum_buffer`` (EMA)."""

    @staticmethod
    def _legacy_checkpoint(buffer, momentum):
        """A state dict in the legacy format: one Muon group, one parameter."""
        return {
            "state": {0: {"muon_buffer": buffer}},
            "param_groups": [
                {
                    "params": [0],
                    "use_muon": True,
                    "lr": 0.01,
                    "weight_decay": 0.1,
                    "matched_adamw_rms": 0.2,
                    "momentum": momentum,
                    "nesterov": True,
                    "ns_steps": 3,
                    "adamw_betas": (0.95, 0.95),
                    "adamw_eps": 1e-8,
                }
            ],
        }

    def _fresh_optimizer(self, momentum):
        parameter = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = Muon(
            [{"params": [parameter], "use_muon": True}],
            lr=0.01,
            momentum=momentum,
            ns_steps=3,
        )
        return parameter, optimizer

    def test_legacy_buffer_is_rescaled_to_the_moving_average(self):
        momentum = 0.9
        legacy = torch.full((8, 8), 10.0)
        parameter, optimizer = self._fresh_optimizer(momentum)
        optimizer.load_state_dict(self._legacy_checkpoint(legacy.clone(), momentum))
        state = optimizer.state[parameter]
        assert "muon_buffer" not in state
        torch.testing.assert_close(state["momentum_buffer"], legacy * (1 - momentum))

    def test_migration_does_not_mutate_the_incoming_state_dict(self):
        """DCP and the broadcast loader reuse the dict they hand to the optimizer."""
        momentum = 0.9
        checkpoint = self._legacy_checkpoint(torch.full((8, 8), 10.0), momentum)
        _, optimizer = self._fresh_optimizer(momentum)
        optimizer.load_state_dict(checkpoint)
        assert "muon_buffer" in checkpoint["state"][0]
        assert "momentum_buffer" not in checkpoint["state"][0]

    def test_current_buffer_wins_when_both_keys_are_present(self):
        momentum = 0.9
        current = torch.full((8, 8), 3.0)
        checkpoint = self._legacy_checkpoint(torch.full((8, 8), 10.0), momentum)
        checkpoint["state"][0]["momentum_buffer"] = current.clone()
        parameter, optimizer = self._fresh_optimizer(momentum)
        optimizer.load_state_dict(checkpoint)
        state = optimizer.state[parameter]
        assert "muon_buffer" not in state
        torch.testing.assert_close(state["momentum_buffer"], current)

    def test_checkpoint_momentum_is_preferred_over_the_live_value(self):
        """The saved sum was produced by the checkpoint's momentum, not the new one."""
        legacy = torch.full((8, 8), 10.0)
        parameter, optimizer = self._fresh_optimizer(0.5)
        optimizer.load_state_dict(self._legacy_checkpoint(legacy.clone(), 0.9))
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], legacy * (1 - 0.9)
        )

    def test_resume_continues_the_legacy_recurrence(self):
        """A migrated resume takes the step the legacy recurrence would have taken."""
        momentum, gradient = 0.9, torch.full((8, 8), 2.0)
        legacy_buffer = torch.full((8, 8), 10.0)

        # What the legacy implementation would have done on the next step:
        # the running sum advances, then Nesterov forms grad + momentum * buf.
        expected_sum = legacy_buffer * momentum + gradient
        expected_ns_input = gradient + momentum * expected_sum

        parameter, optimizer = self._fresh_optimizer(momentum)
        optimizer.load_state_dict(self._legacy_checkpoint(legacy_buffer.clone(), momentum))
        parameter.grad = gradient.clone()
        ns_input = optimizer._muon_ns_input(parameter, optimizer.param_groups[0])

        # Both buffer and Newton-Schulz input are the legacy values scaled by
        # 1 - momentum. Newton-Schulz normalizes its input, so the resulting
        # update direction is the same.
        torch.testing.assert_close(
            optimizer.state[parameter]["momentum_buffer"], expected_sum * (1 - momentum)
        )
        torch.testing.assert_close(ns_input, expected_ns_input * (1 - momentum))

    def test_current_checkpoints_are_untouched(self):
        momentum = 0.9
        parameter = torch.nn.Parameter(torch.ones(8, 8))
        optimizer = Muon(
            [{"params": [parameter], "use_muon": True}],
            lr=0.01,
            momentum=momentum,
            ns_steps=3,
        )
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())

        target, fresh = self._fresh_optimizer(momentum)
        fresh.load_state_dict(copy.deepcopy(saved))
        torch.testing.assert_close(
            fresh.state[target]["momentum_buffer"],
            saved["state"][0]["momentum_buffer"],
        )
