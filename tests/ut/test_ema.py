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

"""Unit tests for ``mindspeed_mm.utils.ema.EMA``.

Covers the full lifecycle of the exponential-moving-average helper:

  * ``register`` clones only trainable parameters into the shadow dict.
  * ``update`` blends the current parameter values with the shadow
    without modifying the model parameters themselves.
  * ``apply_shadow`` swaps shadow values into the model and keeps a
    backup so the original values can be restored.
  * ``restore`` reverts the model parameters from the backup and
    clears the backup dict.

All tests are CPU-runnable and do not require megatron or NPU.
"""

import importlib.util as _importlib_util
import os as _os

import pytest
import torch
import torch.nn as nn

# Load the module directly to avoid triggering mindspeed_mm/__init__.py
# (which pulls in megatron and torch_npu dependencies).
_EMA_PATH = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    "../../mindspeed_mm/utils/ema.py",
)
_EMA_SPEC = _importlib_util.spec_from_file_location("mindspeed_mm_utils_ema", _EMA_PATH)
_ema_mod = _importlib_util.module_from_spec(_EMA_SPEC)
_EMA_SPEC.loader.exec_module(_ema_mod)
EMA = _ema_mod.EMA

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module):
    """A tiny model with a mix of trainable and frozen parameters."""

    def __init__(self):
        super().__init__()
        self.trainable_weight = nn.Parameter(torch.randn(4, 4))
        self.frozen_weight = nn.Parameter(torch.randn(3, 3), requires_grad=False)
        self.trainable_bias = nn.Parameter(torch.randn(4))


@pytest.fixture
def simple_model():
    return _SimpleModel()


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------


class TestRegister:
    """``EMA.register`` must snapshot all and only trainable parameters."""

    def test_registers_trainable_params(self, simple_model):
        ema = EMA(simple_model, decay=0.999)
        ema.register()
        assert "trainable_weight" in ema.shadow
        assert "trainable_bias" in ema.shadow

    def test_ignores_frozen_params(self, simple_model):
        ema = EMA(simple_model, decay=0.999)
        ema.register()
        assert "frozen_weight" not in ema.shadow

    def test_shadow_contains_clones(self, simple_model):
        ema = EMA(simple_model, decay=0.999)
        ema.register()
        original = simple_model.trainable_weight.data.clone()
        # Mutate the model parameter in-place.
        simple_model.trainable_weight.data += 1.0
        # The shadow must still hold the old value.
        assert torch.allclose(ema.shadow["trainable_weight"], original)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    """``EMA.update`` must blend shadow and current param values."""

    def test_update_formula(self, simple_model):
        decay = 0.9
        ema = EMA(simple_model, decay=decay)
        ema.register()

        original_param = simple_model.trainable_weight.data.clone()
        original_shadow = ema.shadow["trainable_weight"].clone()

        # Perturb the model parameter.
        simple_model.trainable_weight.data = torch.ones_like(original_param) * 5.0

        ema.update()

        expected = (1.0 - decay) * simple_model.trainable_weight.data + decay * original_shadow
        assert torch.allclose(ema.shadow["trainable_weight"], expected)

    def test_update_does_not_change_model_params(self, simple_model):
        ema = EMA(simple_model, decay=0.9)
        ema.register()
        before = simple_model.trainable_weight.data.clone()
        ema.update()
        assert torch.allclose(simple_model.trainable_weight.data, before)

    def test_update_ignores_unregistered_params(self, simple_model):
        ema = EMA(simple_model, decay=0.9)
        ema.register()
        # frozen_weight was never registered, so update should be a no-op for it.
        ema.update()
        assert "frozen_weight" not in ema.shadow


# ---------------------------------------------------------------------------
# Apply shadow / Restore
# ---------------------------------------------------------------------------


class TestApplyShadowAndRestore:
    """``apply_shadow`` and ``restore`` must swap parameter values safely."""

    def test_apply_shadow_overwrites_params(self, simple_model):
        ema = EMA(simple_model, decay=0.9)
        ema.register()
        # Change shadow so we can detect the swap.
        ema.shadow["trainable_weight"] = torch.ones_like(ema.shadow["trainable_weight"]) * 42.0

        original = simple_model.trainable_weight.data.clone()
        ema.apply_shadow()

        assert torch.allclose(simple_model.trainable_weight.data, ema.shadow["trainable_weight"])
        assert "trainable_weight" in ema.backup
        assert torch.allclose(ema.backup["trainable_weight"], original)

    def test_restore_reverts_params(self, simple_model):
        ema = EMA(simple_model, decay=0.9)
        ema.register()
        original = simple_model.trainable_weight.data.clone()
        ema.apply_shadow()
        ema.restore()
        assert torch.allclose(simple_model.trainable_weight.data, original)

    def test_restore_clears_backup(self, simple_model):
        ema = EMA(simple_model, decay=0.9)
        ema.register()
        ema.apply_shadow()
        assert ema.backup
        ema.restore()
        assert ema.backup == {}


# ---------------------------------------------------------------------------
# End-to-end lifecycle
# ---------------------------------------------------------------------------


class TestLifecycle:
    """A full train-eval lifecycle using EMA."""

    def test_full_lifecycle(self, simple_model):
        decay = 0.5
        ema = EMA(simple_model, decay=decay)

        # 1. Register initial parameters.
        ema.register()
        init_shadow = ema.shadow["trainable_weight"].clone()

        # 2. Simulate a training step that changes parameters.
        with torch.no_grad():
            simple_model.trainable_weight.data += 1.0

        # 3. Update shadow with the new parameter values.
        ema.update()
        updated_shadow = ema.shadow["trainable_weight"].clone()
        # Shadow must have moved toward the new parameter value.
        assert not torch.allclose(updated_shadow, init_shadow)

        # 4. Apply shadow for evaluation.
        original_param = simple_model.trainable_weight.data.clone()
        ema.apply_shadow()
        assert torch.allclose(simple_model.trainable_weight.data, updated_shadow)

        # 5. Restore for continued training.
        ema.restore()
        assert torch.allclose(simple_model.trainable_weight.data, original_param)
