# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
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

"""Unit tests for the `all-linear` target_modules expansion with freeze exclusion.

These tests live under ``tests/ut_fsdp`` because ``lora_utils`` imports
``mindspeed.fsdp.utils.str_match`` (the same matcher used by freeze/apply_modules),
so the full FSDP2 stack is required.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from mindspeed_mm.fsdp.utils.lora_utils import find_all_linear_target_modules  # noqa: E402


class MiniVL(nn.Module):
    """Mini multimodal model (visual tower + aligner + language model).

    Mirrors the module layout of real MM multimodal models so that freeze-based
    component exclusion can be exercised. Linear layout:
      model.visual.blocks.{i}.self_attn.q_proj   (2)
      model.visual.blocks.{i}.self_attn.proj     (2)
      model.visual.merger.linear_fc1, linear_fc2 (aligner)  (2)
      model.language_model.layers.{i}.self_attn.q_proj (2)
      model.language_model.layers.{i}.mlp.gate_proj    (2)
      lm_head                                          (1, always excluded)
    """

    def __init__(self) -> None:
        super().__init__()
        visual = nn.Module()
        blocks = []
        for _ in range(2):
            blk = nn.Module()
            blk.self_attn = nn.Module()
            blk.self_attn.q_proj = nn.Linear(8, 8)
            blk.self_attn.proj = nn.Linear(8, 8)
            blocks.append(blk)
        visual.blocks = nn.ModuleList(blocks)
        visual.merger = nn.Module()
        visual.merger.linear_fc1 = nn.Linear(8, 8)
        visual.merger.linear_fc2 = nn.Linear(8, 8)

        language_model = nn.Module()
        layers = []
        for _ in range(2):
            dec = nn.Module()
            dec.self_attn = nn.Module()
            dec.self_attn.q_proj = nn.Linear(8, 8)
            dec.mlp = nn.Module()
            dec.mlp.gate_proj = nn.Linear(8, 8)
            layers.append(dec)
        language_model.layers = nn.ModuleList(layers)

        self.model = nn.Module()
        self.model.visual = visual
        self.model.language_model = language_model
        self.lm_head = nn.Linear(8, 100)


class TestFindAllLinearTargetModules:
    """Test all-linear keyword expansion with freeze-based exclusion."""

    def test_all_linear_no_freeze(self) -> None:
        """Without freeze, all nn.Linear leaves (except lm_head) are matched."""
        model = MiniVL()
        matched = find_all_linear_target_modules(model)

        # 4 (visual blocks) + 2 (merger) + 4 (language model) = 10; lm_head excluded
        assert len(matched) == 10
        assert not any("lm_head" in m for m in matched)
        assert all(m.startswith("model.") for m in matched)

    def test_all_linear_freeze_visual(self) -> None:
        """freeze: [model.visual] excludes the whole vision tower + aligner."""
        model = MiniVL()
        matched = find_all_linear_target_modules(model, freeze_patterns=["model.visual"])

        # Only language_model linears survive (4)
        assert len(matched) == 4
        assert all(m.startswith("model.language_model.") for m in matched)
        assert not any("model.visual" in m for m in matched)

    def test_all_linear_freeze_blocks_keep_aligner(self) -> None:
        """freeze: [model.visual.blocks] keeps the merger (aligner) LoRA."""
        model = MiniVL()
        matched = find_all_linear_target_modules(
            model, freeze_patterns=["model.visual.blocks"]
        )

        # language_model (4) + merger (2) = 6; visual.blocks excluded
        assert len(matched) == 6
        merger = [m for m in matched if "model.visual.merger" in m]
        assert len(merger) == 2
        assert not any("model.visual.blocks" in m for m in matched)

    def test_lm_head_always_excluded(self) -> None:
        """lm_head must never be injected even without any freeze."""
        model = MiniVL()
        for patterns in (None, [], ["model.language_model"]):
            matched = find_all_linear_target_modules(model, freeze_patterns=patterns)
            assert not any("lm_head" in m for m in matched), \
                f"lm_head leaked with freeze_patterns={patterns}"

    def test_freeze_with_wildcard(self) -> None:
        """freeze pattern with {*} wildcard still excludes descendants."""
        model = MiniVL()
        matched = find_all_linear_target_modules(
            model, freeze_patterns=["model.visual.blocks.{*}"]
        )
        assert not any("model.visual.blocks" in m for m in matched)
        # merger survives
        assert any("model.visual.merger" in m for m in matched)
