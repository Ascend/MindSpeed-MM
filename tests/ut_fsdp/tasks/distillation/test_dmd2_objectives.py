# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""Deterministic check of the DMD2 student distribution-matching objective.

The production module is pure torch (no framework bootstrap), so it is loaded
directly from its file path instead of going through package imports.
"""

import importlib.util
import pathlib

import torch

_OBJECTIVES_PATH = (
    pathlib.Path(__file__).resolve().parents[4]
    / "mindspeed_mm"
    / "fsdp"
    / "tasks"
    / "distillation"
    / "objectives.py"
)

_spec = importlib.util.spec_from_file_location("dmd2_objectives_under_test", _OBJECTIVES_PATH)
objectives = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(objectives)


def test_student_distribution_loss_matches_fp32_reference():
    # One AV sample: the leading dimension is stereo channels, not batch.
    generated = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    critic = torch.tensor(
        [
            [0.05, 1.05, 1.95, 3.10],
            [3.90, 5.05, 6.00, 7.10],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    # Independent FP32 reference, written from the objective's definition:
    # direction = (critic - teacher) / mean(|generated - teacher|)
    # loss = 0.5 * mean((generated - (generated - direction)) ** 2)
    #      = 0.5 * mean(direction ** 2)
    difference = generated.detach() - teacher.detach()
    scale = difference.abs().mean().clamp_min(1e-6)
    direction = (critic.detach() - teacher.detach()) / scale
    expected_loss = 0.5 * (direction**2).mean()

    loss = objectives.student_distribution_loss(generated, teacher, critic)
    loss.backward()

    # d loss / d generated = (generated - target) / numel = direction / numel,
    # because the target is detached (teacher/critic stay out of the graph).
    expected_gradient = direction / generated.numel()

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(generated.grad, expected_gradient)
    assert teacher.grad is None
    assert critic.grad is None
