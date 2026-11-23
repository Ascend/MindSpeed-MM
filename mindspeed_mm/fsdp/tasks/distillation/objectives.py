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

"""Normalized distribution matching and velocity regression for MiniMax-H3 DMD2."""

import torch
import torch.nn.functional as F


def shifted_sigma(position, shift):
    return shift * position / (1 + (shift - 1) * position)


def add_noise(clean, noise, sigma):
    return (1 - sigma) * clean + sigma * noise


def clean_from_velocity(noisy, velocity, sigma):
    return noisy - sigma * velocity


def student_distribution_loss(generated, teacher_clean, critic_clean, epsilon=1e-06):
    # One AV sample: the audio leading dimension is stereo channels, not batch.
    with torch.no_grad():
        scale = (generated.detach().float() - teacher_clean.float()).abs().mean().clamp_min(epsilon)
        direction = (critic_clean.float() - teacher_clean.float()) / scale
    target = generated.detach().float() - direction
    loss = 0.5 * F.mse_loss(generated.float(), target)
    return loss


def critic_regression_loss(prediction, noisy, generated, noise, sigma, space):
    target = generated.detach()
    if space == "x0":
        return F.mse_loss(clean_from_velocity(noisy.float(), prediction.float(), sigma), target.float())
    if space == "velocity":
        return F.mse_loss(prediction.float(), noise.float() - target.float())
    raise ValueError(f"Unsupported critic loss space: {space}")
