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

"""Few-step rollout and DMD2 role losses for joint H3 latents."""

import torch
from .objectives import add_noise, clean_from_velocity, critic_regression_loss, shifted_sigma, student_distribution_loss


class DMD2Pipeline:

    def __init__(self, config, *, data_parallel_rank=0):
        self.config = config
        self.depth_rng = torch.Generator("cpu").manual_seed(config.rollout_seed)
        self.matching_rng = torch.Generator("cpu").manual_seed(config.matching_seed + data_parallel_rank)

    def state_dict(self):
        return {"depth_rng": self.depth_rng.get_state(), "matching_rng": self.matching_rng.get_state()}

    def load_state_dict(self, state):
        self.depth_rng.set_state(state["depth_rng"])
        self.matching_rng.set_state(state["matching_rng"])

    def sigmas(self, position):
        return (shifted_sigma(position, self.config.video_shift), shifted_sigma(position, self.config.audio_shift))

    def predict(self, model, video, audio, batch, position):
        sv, sa = self.sigmas(position)
        prediction = model(video, audio, batch=batch, sigma_video=sv, sigma_audio=sa)
        return (prediction.video_velocity, prediction.audio_velocity)

    def rollout(self, student, batch, *, track_grad):
        schedule = self.config.schedule()
        depth = int(torch.randint(len(schedule), (), generator=self.depth_rng))
        video = torch.randn_like(batch["input_latents"])
        audio = torch.randn_like(batch["audio_input_latents"])
        with torch.no_grad():
            for i in range(depth):
                sv, sa = self.sigmas(schedule[i])
                nv, na = self.sigmas(schedule[i + 1])
                vv, va = self.predict(student, video, audio, batch, schedule[i])
                if self.config.rollout_mode == "ode":
                    video = video + (nv - sv) * vv
                    audio = audio + (na - sa) * va
                else:
                    video = add_noise(clean_from_velocity(video, vv, sv), torch.randn_like(video), nv)
                    audio = add_noise(clean_from_velocity(audio, va, sa), torch.randn_like(audio), na)
        sv, sa = self.sigmas(schedule[depth])
        with torch.set_grad_enabled(track_grad):
            vv, va = self.predict(student, video, audio, batch, schedule[depth])
            return (clean_from_velocity(video, vv, sv), clean_from_velocity(audio, va, sa), depth)

    def _teacher_prediction(self, teacher, video, audio, batch, position):
        positive = self.predict(teacher, video, audio, batch, position)
        cfg_v, cfg_a = (self.config.teacher_video_cfg, self.config.teacher_audio_cfg)
        if cfg_v == 1 and cfg_a == 1:
            return positive
        if "negative_batch" not in batch:
            raise ValueError("Teacher CFG requires a precomputed negative_batch with its own packed text layout")
        negative = self.predict(teacher, video, audio, batch["negative_batch"], position)
        return tuple((n + scale * (p - n) for (p, n, scale) in zip(positive, negative, (cfg_v, cfg_a))))

    def loss(self, role, models, batch):
        if role not in ("student", "critic"):
            raise ValueError(f"Unknown role: {role}")
        generated_v, generated_a, depth = self.rollout(models.student, batch, track_grad=role == "student")
        position = float(torch.rand((), generator=self.matching_rng))
        position = self.config.matching_min + position * (self.config.matching_max - self.config.matching_min)
        sv, sa = self.sigmas(position)
        noise_v, noise_a = (torch.randn_like(generated_v), torch.randn_like(generated_a))
        noisy_v = add_noise(generated_v.detach(), noise_v, sv)
        noisy_a = add_noise(generated_a.detach(), noise_a, sa)
        if role == "student":
            with torch.no_grad():
                with torch.profiler.record_function("dmd2.teacher_generated"):
                    tv, ta = self._teacher_prediction(models.teacher, noisy_v, noisy_a, batch, position)
                with torch.profiler.record_function("dmd2.critic_score"):
                    cv, ca = self.predict(models.critic, noisy_v, noisy_a, batch, position)
                teacher_v, teacher_a = (clean_from_velocity(noisy_v, tv, sv), clean_from_velocity(noisy_a, ta, sa))
                critic_v, critic_a = (clean_from_velocity(noisy_v, cv, sv), clean_from_velocity(noisy_a, ca, sa))
            video_loss = student_distribution_loss(generated_v, teacher_v, critic_v, self.config.normalize_epsilon)
            audio_loss = student_distribution_loss(generated_a, teacher_a, critic_a, self.config.normalize_epsilon)
        else:
            cv, ca = self.predict(models.critic, noisy_v, noisy_a, batch, position)
            video_loss = critic_regression_loss(cv, noisy_v, generated_v, noise_v, sv, self.config.critic_loss_space)
            audio_loss = critic_regression_loss(ca, noisy_a, generated_a, noise_a, sa, self.config.critic_loss_space)
        return (
            video_loss + self.config.audio_loss_weight * audio_loss,
            {
                "video_loss": video_loss.detach(),
                "audio_loss": audio_loss.detach(),
                "rollout_depth": depth,
                "teacher_forwards": (
                    (1 if self.config.teacher_video_cfg == self.config.teacher_audio_cfg == 1 else 2)
                    if role == "student"
                    else 0
                ),
            },
        )
