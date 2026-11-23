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

"""Validated MiniMax-H3 DMD2 recipe and role settings."""

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math


@dataclass
class RoleConfig:
    lr: float = 0.0001
    clip_grad: float = 1.0
    rank: int = 128
    alpha: float = 128.0
    targets: list[str] = field(default_factory=lambda: ["qkv_proj", "out_proj", "fc1", "fc2", "adaln_proj.linear"])
    import_lora: dict | None = None
    base_model_path: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.lr) or self.lr <= 0 or (not math.isfinite(self.clip_grad)) or (self.clip_grad < 0):
            raise ValueError("Role lr must be positive and clip_grad nonnegative")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError("Role LoRA rank must be a positive integer")
        if not math.isfinite(self.alpha) or self.alpha <= 0 or (not self.targets):
            raise ValueError("Role LoRA alpha and target modules must be configured")
        if not isinstance(self.targets, list) or any((not isinstance(name, str) or not name for name in self.targets)):
            raise ValueError("Role targets must be a list of nonempty module suffixes")
        if self.import_lora is not None and (not isinstance(self.import_lora, dict)):
            raise ValueError("import_lora must be a configuration mapping")


@dataclass
class DMD2Config:
    student: RoleConfig = field(default_factory=RoleConfig)
    critic: RoleConfig = field(default_factory=lambda: RoleConfig(lr=0.0002))
    student_updates: int = 1
    critic_updates: int = 4
    student_steps: int = 8
    student_schedule: list[float] | None = None
    rollout_mode: str = "ode"
    video_shift: float = 12.0
    audio_shift: float = 3.0
    matching_min: float = 0.02
    matching_max: float = 0.98
    teacher_video_cfg: float = 1.0
    teacher_audio_cfg: float = 1.0
    critic_loss_space: str = "x0"
    audio_loss_weight: float = 1.0
    normalize_epsilon: float = 1e-06
    rollout_seed: int = 1234
    matching_seed: int = 5678
    teacher_path: str | None = None
    resume_from: str | None = None

    def __post_init__(self):
        for name in ("student", "critic"):
            if isinstance(getattr(self, name), dict):
                setattr(self, name, RoleConfig(**getattr(self, name)))
        for name in ("student_updates", "critic_updates", "student_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("video_shift", "audio_shift", "normalize_epsilon"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.matching_min < self.matching_max < 1:
            raise ValueError("Matching base positions must satisfy 0 < min < max < 1")
        if self.rollout_mode not in ("ode", "sde") or self.critic_loss_space not in ("x0", "velocity"):
            raise ValueError("Unsupported rollout mode or critic loss space")
        for value in (self.audio_loss_weight, self.teacher_video_cfg, self.teacher_audio_cfg):
            if not math.isfinite(value) or value < 0:
                raise ValueError("Loss weights and guidance scales must be finite and nonnegative")
        schedule = self.schedule()
        if len(schedule) != self.student_steps or schedule[0] != 1:
            raise ValueError("Student schedule must have student_steps entries and begin at pure noise (1)")
        if any((not math.isfinite(u) or not 0 < u <= 1 for u in schedule)) or any(
            (a <= b for (a, b) in zip(schedule, schedule[1:]))
        ):
            raise ValueError("Student schedule must be finite, strictly descending and above zero")

    def schedule(self):
        return (
            self.student_schedule
            if self.student_schedule is not None
            else [1 - i / self.student_steps for i in range(self.student_steps)]
        )

    def role_at(self, update):
        return "student" if update % (self.student_updates + self.critic_updates) < self.student_updates else "critic"

    def role_counts(self, updates):
        cycles, tail = divmod(updates, self.student_updates + self.critic_updates)
        student = cycles * self.student_updates + min(tail, self.student_updates)
        return {"student": student, "critic": updates - student}

    def signature(self):
        fields = asdict(self)
        fields.pop("resume_from")
        return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()
