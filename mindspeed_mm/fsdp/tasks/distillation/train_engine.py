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

"""Role-aware DMD2 updates using injectable MindSpeed-MM execution services."""

import hashlib
import json
import random
import time
import numpy as np
import torch
from .pipeline import DMD2Pipeline


def collective_guard(error):
    errors = [error]
    if torch.distributed.is_initialized():
        errors = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(errors, error)
    failures = [f"rank {rank}: {message}" for (rank, message) in enumerate(errors) if message]
    if failures:
        raise RuntimeError("Distillation task failed: " + "; ".join(failures))


class DistillationEngine:

    def __init__(
        self,
        models,
        optimizers,
        schedulers,
        dataloader,
        config,
        *,
        gradient_accumulation=1,
        global_batch_size=1,
        data_parallel_rank=0,
        move_batch=None,
        clip_grad=None,
        prepare_backward=None,
        synchronize=None,
        on_update=None,
        asset_identity=None,
    ):
        if set(optimizers) != {"student", "critic"} or set(schedulers) != set(optimizers):
            raise ValueError("Student and critic require independent optimizers and schedulers")
        if gradient_accumulation < 1 or global_batch_size < 1:
            raise ValueError("Gradient accumulation and global batch must be positive")
        if any((p.requires_grad for p in models.teacher.parameters())):
            raise ValueError("Teacher must be frozen before building the engine")
        expected = {name: {id(p) for p in getattr(models, name).parameters() if p.requires_grad} for name in optimizers}
        for name, optimizer in optimizers.items():
            actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if not expected[name] or actual != expected[name]:
                raise ValueError(f"Optimizer parameters do not match {name} trainable parameters")
        if expected["student"] & expected["critic"]:
            raise ValueError("Student and critic cannot share trainable parameters")
        self.models, self.optimizers, self.schedulers = (models, optimizers, schedulers)
        self.dataloader, self.config = (dataloader, config)
        self.pipeline = DMD2Pipeline(config, data_parallel_rank=data_parallel_rank)
        self.gradient_accumulation, self.global_batch_size = (gradient_accumulation, global_batch_size)
        self.move_batch = move_batch or (lambda batch: batch)
        self.clip_grad = clip_grad or self._cpu_clip
        self.prepare_backward = prepare_backward or (lambda model, last: None)
        self.synchronize = synchronize or (lambda: None)
        self.on_update = on_update
        self.update, self.consumed_samples = (0, 0)
        self.counts = {"student": 0, "critic": 0}
        self._iterator = None
        self.asset_identity = asset_identity or {}
        signature = {
            "recipe": config.signature(),
            "accumulation": gradient_accumulation,
            "global_batch": global_batch_size,
            "assets": self.asset_identity,
            "optimizers": {
                role: {
                    "type": type(optimizer).__qualname__,
                    "groups": [
                        {key: value for (key, value) in group.items() if key != "params"}
                        for group in optimizer.param_groups
                    ],
                }
                for (role, optimizer) in optimizers.items()
            },
        }
        self.signature = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        if torch.distributed.is_initialized():
            signatures = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(signatures, self.signature)
            if len(set(signatures)) != 1:
                raise ValueError("Distillation recipe, batch or assets disagree between ranks")

    @staticmethod
    def _cpu_clip(model, maximum):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), maximum if maximum else float("inf"))

    def _next_batch(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise ValueError("Distillation DataLoader produced no complete batches") from error

    def train_update(self):
        role = self.config.role_at(self.update)
        active = getattr(self.models, role)
        inactive = getattr(self.models, "critic" if role == "student" else "student")
        self.models.train()
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)
        self.synchronize()
        start = time.perf_counter()
        metrics = {"video_loss": 0.0, "audio_loss": 0.0}
        depths = []
        teacher_forwards = 0
        for micro in range(self.gradient_accumulation):
            error, batch = (None, None)
            try:
                batch = self.move_batch(self._next_batch())
            except Exception as cause:
                error = f"DataLoader failed: {type(cause).__name__}: {cause}"
            collective_guard(error)
            self.prepare_backward(active, micro == self.gradient_accumulation - 1)
            loss, values = self.pipeline.loss(role, self.models, batch)
            collective_guard(None if torch.isfinite(loss).all().item() else f"Non-finite {role} loss")
            (loss / self.gradient_accumulation).backward()
            for name in metrics:
                metrics[name] += float(values[name]) / self.gradient_accumulation
            depths.append(values["rollout_depth"])
            teacher_forwards += values["teacher_forwards"]
        if any((p.grad is not None for p in inactive.parameters())):
            raise RuntimeError(f"Inactive role received gradients during {role} update")
        maximum = getattr(self.config, role).clip_grad
        grad_norm = self.clip_grad(active, maximum)
        collective_guard(
            None if torch.isfinite(torch.as_tensor(grad_norm)).all().item() else f"Non-finite {role} gradient"
        )
        used_lr = self.optimizers[role].param_groups[0]["lr"]
        self.optimizers[role].step()
        self.schedulers[role].step()
        self.optimizers[role].zero_grad(set_to_none=True)
        self.update += 1
        self.counts[role] += 1
        self.consumed_samples += self.global_batch_size
        self.synchronize()
        metrics.update(
            update=self.update,
            role=role,
            role_update=self.counts[role],
            lr=used_lr,
            grad_norm=float(grad_norm),
            seconds=time.perf_counter() - start,
            rollout_depths=depths,
            consumed_samples=self.consumed_samples,
            objective="critic" if role == "critic" else "dmd2",
            teacher_forwards=teacher_forwards,
        )
        if self.on_update is not None:
            self.on_update(metrics)
        return metrics

    def runtime_state_dict(self):
        if not hasattr(self.dataloader, "state_dict") or not hasattr(self.dataloader, "load_state_dict"):
            raise ValueError("Exact resume requires a stateful DataLoader")
        numpy_state = np.random.get_state()
        device = next(self.models.student.parameters()).device.type
        state = {
            "signature": self.signature,
            "update": self.update,
            "counts": dict(self.counts),
            "consumed_samples": self.consumed_samples,
            "pipeline": self.pipeline.state_dict(),
            "schedulers": {name: scheduler.state_dict() for (name, scheduler) in self.schedulers.items()},
            "dataloader": self.dataloader.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "python_rng": random.getstate(),
            "numpy_rng": (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)), *numpy_state[2:]),
            "device": device,
        }
        if device in ("npu", "cuda"):
            state["device_rng"] = getattr(torch, device).get_rng_state()
        return state

    def validate_runtime_state(self, state):
        if state["signature"] != self.signature:
            raise ValueError("Checkpoint recipe/batch/asset identity differs from this run")
        update = state["update"]
        if update < 0 or state["counts"] != self.config.role_counts(update):
            raise ValueError("Checkpoint role counters disagree with update schedule")
        if state["consumed_samples"] != update * self.global_batch_size:
            raise ValueError("Checkpoint consumed-sample count is inconsistent")
        if state["device"] != next(self.models.student.parameters()).device.type:
            raise ValueError("Exact resume requires the same device backend")

    def load_runtime_state_dict(self, state):
        self.validate_runtime_state(state)
        for name, scheduler in self.schedulers.items():
            scheduler.load_state_dict(state["schedulers"][name])
        self.dataloader.load_state_dict(state["dataloader"])
        self.pipeline.load_state_dict(state["pipeline"])
        self.update, self.counts, self.consumed_samples = (
            state["update"],
            dict(state["counts"]),
            state["consumed_samples"],
        )
        self._iterator = iter(self.dataloader)
        random.setstate(state["python_rng"])
        ns = state["numpy_rng"]
        np.random.set_state((ns[0], ns[1].numpy().astype(np.uint32), *ns[2:]))
        torch.set_rng_state(state["torch_rng"])
        if "device_rng" in state:
            getattr(torch, state["device"]).set_rng_state(state["device_rng"])
