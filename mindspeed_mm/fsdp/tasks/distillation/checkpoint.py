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

"""Task-local, synchronous DCP for two trainable roles and rank-local state."""

import json
from pathlib import Path
import time
import uuid

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from .train_engine import collective_guard


def _world():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1


def _rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _visible(path, timeout=120):
    start = time.monotonic()
    while not path.is_file():
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f"Checkpoint file not visible: {path}")
        time.sleep(0.1)


def _schema(engine):
    return {
        role: {
            name: list(param.shape)
            for name, param in getattr(engine.models, role).named_parameters()
            if param.requires_grad
        }
        for role in ("student", "critic")
    }


def _states(engine):
    result = {}
    for role in ("student", "critic"):
        model = getattr(engine.models, role)
        result[role] = {
            "model": get_model_state_dict(model, options=StateDictOptions(ignore_frozen_params=True)),
            "optimizer": get_optimizer_state_dict(model, engine.optimizers[role]),
        }
    return result


def save_checkpoint(engine, directory):
    directory = Path(directory)
    error = None
    token = [uuid.uuid4().hex if _rank() == 0 else None]
    if torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(token, src=0)
    if _rank() == 0:
        try:
            directory.mkdir(parents=True, exist_ok=False)
            (directory / "shared_token").write_text(token[0], encoding="utf-8")
        except OSError as cause:
            error = str(cause)
    collective_guard(error)
    try:
        _visible(directory / "shared_token")
        if (directory / "shared_token").read_text(encoding="utf-8") != token[0]:
            raise ValueError("Checkpoint output is not the same shared directory on all ranks")
        state = engine.runtime_state_dict()
        temporary = directory / f"rank-{_rank()}.tmp"
        torch.save(state, temporary)
        temporary.replace(directory / f"rank-{_rank()}.pt")
    except (OSError, ValueError, TypeError) as cause:
        error = str(cause)
    collective_guard(error)
    dcp.save(_states(engine), checkpoint_id=directory / "roles")
    error = None
    if _rank() == 0:
        try:
            for rank in range(_world()):
                _visible(directory / f"rank-{rank}.pt")
            manifest = {
                "format": "mindspeed_mm_distillation_v1",
                "world_size": _world(),
                "signature": engine.signature,
                "update": engine.update,
                "trainable": _schema(engine),
            }
            temporary = directory / "manifest.tmp"
            temporary.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
            temporary.replace(directory / "manifest.json")
            (directory / "COMPLETED").write_text(str(engine.update), encoding="utf-8")
        except (OSError, ValueError) as cause:
            error = str(cause)
    collective_guard(error)
    _barrier()


def load_checkpoint(engine, directory):
    directory = Path(directory)
    error, state = None, None
    try:
        if not (directory / "COMPLETED").is_file():
            raise ValueError("Checkpoint is incomplete: missing COMPLETED marker")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["format"] != "mindspeed_mm_distillation_v1" or manifest["world_size"] != _world():
            raise ValueError("Unsupported checkpoint format or changed world size")
        if manifest["signature"] != engine.signature or manifest["trainable"] != _schema(engine):
            raise ValueError("Checkpoint recipe or trainable parameter layout is incompatible")
        state = torch.load(directory / f"rank-{_rank()}.pt", map_location="cpu", weights_only=True)
        engine.validate_runtime_state(state)
        if state["update"] != manifest["update"] or int((directory / "COMPLETED").read_text()) != state["update"]:
            raise ValueError("Checkpoint completion and runtime counters disagree")
    except Exception as cause:
        # Preflight failures must reach every rank before any enters DCP.
        error = str(cause)
    collective_guard(error)
    values = _states(engine)
    dcp.load(values, checkpoint_id=directory / "roles")
    for role in ("student", "critic"):
        model = getattr(engine.models, role)
        set_model_state_dict(model, values[role]["model"], options=StateDictOptions(strict=False))
        set_optimizer_state_dict(model, engine.optimizers[role], values[role]["optimizer"])
    engine.load_runtime_state_dict(state)
    _barrier()
