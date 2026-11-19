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

"""Legacy Muon checkpoint resumes and isolation from other optimizers."""
import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import OptimizerState
from mindspeed_mm.fsdp.optimizer.optimizer import MultiOptimizer
from mindspeed_mm.optimizer.muon import Muon


@pytest.fixture(scope="module", autouse=True)
def single_rank_process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    path = tmp_path_factory.mktemp("muon-checkpoint") / "process-group"
    dist.init_process_group("gloo", init_method=path.as_uri(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _build(flattened, *, momentum=.9, lr=.01):
    torch.manual_seed(7)
    model = torch.nn.Linear(4, 4)
    groups = [{"params": [model.weight], "use_muon": True, "param_names": ["weight"]}]
    if not flattened:
        groups.append({"params": [model.bias], "use_muon": False, "param_names": ["bias"]})
    optimizer = Muon(groups, lr=lr, momentum=momentum, ns_steps=3, ns_coefficients="keller")
    if flattened:
        optimizer = MultiOptimizer(model, {
            "muon": optimizer,
            "adamw": torch.optim.AdamW([model.bias], lr=lr),
        }, ["muon", "adamw"])
    return model, optimizer


def _step(model, optimizer):
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .25)
    optimizer.step()


def _save(directory, state):
    dcp.save({"optimizer": state}, storage_writer=dcp.FileSystemWriter(str(directory)))


def _load(directory, model, optimizer):
    reader = dcp.FileSystemReader(str(directory))
    wrapper = OptimizerState(model, optimizer)
    wrapper._prepare_optimizer_load(reader)
    dcp.load({"optimizer": wrapper}, storage_reader=reader)
    return wrapper.state_dict()


@pytest.mark.parametrize("flattened", [False, True])
def test_legacy_checkpoint_resumes_with_saved_momentum(tmp_path, flattened):
    model, optimizer = _build(flattened)
    _step(model, optimizer)
    saved = copy.deepcopy(OptimizerState(model, optimizer).state_dict())
    newer = {"ns_coefficients", "qk_clip_threshold", "qk_clip_alpha", "param_names"}
    if flattened:
        for key in list(saved):
            if key.startswith("param_groups.weight.") and key.rsplit(".", 1)[-1] in newer:
                saved.pop(key)
        saved["state.weight.muon_buffer"] = saved.pop("state.weight.momentum_buffer") / .1
    else:
        for group in saved["param_groups"]:
            for key in newer:
                group.pop(key)
        state = saved["state"]["weight"]
        state["muon_buffer"] = state.pop("momentum_buffer") / .1
    _save(tmp_path, saved)

    restored_model, restored = _build(flattened, momentum=.5, lr=.5)
    restored_model.load_state_dict(model.state_dict())
    current = _load(tmp_path, restored_model, restored)
    muon = optimizer.optimizers_dict["muon"] if flattened else optimizer
    buffer = current["state.weight.momentum_buffer"] if flattened else current["state"]["weight"]["momentum_buffer"]
    torch.testing.assert_close(buffer, muon.state[model.weight]["momentum_buffer"])
    if flattened:
        assert current["param_groups.weight.momentum"] == .9
        assert "state.weight.muon_buffer" not in current
    else:
        assert current["param_groups"][0]["momentum"] == .9
        assert "muon_buffer" not in current["state"]["weight"]
    _step(model, optimizer)
    _step(restored_model, restored)
    torch.testing.assert_close(list(restored_model.parameters()), list(model.parameters()))


@pytest.mark.parametrize("flattened", [False, True])
def test_current_buffer_wins_over_stale_legacy_buffer(tmp_path, flattened):
    model, optimizer = _build(flattened)
    _step(model, optimizer)
    saved = copy.deepcopy(OptimizerState(model, optimizer).state_dict())
    if flattened:
        saved["state.weight.muon_buffer"] = torch.full_like(model.weight, 100.)
    else:
        saved["state"]["weight"]["muon_buffer"] = torch.full_like(model.weight, 100.)
    _save(tmp_path, saved)

    restored_model, restored = _build(flattened)
    restored_model.load_state_dict(model.state_dict())
    current = _load(tmp_path, restored_model, restored)
    muon = optimizer.optimizers_dict["muon"] if flattened else optimizer
    buffer = current["state.weight.momentum_buffer"] if flattened else current["state"]["weight"]["momentum_buffer"]
    torch.testing.assert_close(buffer, muon.state[model.weight]["momentum_buffer"])
    _step(model, optimizer)
    _step(restored_model, restored)
    torch.testing.assert_close(list(restored_model.parameters()), list(model.parameters()))


def test_missing_learning_rate_fails_strict_load(tmp_path):
    model, optimizer = _build(True)
    _step(model, optimizer)
    saved = copy.deepcopy(OptimizerState(model, optimizer).state_dict())
    saved.pop("param_groups.weight.lr")
    _save(tmp_path, saved)
    model, optimizer = _build(True, momentum=.5, lr=.5)
    with pytest.raises(CheckpointException):
        _load(tmp_path, model, optimizer)


def test_adamw_resume_keeps_existing_path(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    _step(model, optimizer)
    _save(tmp_path, OptimizerState(model, optimizer).state_dict())
    restored_model = copy.deepcopy(model)
    restored = torch.optim.AdamW(restored_model.parameters(), lr=.5)
    _load(tmp_path, restored_model, restored)
    _step(model, optimizer)
    _step(restored_model, restored)
    torch.testing.assert_close(list(restored_model.parameters()), list(model.parameters()))
