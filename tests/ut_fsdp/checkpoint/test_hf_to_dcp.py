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

"""Test Kimi-K3.1 router validation and recovery from interrupted conversion."""

from concurrent.futures import ThreadPoolExecutor
import json
import sys

import pytest
from safetensors.torch import save_file
import torch
import torch.distributed.checkpoint as dcp


@pytest.fixture
def kimi_source(tmp_path):
    src = tmp_path / "hf"
    src.mkdir()
    config = {"text_config": {
        "num_hidden_layers": 1,
        "num_experts": 2,
        "hidden_size": 2,
        "moe_intermediate_size": 3,
        "first_k_dense_replace": 0,
    }}
    (src / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return src


@pytest.mark.parametrize(("suffix", "tail"), [("weight", (2,)), ("e_score_correction_bias", ())])
@pytest.mark.parametrize("rows", [1, 3])
def test_rejects_router_rows_different_from_num_experts(kimi_source, monkeypatch, suffix, tail, rows):
    monkeypatch.setenv("NON_MEGATRON", "true")
    from mindspeed_mm.fsdp.checkpoint.convert import KimiK31WeightTransformPipeline

    pipeline = KimiK31WeightTransformPipeline(str(kimi_source))
    key = f"language_model.model.layers.0.block_sparse_moe.gate.{suffix}"
    tensor = torch.ones((rows, *tail))
    with pytest.raises(ValueError, match="incompatible router shape"):
        pipeline.describe_hf_tensors({key: {"dtype": "F32", "shape": list(tensor.shape)}})
    with pytest.raises(ValueError, match="incompatible router shape"):
        pipeline.hf_to_dcp(key, tensor)


def test_resume_reuses_committed_jobs(kimi_source, tmp_path, monkeypatch):
    monkeypatch.setenv("NON_MEGATRON", "true")
    from mindspeed_mm.fsdp.checkpoint import hf_to_dcp, sharded_dcp

    src = kimi_source
    dst = tmp_path / "dcp"

    ordinary_key = "language_model.model.embed_tokens.weight"
    moe_prefix = "language_model.model.layers.0.block_sparse_moe"
    prefix = f"{moe_prefix}.experts"
    ordinary = {
        ordinary_key: torch.arange(6, dtype=torch.float32).reshape(3, 2),
        f"{moe_prefix}.gate.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        f"{moe_prefix}.gate.e_score_correction_bias": torch.tensor([-1.0, 1.0]),
    }
    experts = {
        f"{prefix}.{expert}.w{part}.weight": (
            torch.arange(6, dtype=torch.bfloat16) + 20 * expert + 6 * part
        ).reshape((2, 3) if part == 2 else (3, 2))
        for expert in range(2)
        for part in (1, 2, 3)
    }
    weight_map = {}
    for filename, weights in (("ordinary.safetensors", ordinary), ("experts.safetensors", experts)):
        save_file(weights, src / filename)
        weight_map.update(dict.fromkeys(weights, filename))
    index = {"weight_map": weight_map}
    (src / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")

    gate_key = f"{prefix}.gate_up_proj"
    down_key = f"{prefix}.down_proj"
    expected = {
        **ordinary,
        gate_key: torch.stack([
            torch.cat([experts[f"{prefix}.{expert}.w{part}.weight"] for part in (1, 3)]).T
            for expert in range(2)
        ]),
        down_key: torch.stack([experts[f"{prefix}.{expert}.w2.weight"].T for expert in range(2)]),
    }

    emit = sharded_dcp.emit

    def interrupt_second_expert_group(dest, path, name, chunk, tensor):
        if name == f"model.{gate_key}" and chunk["offsets"][:2] == [1, 1]:
            raise RuntimeError("injected conversion interruption")
        return emit(dest, path, name, chunk, tensor)

    def in_process_executor(*, mp_context, **kwargs):
        return ThreadPoolExecutor(**kwargs)

    argv = [
        "hf_to_dcp", "--model-id", "kimi_k31_pro", "--src-dir", str(src), "--dst-dir", str(dst),
        "--fsdp-shards", "4", "--ep-size", "2", "--workers", "1", "--read-threads", "2",
    ]
    threads = torch.get_num_threads()
    try:
        with monkeypatch.context() as interrupted:
            # Keep the worker in process so it sees the injected write failure.
            interrupted.setattr(hf_to_dcp, "ProcessPoolExecutor", in_process_executor)
            interrupted.setattr(hf_to_dcp, "_WORKER_PIPELINE", None)
            interrupted.setattr(sharded_dcp, "emit", interrupt_second_expert_group)
            interrupted.setattr(sys, "argv", argv)
            with pytest.raises(RuntimeError, match="injected conversion interruption"):
                hf_to_dcp.main()
    finally:
        torch.set_num_threads(threads)

    release = dst / "release"
    receipts = release / ".receipts"
    tracker = dst / "latest_checkpointed_iteration.txt"
    committed = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for index in range(2)
        for path in (receipts / f"{index:06d}.json", release / f"__hf_{index:06d}.distcp")
    }
    assert (release / ".incomplete").is_file()
    assert not (release / ".metadata").exists()
    assert not tracker.exists()
    assert not (receipts / "000002.json").exists()
    assert not list(release.glob("__hf_000002.distcp*"))

    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    hf_to_dcp.main()

    for path, before in committed.items():
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert (receipts / "000002.json").is_file()
    assert tracker.read_text(encoding="utf-8").strip() == "release"
    assert not (release / ".incomplete").exists()

    reader = dcp.FileSystemReader(release)
    metadata = reader.read_metadata().state_dict_metadata
    restored = {"model": {key: torch.full_like(value, float("nan")) for key, value in expected.items()}}
    dcp.load(restored, storage_reader=reader)
    for key, value in expected.items():
        assert metadata[f"model.{key}"].properties.dtype == value.dtype
        torch.testing.assert_close(restored["model"][key], value, rtol=0, atol=0)
