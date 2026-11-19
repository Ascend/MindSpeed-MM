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

"""Round-trip physically sharded MoE weights through PyTorch DCP."""

import pytest
import torch
import torch.distributed.checkpoint as dcp


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_moe_shards_round_trip_through_dcp(tmp_path, monkeypatch, dtype):
    monkeypatch.setenv("NON_MEGATRON", "true")
    from mindspeed_mm.fsdp.checkpoint import sharded_dcp

    key = "layers.0.mlp.experts.down_proj"
    expected = torch.arange(40, dtype=dtype).reshape(4, 5, 2)
    sharded_dcp.save_sharded_tensors(
        {key: expected}, tmp_path, fsdp_shards=4, ep_size=2, expert_keys={key}
    )

    reader = dcp.FileSystemReader(tmp_path / "release")
    metadata = reader.read_metadata().state_dict_metadata[f"model.{key}"]
    assert metadata.size == expected.shape
    assert metadata.properties.dtype == dtype
    assert [(tuple(chunk.offsets), tuple(chunk.sizes)) for chunk in metadata.chunks] == [
        ((0, 0, 0), (2, 3, 2)),
        ((2, 0, 0), (2, 3, 2)),
        ((0, 3, 0), (2, 2, 2)),
        ((2, 3, 0), (2, 2, 2)),
    ]

    restored = {"model": {key: torch.full_like(expected, float("nan"))}}
    dcp.load(restored, storage_reader=reader)
    torch.testing.assert_close(restored["model"][key], expected, rtol=0, atol=0)
