"""Standalone owner collection excludes unsupported modes and rejects rank disagreement."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Shard

from mindspeed_mm.optimizer.muon.owner_plan import collect_owner_batches


def _distributed_checks(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=3,
                            timeout=timedelta(seconds=60))
    try:
        mesh = DeviceMesh("cpu", [0, 1, 2])
        for mode, shape, count_key in (("per_head", (4, 8), "muon_num_heads"),
                                       ("per_expert", (2, 4, 8), "muon_num_experts")):
            layout_mesh = DeviceMesh("cpu", [0, 1, 2], mesh_dim_names=("ep",)) if mode == "per_expert" else mesh
            parameter = torch.nn.Parameter(distribute_tensor(torch.ones(shape), layout_mesh, [Shard(0)]))
            parameter.grad = torch.ones_like(parameter)
            batches, parameter_ids = collect_owner_batches(
                [dict(params=[parameter], use_muon=True, muon_mode=mode, **{count_key: 2})],
                wave_bytes=1 << 20, max_inflight=2,
            )
            assert not batches and not parameter_ids
            if rank == 2:
                assert parameter.to_local().numel() == 0

        params = [torch.nn.Parameter(distribute_tensor(torch.ones(4, 16), mesh, [Shard(0)]))
                  for _ in range(2)]
        group = dict(params=params, use_muon=True, param_names=["a", "b"],
                     ns_steps=5, momentum=.95, nesterov=True)
        batches, parameter_ids = collect_owner_batches([group], wave_bytes=1 << 20, max_inflight=2)
        assert not batches and not parameter_ids  # No gradients means nothing to exchange.
        for parameter in params:
            torch.testing.assert_close(parameter.full_tensor(), torch.ones(4, 16))

        # A cached plan must still reject later disagreements before changing state.
        for mismatch in ("gradient", "names", "steps", "coefficients"):
            group.update(param_names=["a", "b"], ns_steps=5, ns_coefficients="polar_express")
            for parameter in params:
                parameter.grad = torch.ones_like(parameter)
            batches, parameter_ids = collect_owner_batches([group], wave_bytes=1 << 20, max_inflight=2)
            assert parameter_ids == {id(parameter) for parameter in params}
            assert len(batches) == 1
            _, entries = next(iter(batches.values()))
            assert [id(entry[0]) for entry in entries] == [id(parameter) for parameter in params]
            assert all(entry[1] is group and entry[2:] == (4, 16) for entry in entries)
            if rank == 0:
                if mismatch == "gradient":
                    params[0].grad = None
                elif mismatch == "names":
                    group["param_names"][0] = "changed"
                elif mismatch == "steps":
                    group["ns_steps"] = 10
                else:
                    group["ns_coefficients"] = "keller"
            with pytest.raises(RuntimeError, match="manifest differs"):
                collect_owner_batches([group], wave_bytes=1 << 20, max_inflight=2)
            for parameter in params:
                torch.testing.assert_close(parameter.full_tensor(), torch.ones(4, 16))
    finally:
        dist.destroy_process_group()


def test_owner_collection_and_rank_agreement(tmp_path):
    mp.spawn(_distributed_checks, args=((tmp_path / "owner_plan").as_uri(),), nprocs=3)
