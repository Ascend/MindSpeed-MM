import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from mindspeed_mm.fsdp.utils.dtensor_compat import (
    register_dtensor_anomaly_detection_ops,
)


class _ForceNanGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("nan"))


def test_register_dtensor_anomaly_detection_ops_is_idempotent():
    from torch.distributed.tensor._api import DTensor

    op = torch.ops.aten._is_any_true.default
    propagator = DTensor._op_dispatcher.sharding_propagator
    previous_strategy = propagator.op_strategy_funcs.pop(op, None)
    previous_rule = propagator.op_to_rules.pop(op, None)

    try:
        assert register_dtensor_anomaly_detection_ops() is True
        assert op in propagator.op_strategy_funcs
        assert register_dtensor_anomaly_detection_ops() is False
    finally:
        propagator.op_strategy_funcs.pop(op, None)
        propagator.op_to_rules.pop(op, None)
        if previous_strategy is not None:
            propagator.op_strategy_funcs[op] = previous_strategy
        if previous_rule is not None:
            propagator.op_to_rules[op] = previous_rule


def _dtensor_anomaly_worker(rank, world_size, init_file):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    previous_anomaly_state = torch.is_anomaly_enabled()
    try:
        register_dtensor_anomaly_detection_ops()
        mesh = init_device_mesh("cpu", (world_size,))
        source = torch.arange(4.0, requires_grad=True)
        value = distribute_tensor(source, mesh, [Shard(0)])
        torch.autograd.set_detect_anomaly(True)
        value.to_local().sum().backward()

        nan_source = torch.arange(4.0, requires_grad=True)
        nan_value = distribute_tensor(nan_source, mesh, [Shard(0)])
        nan_value = _ForceNanGradient.apply(nan_value)
        with pytest.raises(RuntimeError, match="returned nan values"):
            nan_value.to_local().sum().backward()
    finally:
        torch.autograd.set_detect_anomaly(previous_anomaly_state)
        dist.destroy_process_group()


def test_dtensor_backward_works_with_anomaly_detection():
    world_size = 2
    with tempfile.NamedTemporaryFile(delete=False) as init_file:
        init_path = init_file.name
    try:
        mp.spawn(
            _dtensor_anomaly_worker,
            args=(world_size, init_path),
            nprocs=world_size,
            join=True,
        )
    finally:
        try:
            os.remove(init_path)
        except OSError:
            pass
