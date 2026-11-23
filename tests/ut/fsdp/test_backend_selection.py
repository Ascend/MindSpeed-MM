from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier


@pytest.mark.parametrize("shard_size,meta_init,expected", [(1, False, "ddp"), (1, True, "ddp"), (2, False, "fsdp")])
def test_backend_selection_uses_sharding_size(shard_size, meta_init, expected):
    applier = object.__new__(ParallelApplier)
    applier.config = SimpleNamespace(fsdp_plan=SimpleNamespace(cpu_offload=False))
    applier.training_config = SimpleNamespace(init_model_with_meta_device=meta_init)
    applier.parallel_state = SimpleNamespace(
        fully_shard_parallel_size=shard_size,
        get_dp_group=Mock(return_value=object()),
        get_hsdp_device_mesh=Mock(return_value=object()),
    )
    training = SimpleNamespace(init_model_with_meta_device=meta_init)
    model = object()
    with patch(
        "mindspeed_mm.fsdp.distributed.torch_parallelize.distributed_data_parallel_modules", return_value=model
    ) as ddp, patch(
        "mindspeed_mm.fsdp.distributed.torch_parallelize.fully_shard_parallel_modules", return_value=model
    ) as fsdp:
        assert applier.apply_fsdp_modules(model, training) is model
        assert ddp.call_count == (expected == "ddp")
        assert fsdp.call_count == (expected == "fsdp")
        selected = ddp if expected == "ddp" else fsdp
        assert selected.call_args.args[0] is model
        assert selected.call_args.args[-1] is training


@pytest.mark.parametrize("nested", [False, True])
def test_pregather_uses_actual_fsdp_units(nested):
    import torch
    from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params

    root = torch.nn.Module()
    unit = torch.nn.Module()
    unit.unshard = Mock()
    if nested:
        root.add_module("expert", unit)
    else:
        root = unit
    with patch("mindspeed_mm.fsdp.distributed.fully_shard_parallel.get_torch_device") as device:
        pregather_fsdp_params(root)
        unit.unshard.assert_called_once_with()
        device.return_value.synchronize.assert_called_once_with()


def test_pregather_plain_ddp_preserves_upstream_sync():
    import torch
    from mindspeed_mm.fsdp.distributed.fully_shard_parallel import pregather_fsdp_params

    with patch("mindspeed_mm.fsdp.distributed.fully_shard_parallel.get_torch_device") as device:
        pregather_fsdp_params(torch.nn.Linear(4, 4))
        device.return_value.synchronize.assert_called_once_with()


def test_no_separate_ddp_config_section():
    from mindspeed_mm.fsdp.params.parallel_args import ParallelArguments

    assert "ddp_plan" not in ParallelArguments.model_fields
    assert "use_ddp" not in ParallelArguments.model_fields


def test_tp_greater_than_one_is_explicitly_unsupported():
    applier = object.__new__(ParallelApplier)
    applier.config = SimpleNamespace(tensor_parallel_size=2)
    with pytest.raises(NotImplementedError, match="Tensor parallelization"):
        applier.apply_tp_modules(object())
