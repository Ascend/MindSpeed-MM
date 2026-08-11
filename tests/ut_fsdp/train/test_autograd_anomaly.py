from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindspeed_mm.fsdp.train import trainer as trainer_module


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_initialization_configures_autograd_anomaly_detection(
    monkeypatch, enabled
):
    previous_anomaly_state = trainer_module.torch.is_anomaly_enabled()
    set_detect_anomaly = trainer_module.torch.autograd.set_detect_anomaly
    args = SimpleNamespace(
        training=SimpleNamespace(
            allow_hf32=None,
            plugin=[],
            seed=1234,
            use_deter_comp=False,
        ),
        parallel=SimpleNamespace(
            fsdp_plan=SimpleNamespace(cpu_offload=False),
            to_dict=dict,
        ),
    )
    trainer = trainer_module.Trainer.__new__(trainer_module.Trainer)
    trainer.args = args

    detect_anomaly = Mock(wraps=set_detect_anomaly)
    monkeypatch.setattr(
        trainer_module.torch.autograd, "set_detect_anomaly", detect_anomaly
    )
    monkeypatch.setattr(trainer_module, "print_rank", Mock())
    monkeypatch.setattr(trainer_module, "set_allow_hf32", Mock())
    monkeypatch.setattr(trainer_module, "get_torch_device", Mock())
    monkeypatch.setattr(trainer_module, "set_accelerator_compatible", Mock())
    monkeypatch.setattr(trainer_module, "set_log_level", Mock())
    monkeypatch.setattr(trainer_module.envs, "MM_DETECT_ANOMALY", enabled)
    monkeypatch.setattr(trainer_module.envs, "get", Mock(return_value=0))
    monkeypatch.setattr(trainer_module.torch.accelerator, "set_device_index", Mock())
    monkeypatch.setattr(trainer_module, "set_seed", Mock())
    monkeypatch.setattr(trainer_module, "import_plugin", Mock())
    monkeypatch.setattr(
        trainer_module.torch.distributed, "is_initialized", Mock(return_value=True)
    )
    monkeypatch.setattr(trainer_module, "init_parallel_state", Mock())

    try:
        trainer.initialize()

        detect_anomaly.assert_called_once_with(enabled)

        value = trainer_module.torch.tensor([-1.0], requires_grad=True)
        if enabled:
            with pytest.raises(RuntimeError, match="returned nan values"):
                trainer_module.torch.sqrt(value).sum().backward()
        else:
            trainer_module.torch.sqrt(value).sum().backward()
            assert trainer_module.torch.isnan(value.grad).all()
    finally:
        set_detect_anomaly(previous_anomaly_state)
