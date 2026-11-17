# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP patch registrations in patch_manager."""

import sys
import types
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import mindspeed.megatron_adaptor

from tests.ut.utils import judge_expression

_root = Path(__file__).resolve().parents[3]


def _load_patch_manager():
    """Load patch_manager.py in an isolated sys.modules context."""
    extra = {}

    if "mindspeed_mm.patches" not in sys.modules:
        pkg = types.ModuleType("mindspeed_mm.patches")
        pkg.__path__ = [str(_root / "mindspeed_mm" / "patches")]
        extra["mindspeed_mm.patches"] = pkg

    for name in [
        "mindspeed_mm.patches.adaptive_clip_grad_patch",
        "mindspeed_mm.patches.infer_fa_patch",
        "mindspeed_mm.patches.models_patches",
        "mindspeed_mm.patches.fsdp1_patches",
        "mindspeed_mm.patches.training_patches",
        "mindspeed_mm.patches.fsdp2_patches",
        "mindspeed_mm.patches.optimizer_patch",
        "mindspeed_mm.patches.bridge_patch",
        "mindspeed_mm.patches.validate_args_patch",
    ]:
        if name not in sys.modules:
            extra[name] = MagicMock()

    ldt_pkg_name = "mindspeed_mm.patches.layerwise_disaggregated_training"
    if ldt_pkg_name not in sys.modules:
        ldt_pkg = types.ModuleType(ldt_pkg_name)
        ldt_pkg.__path__ = [str(_root / "mindspeed_mm" / "patches" / "layerwise_disaggregated_training")]
        extra[ldt_pkg_name] = ldt_pkg

    for name in [
        ldt_pkg_name + ".schedules_patch",
        ldt_pkg_name + ".training_patch",
        ldt_pkg_name + ".u_shaped_split_learning_patch",
        ldt_pkg_name + ".vlm_model_patch",
        ldt_pkg_name + ".utils_patch",
    ]:
        if name not in sys.modules:
            extra[name] = MagicMock()

    with patch.dict(sys.modules, extra):
        spec = importlib.util.spec_from_file_location(
            "mindspeed_mm.patches.patch_manager",
            str(_root / "mindspeed_mm" / "patches" / "patch_manager.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


_mod = _load_patch_manager()


class TestPatchManagerLdtRegistrations:

    def test_ldt_config_has_all_vtp_patches(self):
        ldt_config = _mod.PatchesManager.configs.get("layerwise_disaggregated_training")
        judge_expression(ldt_config is not None)

        target_paths = [entry[0] for entry in ldt_config]

        expected_targets = [
            "megatron.core.optimizer.clip_grads.get_grad_norm_fp32",
            "torch.distributed.barrier",
            "torch.distributed.all_gather_into_tensor",
            "megatron.training.utils.reduce_max_stat_across_model_parallel_group",
            "megatron.training.utils.logical_and_across_model_parallel_group",
        ]
        for target in expected_targets:
            judge_expression(target in target_paths)

    def test_ldt_patch_functions_are_callable(self):
        ldt_config = _mod.PatchesManager.configs["layerwise_disaggregated_training"]
        for target_path, replacement_func in ldt_config:
            judge_expression(callable(replacement_func))
