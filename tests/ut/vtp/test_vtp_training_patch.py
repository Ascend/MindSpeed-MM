# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Unit tests for VTP training_patch (train_step_wrapper)."""

from unittest.mock import MagicMock, patch

import mindspeed.megatron_adaptor

from mindspeed_mm.patchs.layerwise_disaggregated_training import training_patch as _mod
from tests.ut.utils import judge_expression


class TestTrainStepWrapper:
    """Tests for train_step_wrapper."""

    def test_vtp_disabled_passthrough(self):
        with patch.object(_mod, "is_vtp_enabled", return_value=False):
            original = MagicMock(return_value=(1.0, 0, 5.0, 0))
            wrapped = _mod.train_step_wrapper(original)

            result = wrapped("fwd", "data", "model", "opt", "sched", "cfg", "cb")
            judge_expression(result == (1.0, 0, 5.0, 0))
            original.assert_called_once_with("fwd", "data", "model", "opt", "sched", "cfg", "cb")

    def test_vtp_enabled_aggregates_grad_norm(self):
        with patch.object(_mod, "is_vtp_enabled", return_value=True), \
             patch.object(_mod, "vtp_reduce_max_stat_across_model_parallel_group", return_value=10.0) as mock_reduce:
            original = MagicMock(return_value=(1.0, 0, 5.0, 0))
            wrapped = _mod.train_step_wrapper(original)

            result = wrapped("fwd", "data", "model", "opt", "sched", "cfg", "cb")
            loss, skipped, grad_norm, num_zeros = result

            judge_expression(loss == 1.0)
            judge_expression(skipped == 0)
            judge_expression(grad_norm == 10.0)
            judge_expression(num_zeros == 0)
            mock_reduce.assert_called_once_with(5.0)

    def test_vtp_enabled_grad_norm_none(self):
        with patch.object(_mod, "is_vtp_enabled", return_value=True), \
             patch.object(_mod, "vtp_reduce_max_stat_across_model_parallel_group", return_value=None) as mock_reduce:
            original = MagicMock(return_value=(2.0, 1, None, 3))
            wrapped = _mod.train_step_wrapper(original)

            result = wrapped("fwd", "data", "model", "opt", "sched", "cfg", "cb")
            _, _, grad_norm, _ = result
            judge_expression(grad_norm is None)
            mock_reduce.assert_called_once_with(None)
