# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""LDT/VTP-aware train_step wrapper.

Wraps the original MM train_step to add cross model-parallel group
aggregation for update_successful and grad_norm, matching the behavior
of MindSpeed-LLM's layerwise_disaggregated_training/training.py.
"""

from mindspeed_mm.patchs.layerwise_disaggregated_training.parallel_state_patch import (
    is_vtp_enabled,
)
from mindspeed_mm.patchs.layerwise_disaggregated_training.utils import (
    vtp_logical_and_across_model_parallel_group,
    vtp_reduce_max_stat_across_model_parallel_group,
)


def train_step_wrapper(original_train_step):
    """Wrap MM's train_step to add VTP-aware aggregation after optimizer.step().

    When VTP is enabled with frozen sub-models across stages,
    update_successful and grad_norm must be gathered across all
    model-parallel ranks via hierarchical allreduce.
    When VTP is disabled, falls through with zero overhead.
    """
    def wrapper(
        forward_step_func, data_iterator, model, optimizer,
        opt_param_scheduler, config, call_backs
    ):
        result = original_train_step(
            forward_step_func, data_iterator, model, optimizer,
            opt_param_scheduler, config, call_backs
        )

        if not is_vtp_enabled():
            return result

        loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad = result

        grad_norm = vtp_reduce_max_stat_across_model_parallel_group(grad_norm)

        return loss_reduced, skipped_iter, grad_norm, num_zeros_in_grad

    return wrapper
