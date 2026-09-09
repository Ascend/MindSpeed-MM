"""Compatibility helpers for PyTorch DTensor."""

import torch


def register_dtensor_anomaly_detection_ops() -> bool:
    """Register `_is_any_true` so anomaly checks can inspect DTensor gradients.

    PyTorch autograd anomaly detection uses the internal `_is_any_true` operator
    when checking backward outputs. PyTorch 2.10 does not register a DTensor
    strategy for this operator, so checking a DTensor gradient raises
    ``NotImplementedError`` before anomaly detection can report the real node.

    Returns ``True`` when this call installs the strategy and ``False`` when the
    current PyTorch version already provides one.
    """
    from torch.distributed.tensor._api import DTensor
    from torch.distributed.tensor._op_schema import OpStrategy
    from torch.distributed.tensor._ops._math_ops import common_reduction_strategy
    from torch.distributed.tensor._ops.registration import register_op_strategy

    op = torch.ops.aten._is_any_true.default
    propagator = DTensor._op_dispatcher.sharding_propagator
    if op in propagator.op_strategy_funcs or op in propagator.op_to_rules:
        return False

    # The decorator stores this callback in DTensor's sharding propagator.
    # DTensor invokes it later when dispatching ``aten._is_any_true``.
    @register_op_strategy(op)
    def _is_any_true_strategy(op_schema):
        input_strategy = op_schema.args_schema[0]
        if not isinstance(input_strategy, OpStrategy):
            raise AssertionError(f"Expected OpStrategy, got {type(input_strategy)}")

        return common_reduction_strategy(
            input_strategy,
            list(range(input_strategy.ndim)),
            reduction_linear=True,
            reduction_op="sum",
        )

    return True
