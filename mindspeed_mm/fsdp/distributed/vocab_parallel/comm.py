"""Communication primitives for the vocab lookup (the VP analogue of expert_parallel/comm.py)."""
from typing import List, Optional

import torch
import torch.distributed as dist


def all_to_all_single(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    input_splits: Optional[List[int]] = None,
    output_splits: Optional[List[int]] = None,
) -> torch.Tensor:
    """Run an all-to-all along the first tensor dimension. None splits mean an even split."""
    if group is None or dist.get_world_size(group) == 1:
        return input_

    inputs = input_.contiguous()
    if output_splits is None:
        output = torch.empty_like(inputs)
    else:
        output = inputs.new_empty((sum(output_splits), *inputs.shape[1:]))
    dist.all_to_all_single(
        output,
        inputs,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )
    return output


class _DifferentiableAllToAll(torch.autograd.Function):
    """All-to-all whose backward is the inverse all-to-all (mirrors expert_parallel/comm.py).

    ``gradient_divide_factor`` is applied to the gradient in the backward, like EP's own all-to-all and
    like the reference's VP dispatcher: the embedding is not FSDP-managed, so nothing else divides the row
    gradient by the number of data-parallel ranks. 1.0 leaves the gradient untouched.
    """

    @staticmethod
    def forward(ctx, input_, group, input_splits, output_splits, gradient_divide_factor):
        ctx.group = group
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.gradient_divide_factor = gradient_divide_factor
        return all_to_all_single(input_, group, input_splits, output_splits)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = all_to_all_single(
            grad_output,
            ctx.group,
            ctx.output_splits,
            ctx.input_splits,
        )
        if ctx.gradient_divide_factor != 1.0:
            grad_input = grad_input / ctx.gradient_divide_factor
        return grad_input, None, None, None, None


def differentiable_all_to_all(
    input_: torch.Tensor,
    group: Optional[dist.ProcessGroup],
    input_splits: List[int],
    output_splits: List[int],
    gradient_divide_factor: float = 1.0,
) -> torch.Tensor:
    if gradient_divide_factor <= 0:
        raise ValueError("gradient_divide_factor must be positive.")
    return _DifferentiableAllToAll.apply(input_, group, input_splits, output_splits, gradient_divide_factor)
