# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import Optional, List

import torch
import torch.distributed as dist

from .utils import cal_split_sizes


PERMUTE_DIMS1 = {
    4: (1, 2, 3, 0),
    5: (1, 2, 3, 0, 4),
}


PERMUTE_DIMS2 = {
    4: (1, 2, 0, 3),
    5: (1, 2, 0, 3, 4),
}


def adjust_tensor_dimensions(tensor, scatter_idx, gather_idx):
    """
    Adjusts the dimensions of a tensor to move scatter_idx and gather_idx to dim 0 and dim 1 respectively.

    Args:
        tensor (torch.Tensor): The input tensor.
        scatter_idx (int): The index of the dimension to scatter.
        gather_idx (int): The index of the dimension to gather.

    Returns:
        tuple: A tuple containing the adjusted tensor and the list of adjusted dimensions.
    """
    dims = list(range(tensor.dim()))

    if gather_idx == 0:
        if scatter_idx != 1:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
        else:
            dims[scatter_idx], dims[gather_idx] = dims[gather_idx], dims[scatter_idx]

    elif gather_idx == 1:
        if scatter_idx != 0:
            # If scatter_idx is not 0, move it to 0
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
    else:
        if scatter_idx == 0:
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
        else:
            dims[0], dims[scatter_idx] = dims[scatter_idx], dims[0]
            dims[1], dims[gather_idx] = dims[gather_idx], dims[1]
    return tensor.permute(dims).contiguous(), dims


def unadjust_tensor_dimensions(tensor, adjusted_dims):
    """
    Reverses the dimension adjustments using the list of adjusted dimensions.

    Args:
        tensor (torch.Tensor): The tensor whose dimensions need to be restored.
        adjusted_dims (list): The list of adjusted dimensions used during the adjustment process.

    Returns:
        torch.Tensor: The tensor with its dimensions reverted to the original order.
    """
    inverse_dims = [0] * len(adjusted_dims)

    for new_pos, old_pos in enumerate(adjusted_dims):
        inverse_dims[old_pos] = new_pos

    # Restore the dimension order
    unadjusted_tensor = tensor.permute(inverse_dims).contiguous()
    return unadjusted_tensor


def _all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. The function supports aligned and unaligned
    data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        group (dist.ProcessGroup): The process group perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.

    Note:
        - The tensor will be split into `world_size` chunks along the `scatter_dim`. Each process will receive one
          chunk. If the total size of the `scatter_dim` is not divisible by `world_size`, the extra elements will be
          distributed to the first few processes, ensuring that no process receives more than one additional element
          compared to the others.
        - The tensor will be gathered along the `gather_dim`, with each process contributing its part to form the
          final output tensor. The gathering process also supports unaligned data, where the remainder elements
          are distributed to the first few processes.
    """

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return input_

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size
    gather_mod = gather_size % world_size
    scatter_mod = scatter_size % world_size

    if gather_mod == 0 and scatter_mod == 0:
        # In the case of aligned data (both scatter_size and gather_size are divisible by world_size),
        # _aligned_all_to_all function performs better than _partial_unaligned_all_to_all function
        return _aligned_all_to_all(input_, group, scatter_dim, gather_dim)
    elif gather_mod != 0 and scatter_mod != 0:
        return _full_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)
    else:
        return _partial_unaligned_all_to_all(input_, group, scatter_dim, gather_dim, gather_size)


def _full_unaligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. This function supports unaligned scatter
    and gather sizes.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        world_size (int): The number of processes in the process group.
        group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int): The index of the dimension that needs to be scattered.
        gather_dim (int): The index of the dimension that needs to be gathered.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    scatter_sizes = cal_split_sizes(dim_size=input_.size(scatter_dim), world_size=world_size)
    input_list = [t.contiguous() for t in torch.split(input_, scatter_sizes, scatter_dim)]

    gather_sizes = cal_split_sizes(dim_size=gather_size, world_size=world_size)
    output_list = []
    tensor_shape_base = input_list[rank].size()
    for i in range(world_size):
        tensor_shape = list(tensor_shape_base)
        tensor_shape[gather_dim] = gather_sizes[i]
        output_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))

    dist.all_to_all(output_list, input_list, group=group)

    return torch.cat(output_list, dim=gather_dim).contiguous()


def _aligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension.
    Special note: The function only supports aligned data (both scatter_size and gather_size are divisible by
    world_size)
    """
    world_size = dist.get_world_size(group)
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // world_size
    if scatter_dim == 0:
        input_t = input_.reshape([world_size] + inp_shape).contiguous()
    else:
        input_t = input_.reshape([-1, world_size] + inp_shape[scatter_dim:]).transpose(0, 1).contiguous()

    output = torch.empty_like(input_t)

    dist.all_to_all_single(output, input_t, group=group)

    output = output.view([world_size] + inp_shape).contiguous()
    output_dim = output.dim()
    if gather_dim == 1:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[gather_dim], *inp_shape[2:])
        output = output.transpose(0, 1).contiguous()
        # the shape of output is (inp_shape[0], world_size, inp_shape[gather_dim], *inp_shape[2:])
    elif gather_dim == 2:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], *inp_shape[gather_dim:])
        output = output.permute(*PERMUTE_DIMS2[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], world_size, *inp_shape[gather_dim:])
    elif gather_dim == 3:
        # the shape of input_t is (world_size, inp_shape[0], inp_shape[1], inp_shape[2], inp_shape[gather_dim])
        output = output.permute(*PERMUTE_DIMS1[output_dim]).contiguous()
        # the shape of output is (inp_shape[0], inp_shape[1], inp_shape[2], world_size, inp_shape[gather_dim])
    # The last case: gather_dim == 0:
    # the shape of input_t is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    # output requires no action
    # the shape of output is (world_size, inp_shape[gather_dim], inp_shape[0], *inp_shape[1:])
    output = output.view(inp_shape[:gather_dim] + [inp_shape[gather_dim] * world_size, ] + inp_shape[gather_dim + 1:]
                         ).contiguous()

    return output


def _partial_unaligned_all_to_all(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
    gather_size: Optional[int] = None
):
    """
    Helper function to perform the all-to-all operation. It scatters the input tensor along the specified scatter
    dimension and then gathers it along the specified gather dimension. The function supports aligned and unaligned
    data.
    Special note: In the case of aligned data (both scatter_size and gather_size are divisible by world_size),
    _partial_unaligned_all_to_all function performs worse than _aligned_all_to_all function. Therefore, in the case of
    aligning data, it is recommended to use _aligned_all_to_all function.
    """
    world_size = dist.get_world_size(group)
    input_ = input_.contiguous()
    rank = dist.get_rank(group=group)

    scatter_size = input_.size(scatter_dim)
    if gather_size is None:
        gather_size = input_.size(gather_dim) * world_size


    scatter_size_per_rank = scatter_size // world_size
    scatter_size_remainder = scatter_size % world_size
    input_split_sizes = [scatter_size_per_rank + (1 if i < scatter_size_remainder else 0) for i in range(world_size)]

    gather_size_per_rank = gather_size // world_size
    gather_size_remainder = gather_size % world_size
    output_split_sizes = [gather_size_per_rank + (1 if i < gather_size_remainder else 0) for i in range(world_size)]

    # Adjusts the dimensions of a tensor to move scatter_idx and gather_idx to dim 0 and dim 1 respectively.
    reshaped_input, reshaped_input_dims = adjust_tensor_dimensions(input_, scatter_dim, gather_dim)
    reshaped_input_shape = list(reshaped_input.shape)
    # the shape of reshaped_input is (input_.size(scatter_dim), input_.size(gather_dim), *reshaped_input_shape[2:])

    if scatter_size % world_size == 0:
        reshaped_input = reshaped_input.view(
            [world_size, input_.size(scatter_dim) // world_size, input_.size(gather_dim)] + reshaped_input_shape[2:]
        ).transpose(1, 2).contiguous()

    output_dims = reshaped_input_dims
    # Relative to reshaped_input(the return value of adjust_tensor_dimensions func),
    # which shape is (input_.size(scatter_dim), input_.size(gather_dim), *reshaped_input_shape[2:]),
    # output just swaps the 0th and 1st axes.
    output_dims[1], output_dims[0] = output_dims[0], output_dims[1]
    output = torch.empty((gather_size, input_split_sizes[rank], *reshaped_input_shape[2:]),
                         dtype=input_.dtype, device=input_.device)
    output_shape = list(output.shape)

    dist.all_to_all_single(
        output,
        reshaped_input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes if scatter_size % world_size != 0 else [1 for _ in range(world_size)],
        group=group,
    )

    if gather_size % world_size == 0 and scatter_size % world_size != 0:
        output = output.view(
            [world_size, input_split_sizes[rank], gather_size // world_size] + reshaped_input_shape[2:]
        ).transpose(1, 2).reshape(output_shape).contiguous()

    # Reverses the dimension adjustments using the list of adjusted dimensions.
    unadjust_output_ = unadjust_tensor_dimensions(output, output_dims)

    return unadjust_output_


class _AllToAll(torch.autograd.Function):
    """Custom autograd function that performs an all-to-all communication.
    This function supports both aligned and unaligned data.
    """
    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim, gather_size=None):
        """
        Forward pass: Perform all-to-all communication by scattering the input tensor along the specified scatter
        dimension and then gathering it along the specified gather dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            scatter_dim (int): The index of the dimension that needs to be scattered.
            gather_dim (int): The index of the dimension that needs to be gathered.
            gather_size (int): The size of the gather dimension.

        Returns:
            torch.Tensor: The resulting tensor after performing the all-to-all operation.
        """
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.scatter_size = input_.size(scatter_dim)
        ctx.gather_dim = gather_dim
        ctx.gather_size = gather_size
        output = _all_to_all(
            input_, process_group, scatter_dim, gather_dim, gather_size
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Perform the reverse all-to-all communication

        Args:
            grad_output (torch.Tensor): The gradient of the output with respect to the loss.

        Returns:
            tuple: The gradient of the input with respect to the loss and `None` for other arguments.
        """
        grad_output = _all_to_all(
            grad_output,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
            ctx.scatter_size
        )
        return (
            grad_output,
            None,
            None,
            None,
            None,
            None
        )


def _split(
        input_: torch.Tensor,
        pg: dist.ProcessGroup,
        dim: int = -1,
        split_sizes: Optional[List[int]] = None
) -> torch.Tensor:
    """
    Splits a tensor across the specified dimension and returns the part corresponding to the current rank,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be split.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to split the tensor. Defaults to -1 (last dimension).
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes, with the remainder
            distributed to the first few processes. Defaults to None.

    Returns:
        torch.Tensor: The part of the tensor corresponding to the current rank in the process group.
    """
    # Ensure split_sizes is a list if provided
    if split_sizes is not None and not isinstance(split_sizes, list):
        raise ValueError("split_sizes must be a list if provided.")

    # skip if only one rank involved
    world_size = dist.get_world_size(pg)

    if world_size == 1:
        return input_

    # Calculate split sizes if not provided
    if split_sizes is None:
        dim_size = input_.size(dim)
        base_size = dim_size // world_size
        remainder = dim_size % world_size

        # Calculate the size for each process
        split_sizes = [base_size + 1 if i < remainder else base_size for i in range(world_size)]

    tensor_list = torch.split(input_, split_sizes, dim=dim)

    # Get the part corresponding to the current rank
    rank = dist.get_rank(pg)
    output = tensor_list[rank].contiguous()

    return output


def _gather(input_: torch.Tensor,
            pg: dist.ProcessGroup,
            dim: int = -1,
            gather_sizes: Optional[List[int]] = None):
    """
    Gathers tensors from all processes in the process group and concatenates them along the specified dimension,
    supporting aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be gathered.
        pg (dist.ProcessGroup): The process group to perform the operation within.
        dim (int, optional): The dimension along which to concatenate the gathered tensors. Defaults to -1 (last dimension).
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.

    Returns:
        torch.Tensor: The concatenated tensor after gathering from all processes in the process group.
    """
    # Ensure gather_sizes is a list if provided
    if gather_sizes is not None and not isinstance(gather_sizes, list):
        raise ValueError("gather_sizes must be a list if provided.")

    # Skip if only one rank is involved
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return input_

    input_ = input_.contiguous()

    # Prepare the output list with appropriate shapes
    if gather_sizes:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_sizes[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_, dtype=input_.dtype, device=input_.device) for _ in range(world_size)]

    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Custom autograd function that gathers the input tensor from all processes in the model parallel region and
    concatenates them.
    During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.

    """

    @staticmethod
    def symbolic(graph, input_, process_group, dim, gather_sizes):
        """
        Define the symbolic representation of the custom operation.
        """
        return _gather(input_, process_group, dim, gather_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, gather_sizes, grad_scale="up"):
        """
        Forward pass: Gathers tensors from all processes in the specified process group and concatenates them along the specified dimension.

        Args:
            input_ (torch.Tensor): The input tensor to be processed.
            process_group (dist.ProcessGroup): The process group to perform the operation within.
            dim (int): The dimension along which to concatenate the gathered tensors.
            gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

        Returns:
            torch.Tensor: The resulting tensor after gathering and concatenating.
        """
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.gather_sizes = gather_sizes
        return _gather(input_, process_group, dim, ctx.gather_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Distribute the gradients to the input tensors and scales them according to the gradient scaling mode.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            torch.Tensor: The gradient of the input with respect to the loss.
        """
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)

        return _split(grad_output, ctx.mode, ctx.dim, ctx.gather_sizes), None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Custom autograd function that splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.

    """
    @staticmethod
    def symbolic(graph, input_, process_group, dim, split_sizes):
        return _split(input_, process_group, dim, split_sizes)

    @staticmethod
    def forward(ctx, input_, process_group, dim, split_sizes, grad_scale):
        ctx.mode = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale

        ctx.split_sizes = split_sizes

        return _split(input_, process_group, dim, ctx.split_sizes)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.mode)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.mode)
        return _gather(grad_output, ctx.mode, ctx.dim, ctx.split_sizes), None, None, None, None


def all_to_all(
        input_: torch.Tensor,
        process_group: dist.ProcessGroup,
        scatter_dim: int = 2,
        gather_dim: int = 1,
        gather_size: Optional[int] = None
):
    """
    Performs an all-to-all operation on the input tensor. The input tensor is scattered along the specified scatter
    dimension and then gathered along the specified gather dimension.
    This function supports both aligned and unaligned data.

    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        scatter_dim (int, optional): The index of the dimension that needs to be scattered. Defaults to 2.
        gather_dim (int, optional): The index of the dimension that needs to be gathered. Defaults to 1.
        gather_size (Optional[int]): The total size of the output tensor along the `gather_dim`. If not provided, it
        will be calculated as the product of the original size of the `gather_dim` of the input tensor and the
        `world_size`.

    Returns:
        torch.Tensor: The resulting tensor after performing the all-to-all operation.
    """
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim, gather_size)


def split_forward_gather_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    split_sizes: Optional[List[int]] = None,
    grad_scale: str = "down"

) -> torch.Tensor:
    """
    Splits the input tensor and keeps only the corresponding chunk for the current rank.
    During the backward pass, it gathers the gradients and scales them according to the gradient scaling mode.
    This function supports both aligned and unaligned data.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to split the tensor.
        split_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be split.
            If not provided, the tensor will be split equally among the processes. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "down".

    Returns:
        torch.Tensor: The resulting tensor after splitting and keeping only the corresponding chunk.
    """
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, split_sizes, grad_scale)


def gather_forward_split_backward(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    dim: int,
    gather_sizes: Optional[List[int]] = None,
    grad_scale: str = "up"
) -> torch.Tensor:
    """
    Gathers the input tensor from all processes in the model parallel region and concatenates them along the specified
    dimension. During the backward pass, it splits the gradients and scales them according to the gradient scaling mode.
    This function handles both aligned and unaligned data during the gather and scatter operations.
    Args:
        input_ (torch.Tensor): The input tensor to be processed.
        process_group (dist.ProcessGroup): The process group to perform the operation within.
        dim (int): The dimension along which to concatenate the gathered tensors.
        gather_sizes (Optional[List[int]], optional): A list of sizes for each part of the tensor to be gathered.
            If not provided, it is assumed that all tensors have the same shape as the input tensor. Defaults to None.
        grad_scale (str, optional): Gradient scaling mode. Can be "up", "down", or None. Defaults to "up".

    Returns:
        torch.Tensor: The resulting tensor after gathering and concatenating.
    """
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, gather_sizes, grad_scale)
