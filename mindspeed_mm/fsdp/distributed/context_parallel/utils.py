from typing import List, Optional, Union, Tuple

import torch
import torch.distributed as dist


def cal_split_sizes(dim_size: int, world_size: int):
    split_size = dim_size // world_size
    remainder = dim_size % world_size
    sizes = [split_size + (1 if i < remainder else 0) for i in range(world_size)]
    return sizes


def cal_split_sizes_multi(sizes: Union[List[int], Tuple[int, ...], torch.Tensor], world_size: int):
    """
    Calculate split sizes for multiple sizes across distributed ranks.

    Returns:
        torch.Tensor: A tensor of shape [world_size, num_sizes] where each row
                     represents the split sizes for one rank across all input sizes.

    Example:
        >>> cal_split_sizes_multi([10, 15], 3)
        tensor([[4, 5],  # Rank 0: 4 from first size, 5 from second size
                [3, 5],  # Rank 1: 3 from first size, 5 from second size
                [3, 5]]) # Rank 2: 3 from first size, 5 from second size
    """
    # Process each size independently
    splits_per_size = []
    for size in sizes:
        split_size = size // world_size
        remainder = size % world_size
        size_splits = [split_size + (1 if i < remainder else 0) for i in range(world_size)]
        splits_per_size.append(size_splits)

    return torch.tensor(splits_per_size).T