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


def reorder_output(attn_output, cp_rank, cp_size, cp_group, dim=0):
    """
    Reorder attention output chunks across context-parallel (CP) ranks using a specific pattern.
    
    This function implements a reordering scheme where output chunks are redistributed
    across CP ranks according to a predetermined pattern. Each rank computes indices 
    for where its chunks should go, exchanges this information with all ranks, then
    rearranges the local chunks accordingly.
    
    The reordering pattern follows a specific scheme:
    - Rank 0's chunks go to positions [0, 2*cp_size-1]
    - Rank 1's chunks go to positions [1, 2*cp_size-2]
    - ... and so on, creating a symmetrical mapping
    """
    index_this_rank = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], dtype=torch.int8, device=attn_output.device)
    index_list = [torch.zeros_like(index_this_rank, device=attn_output.device) for _ in range(cp_size)]
    torch.distributed.all_gather(index_list, index_this_rank, group=cp_group)

    index_list = [int(item) for item in list(torch.concat(index_list))]
    index_map = {element: idx for idx, element in enumerate(index_list)}
    target = [i for i in range(len(index_list))]
    target_list = [index_map[element] for element in target]
    
    chunks = torch.chunk(attn_output, chunks=len(target_list), dim=dim)
    reordered_chunks = [chunks[idx] for idx in target_list]
    attn_output = torch.concat(reordered_chunks, dim=dim)
    return attn_output