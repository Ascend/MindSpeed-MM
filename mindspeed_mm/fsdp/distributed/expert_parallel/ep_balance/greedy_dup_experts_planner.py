import torch
import torch.distributed as dist
import numpy as np
# numba kept in case JIT acceleration is needed for pure Python loops, though current logic mainly relies on Numpy vectorization
import numba
from numba import njit
from numba import types as nb_types
from numba.typed import List


def update_experts(
    selected_experts_with_dup: torch.Tensor,
    old_dup_experts_id: int,
    dup_experts_id: int,
    rearrange_workload: int,
) -> torch.Tensor:
    """
    Update expert assignments in the selected_experts_with_dup tensor.
    Only selected_experts_with_dup is a Tensor (on device).
    Other parameters are native Python types (int) computed by CPU Numpy and passed in.

    Args:
        selected_experts_with_dup: Tensor containing selected expert IDs with duplicates
        old_dup_experts_id: Old duplicated expert ID to be replaced
        dup_experts_id: New duplicated expert ID to replace with
        rearrange_workload: Number of tokens to rearrange

    Returns:
        Updated tensor with expert assignments modified
    """

    flat_tensor = selected_experts_with_dup.view(-1)

    mask = (flat_tensor == old_dup_experts_id)

    total_matches = mask.sum().item()

    if total_matches == 0:
        return selected_experts_with_dup

    if rearrange_workload >= total_matches:
        flat_tensor = torch.where(mask, dup_experts_id, flat_tensor)
        return flat_tensor.view_as(selected_experts_with_dup)

    flat_indices = torch.nonzero(mask, as_tuple=True)[0]

    indices_to_replace = flat_indices[:rearrange_workload]

    flat_tensor[indices_to_replace] = dup_experts_id

    return selected_experts_with_dup


SIG_SELECT_MIN = nb_types.Tuple([nb_types.int64, nb_types.boolean])(
    nb_types.int64,
    nb_types.int64,
    nb_types.Array(nb_types.int64, 1, 'C'),
    nb_types.Array(nb_types.int64, 2, 'C'),
    nb_types.int64
)
@njit(SIG_SELECT_MIN, nopython=True, cache=True)
def select_min_workload_rank_numba(
    ep_size: int,
    max_dup_num: int,
    workload_per_rank: np.ndarray,
    dup_experts_map: np.ndarray,
    source_rank_id: int
):
    """
    Numba accelerated version: Find the rank with minimum workload that has available slots.
    Fixes compatibility issues caused by list comprehensions and any() in original code.

    Args:
        workload_per_rank: [ep_size] array of workloads per rank
        dup_experts_map: [ep_size, max_dup_num] array where -1 indicates empty slot
        source_rank_id: Current source rank ID (to exclude)

    Returns:
        (chosen_ep_rank, success)
    """
    is_valid = np.zeros(ep_size, dtype=np.bool_)

    for r in range(ep_size):
        if r == source_rank_id:
            continue

        has_slot = False
        for s in range(max_dup_num):
            if dup_experts_map[r, s] == -1:
                has_slot = True
                break # Break inner loop once free slot is found

        is_valid[r] = has_slot

    if not np.any(is_valid):
        return -1, False

    dtype_info = np.iinfo(workload_per_rank.dtype)
    max_val = dtype_info.max

    masked_workload = np.where(is_valid, workload_per_rank, max_val)

    chosen_ep_rank = int(np.argmin(masked_workload))

    return chosen_ep_rank, True


SIG_REARRANGE_RANK = nb_types.Tuple([
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens... (returns reference)
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits (returns reference)
    nb_types.int64,       # new_dup_workload
    nb_types.Array(nb_types.int64, 2, 'C'), # group_workload_np (returns reference)
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4)) # update_experts_lists
])(
    nb_types.int64,       # num_local_experts
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits
    nb_types.int64,       # ep_rank
    nb_types.int64,       # max_dup_experts_num
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens...
    nb_types.int64,       # dup_workload
    nb_types.Array(nb_types.int64, 2, 'C'), # group_workload_np
    nb_types.int64,       # rearrange_rank
    nb_types.int64,       # min_workload_ep_rank (unused but kept for signature consistency)
    nb_types.int64,       # dup_experts_source_id
    nb_types.int64,       # dup_experts_id
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4)) # update_experts_lists
)

@njit(SIG_REARRANGE_RANK, nopython=True, cache=True)
def rearrange_rank_select_expert_ids_np(
    num_local_experts,
    input_splits,
    ep_rank,
    max_dup_experts_num,
    num_global_tokens_per_local_expert_np,
    dup_workload,
    group_workload_np,
    rearrange_rank,
    min_workload_ep_rank,
    dup_experts_source_id,
    dup_experts_id,
    update_experts_lists
):
    expert_workload_val = int(group_workload_np[rearrange_rank, dup_experts_source_id])

    rearrange_workload = expert_workload_val if expert_workload_val < dup_workload else dup_workload

    if rearrange_workload <= 0:
        return num_global_tokens_per_local_expert_np, input_splits, dup_workload, group_workload_np, update_experts_lists

    group_workload_np[rearrange_rank, dup_experts_source_id] -= rearrange_workload
    dup_workload -= rearrange_workload

    old_dup_experts_id = dup_experts_source_id + (dup_experts_source_id // num_local_experts) * max_dup_experts_num

    num_global_tokens_per_local_expert_np[rearrange_rank, old_dup_experts_id] -= rearrange_workload
    num_global_tokens_per_local_expert_np[rearrange_rank, dup_experts_id] += rearrange_workload

    if ep_rank == rearrange_rank:
        record_item = (old_dup_experts_id, dup_experts_id, rearrange_workload, expert_workload_val)
        update_experts_lists.append(record_item)

        bucket_old = old_dup_experts_id // (num_local_experts + max_dup_experts_num)
        bucket_new = dup_experts_id // (num_local_experts + max_dup_experts_num)

        input_splits[bucket_old] -= rearrange_workload
        input_splits[bucket_new] += rearrange_workload

    return num_global_tokens_per_local_expert_np, input_splits, dup_workload, group_workload_np, update_experts_lists


SIG_REARRANGE_TOKENS = nb_types.Tuple([
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens...
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits
    nb_types.Array(nb_types.int64, 2, 'C'), # group_workload_np
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4))   # update_experts_lists
])(
    nb_types.int64,       # num_local_experts
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits
    nb_types.int64,       # ep_rank
    nb_types.int64,       # max_dup_experts_num
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens...
    nb_types.int64,       # dup_workload
    nb_types.Array(nb_types.int64, 2, 'C'), # group_workload_np
    nb_types.int64,       # min_workload_ep_rank
    nb_types.int64,       # dup_experts_source_id
    nb_types.int64,       # dup_experts_id
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4)) # update_experts_lists
)
@njit(SIG_REARRANGE_TOKENS, nopython=True, cache=True)
def rearrange_select_expert_ids_np(
    num_local_experts,
    input_splits,
    ep_rank,
    max_dup_experts_num,
    num_global_tokens_per_local_expert_np,
    dup_workload,
    group_workload_np,
    min_workload_ep_rank,
    dup_experts_source_id,
    dup_experts_id,
    update_experts_lists
):
    num_global_tokens_per_local_expert_np, input_splits, dup_workload, group_workload_np, update_experts_lists = rearrange_rank_select_expert_ids_np(
        num_local_experts=num_local_experts,
        input_splits=input_splits,
        ep_rank=ep_rank,
        max_dup_experts_num=max_dup_experts_num,
        num_global_tokens_per_local_expert_np=num_global_tokens_per_local_expert_np,
        dup_workload=dup_workload,
        group_workload_np=group_workload_np,
        rearrange_rank=min_workload_ep_rank,
        min_workload_ep_rank=min_workload_ep_rank,
        dup_experts_source_id=dup_experts_source_id,
        dup_experts_id=dup_experts_id,
        update_experts_lists=update_experts_lists
    )

    col_data = group_workload_np[:, dup_experts_source_id]
    indices = np.argsort(col_data)[::-1] # Descending order indices

    for rank in indices:
        if dup_workload <= 0:
            break
        num_global_tokens_per_local_expert_np, input_splits, dup_workload, group_workload_np, update_experts_lists = rearrange_rank_select_expert_ids_np(
            num_local_experts=num_local_experts,
            input_splits=input_splits,
            ep_rank=ep_rank,
            max_dup_experts_num=max_dup_experts_num,
            num_global_tokens_per_local_expert_np=num_global_tokens_per_local_expert_np,
            dup_workload=dup_workload,
            group_workload_np=group_workload_np,
            rearrange_rank=int(rank),
            min_workload_ep_rank=min_workload_ep_rank,
            dup_experts_source_id=dup_experts_source_id,
            dup_experts_id=dup_experts_id,
            update_experts_lists=update_experts_lists
        )

    return num_global_tokens_per_local_expert_np, input_splits, group_workload_np, update_experts_lists

SIG_GREEDY_PLAN = nb_types.Tuple([
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens_per_local_expert_np
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4)),    # update_experts_lists
    nb_types.Array(nb_types.int64, 2, 'C'), # dup_experts_workload (original list of lists -> 2D array)
    nb_types.Array(nb_types.int64, 2, 'C'), # dup_experts_map (original list of lists -> 2D array)
])(
    nb_types.Array(nb_types.int64, 1, 'C'), # np_workload_per_rank
    nb_types.Array(nb_types.int64, 1, 'C'), # rank_workloads
    nb_types.Array(nb_types.int64, 2, 'C'), # workload_per_rank_per_expert
    nb_types.Array(nb_types.int64, 2, 'C'), # group_workload
    nb_types.int64,                         # target_workload_per_rank
    nb_types.int64,                         # ep_size
    nb_types.int64,                         # ep_rank
    nb_types.int64,                         # max_dup_experts_num
    nb_types.Array(nb_types.int64, 2, 'C'), # dup_experts_map (must pass 2D Array)
    nb_types.Array(nb_types.int64, 2, 'C'), # dup_experts_workload (must pass 2D Array)
    nb_types.int64,                         # num_local_experts
    nb_types.Array(nb_types.int64, 2, 'C'), # num_global_tokens_per_local_expert_np
    nb_types.Array(nb_types.int64, 1, 'C'), # input_splits
    nb_types.ListType(nb_types.UniTuple(nb_types.int64, 4)) # update_experts_lists
)
@njit(SIG_GREEDY_PLAN, nopython=True, cache=True)
def greedy_plan(
    np_workload_per_rank,
    rank_workloads,
    workload_per_rank_per_expert,
    group_workload,
    target_workload_per_rank,
    ep_size,
    ep_rank,
    max_dup_experts_num,
    dup_experts_map,
    dup_experts_workload,
    num_local_experts,
    num_global_tokens_per_local_expert_np,
    input_splits,
    update_experts_lists,
):
    _np_workload_per_rank = np_workload_per_rank

    while True:
        max_w = np.max(_np_workload_per_rank)
        min_w = np.min(_np_workload_per_rank)
        if max_w - min_w <= 1:
            break

        rank_id = int(np.argmax(rank_workloads))
        max_rank_workload = rank_workloads[rank_id]

        expert_id = int(np.argmax(workload_per_rank_per_expert[rank_id]))
        expert_workload_val = int(workload_per_rank_per_expert[rank_id, expert_id])

        min_workload_ep_rank, success = select_min_workload_rank_numba(
            ep_size=ep_size,
            max_dup_num=max_dup_experts_num,
            workload_per_rank=_np_workload_per_rank,
            dup_experts_map=dup_experts_map,
            source_rank_id=rank_id
        )

        if not success:
            break

        current_min_w = _np_workload_per_rank[min_workload_ep_rank]

        if current_min_w >= target_workload_per_rank:
            target = (current_min_w + rank_workloads[rank_id]) // 2
            dup_expert_workload = min(
                target - current_min_w,
                expert_workload_val
            )
        else:
            dup_expert_workload = min(
                target_workload_per_rank - current_min_w,
                expert_workload_val
            )

        if dup_expert_workload <= 0:
            break

        dup_experts_source_id = expert_id + rank_id * num_local_experts
        idx = None

        for i_slot in range(max_dup_experts_num):
            if dup_experts_map[min_workload_ep_rank][i_slot] == -1:
                dup_experts_map[min_workload_ep_rank][i_slot] = dup_experts_source_id
                dup_experts_workload[min_workload_ep_rank][i_slot] = dup_expert_workload
                idx = i_slot
                break
            elif dup_experts_map[min_workload_ep_rank][i_slot] == dup_experts_source_id:
                break

        if idx is None:
            break

        workload_per_rank_per_expert[rank_id, expert_id] -= dup_expert_workload
        _np_workload_per_rank[rank_id] -= dup_expert_workload
        _np_workload_per_rank[min_workload_ep_rank] += dup_expert_workload
        rank_workloads[rank_id] -= dup_expert_workload

        dup_experts_id = (idx + num_local_experts) + (num_local_experts + max_dup_experts_num) * min_workload_ep_rank


        num_global_tokens_per_local_expert_np, input_splits, group_workload, update_experts_lists = rearrange_select_expert_ids_np(
            num_local_experts=num_local_experts,
            input_splits=input_splits,
            ep_rank=ep_rank,
            max_dup_experts_num=max_dup_experts_num,
            num_global_tokens_per_local_expert_np=num_global_tokens_per_local_expert_np,
            dup_workload=dup_expert_workload,
            group_workload_np=group_workload,
            min_workload_ep_rank=min_workload_ep_rank,
            dup_experts_source_id=dup_experts_source_id,
            dup_experts_id=dup_experts_id,
            update_experts_lists=update_experts_lists
        )

    return num_global_tokens_per_local_expert_np, input_splits, update_experts_lists, dup_experts_workload, dup_experts_map


class GreedyDupExpertsPlanner:
    def __init__(self, ep_group, num_experts=128, max_dup_experts_num=2):
        self.ep_size = None
        self.num_experts = num_experts
        self.max_dup_experts_num = max_dup_experts_num

        self.ep_size = dist.get_world_size(ep_group)
        self.num_local_experts = self.num_experts // self.ep_size
        self.ep_rank = dist.get_rank(ep_group)

        # Runtime results
        self.dup_experts_map = None
        self.dup_experts_workload = None
        self.selected_experts_with_dup = None
        self.workload_per_rank = None
        self.num_global_tokens_per_local_expert = None
        self.num_global_sum_tokens_per_local_expert = None
        self.input_splits = None
        self.output_splits = None
        self.num_local_tokens_per_expert = None

        # Cache numpy versions of workload to avoid repeated conversion
        self._np_group_workload = None
        self._np_workload_per_rank = None

    def select_dup_experts_and_rearrange_experts(self, group_workload, selected_experts):
        """
        Args:
            group_workload: [ep_size, num_experts] — Supports Tensor or Numpy, internally unified to Numpy
            selected_experts: [total_tokens] — Must be on device (as it needs modification)
        """
        if self.dup_experts_map is not None:
            return

        device = selected_experts.device
        dtype = group_workload.dtype if isinstance(group_workload, torch.Tensor) else np.array(group_workload).dtype

        if isinstance(group_workload, torch.Tensor):
            gw_np = group_workload.cpu().numpy()
        else:
            gw_np = np.asarray(group_workload)

        selected_experts_with_dup = selected_experts + selected_experts // self.num_local_experts * self.max_dup_experts_num

        per_experts_workload = gw_np.sum(axis=0)  # [num_experts]

        workload_per_rank_per_expert = per_experts_workload.reshape(self.ep_size, -1)  # [ep_size, num_local_experts]

        current_rank_row = gw_np[self.ep_rank].reshape(self.ep_size, -1)
        self.input_splits = current_rank_row.sum(axis=1) # Result is numpy array

        dup_experts_map = [[-1] * self.max_dup_experts_num for _ in range(self.ep_size)]
        dup_experts_workload = [[0] * self.max_dup_experts_num for _ in range(self.ep_size)]
        dup_experts_map = np.ascontiguousarray(np.array(dup_experts_map))
        dup_experts_workload = np.ascontiguousarray(np.array(dup_experts_workload))

        num_global_tokens_shape = (
            self.ep_size,
            (self.num_local_experts + self.max_dup_experts_num) * self.ep_size
        )
        self.num_global_tokens_per_local_expert_np = np.zeros(num_global_tokens_shape, dtype=np.int64)

        for ep_rank in range(self.ep_size):
            dup_start_idx = ep_rank * (self.num_local_experts + self.max_dup_experts_num)
            dup_end_idx = dup_start_idx + self.num_local_experts
            start_idx = ep_rank * self.num_local_experts
            end_idx = (ep_rank + 1) * self.num_local_experts
            self.num_global_tokens_per_local_expert_np[:, dup_start_idx:dup_end_idx] = gw_np[:, start_idx:end_idx]

        total_workload = int(gw_np.sum())
        target_workload_per_rank = total_workload // self.ep_size

        self._np_workload_per_rank = workload_per_rank_per_expert.sum(axis=-1) # [ep_size], numpy array
        rank_workloads = self._np_workload_per_rank.copy()

        update_experts_lists = List.empty_list(nb_types.UniTuple(nb_types.int64, 4))

        self.num_global_tokens_per_local_expert_np, self.input_splits, update_experts_lists, dup_experts_workload, dup_experts_map = greedy_plan(
            np_workload_per_rank=self._np_workload_per_rank,
            rank_workloads=rank_workloads,
            workload_per_rank_per_expert=workload_per_rank_per_expert,
            group_workload=gw_np,
            target_workload_per_rank=target_workload_per_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            max_dup_experts_num=self.max_dup_experts_num,
            dup_experts_map=dup_experts_map,
            dup_experts_workload=dup_experts_workload,
            num_local_experts=self.num_local_experts,
            num_global_tokens_per_local_expert_np=self.num_global_tokens_per_local_expert_np,
            input_splits=self.input_splits,
            update_experts_lists=update_experts_lists,
        )

        for update_experts_info in update_experts_lists:
            old_dup_experts_id, dup_experts_id, rearrange_workload, expert_workload_val = update_experts_info
            selected_experts_with_dup = update_experts(
                selected_experts_with_dup,
                int(old_dup_experts_id),
                int(dup_experts_id),
                int(rearrange_workload),
            )

        self.dup_experts_map = dup_experts_map
        self.dup_experts_workload = dup_experts_workload
        self.selected_experts_with_dup = selected_experts_with_dup # Keep on device

        dup_start_idx = self.ep_rank * (self.num_local_experts + self.max_dup_experts_num)
        dup_end_idx = (self.ep_rank + 1) * (self.num_local_experts + self.max_dup_experts_num)
        local_tokens_np = self.num_global_tokens_per_local_expert_np[:, dup_start_idx:dup_end_idx]

        self.num_global_tokens_per_local_expert = torch.from_numpy(local_tokens_np).to(device)
        self.num_global_sum_tokens_per_local_expert = torch.from_numpy(local_tokens_np.sum(axis=0)).to(device)

        self.output_splits = torch.from_numpy(local_tokens_np.sum(axis=-1)).cpu()
        self.input_splits = torch.from_numpy(self.input_splits).cpu()

        return

    def clear_record_planner_result(self):
        self.dup_experts_map = None
        self.dup_experts_workload = None
        self.selected_experts_with_dup = None
        self._np_workload_per_rank = None
        self.num_global_tokens_per_local_expert = None
        self.num_global_tokens_per_local_expert_np = None
        self.num_global_sum_tokens_per_local_expert = None
        self.input_splits = None
        self.output_splits = None
