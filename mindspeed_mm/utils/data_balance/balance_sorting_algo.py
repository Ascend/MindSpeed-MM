# Sorting algorithm for data balance
import torch
from google.protobuf.internal.wire_format import INT64_MAX


def post_global_balancing_greedy_without_pad(
        global_data_length: torch.Tensor,
        num_replicas: int,
        image_encoder_dp: int = None,
        max_batch_capacity: int = INT64_MAX
) -> list[list[tuple[int, int, torch.Tensor]]]:
    if image_encoder_dp is None:
        image_encoder_dp = num_replicas
    max_batch_capacity = max_batch_capacity * num_replicas // image_encoder_dp

    per_dp_size = len(global_data_length) // num_replicas
    num_groups = len(global_data_length) // max_batch_capacity

    sort_indice = torch.argsort(global_data_length[:, 2], descending=True)
    global_data_length = global_data_length[sort_indice]

    dp_group_total_length = torch.zeros(num_groups, dtype=torch.long, device=global_data_length.device)
    lengths_per_sequence = global_data_length[:, 2] ** 2
    balanced_image_dp_batch = torch.empty(
        (num_groups, max_batch_capacity, global_data_length.shape[-1]),
        dtype=global_data_length.dtype, device=global_data_length.device
    )
    balanced_image_dp_batch_idxs = [0] * num_groups
    for i, sequence_length in enumerate(global_data_length):
        target_dp_group = dp_group_total_length.argmin()
        balanced_image_dp_batch[target_dp_group, balanced_image_dp_batch_idxs[target_dp_group]] = sequence_length
        balanced_image_dp_batch_idxs[target_dp_group] += 1
        if balanced_image_dp_batch_idxs[target_dp_group] >= max_batch_capacity:
            dp_group_total_length[target_dp_group] = INT64_MAX
        else:
            dp_group_total_length[target_dp_group] += lengths_per_sequence[i]
    balanced_batchs = balanced_image_dp_batch.flatten(0, 1).split([per_dp_size] * num_replicas)

    return balanced_batchs


SORTING_ALGO_FUNC = {
    'post_global_balancing_greedy_without_pad': post_global_balancing_greedy_without_pad,
}
