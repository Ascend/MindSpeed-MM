# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import enum
from typing import Optional, Callable

import torch
from torch import Tensor
from megatron.core import mpu, parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_block import TransformerConfig
from megatron.core.transformer.multi_token_prediction import roll_tensor, MTPLossAutoScaler
from mindspeed_mm.mcore.process_group_configs import ProcessGroupCollection



def find_vision_id_index(
    input_ids: torch.Tensor,
    image_token_id: int,
    video_token_id: int,
):
    """
    Find the vision id index for Qwen3.5 vision model.
    """
    assert input_ids.dim() == 1, "input_ids should be flaaten"
    if input_ids.numel() == 0:
        return []

    device = input_ids.device
    dtype = input_ids.dtype
    assert dtype in [torch.int, torch.int64]

    # keep the value of image_token_id/video_token_id value, others are -1
    code = torch.where(
        (input_ids == image_token_id) | (input_ids == video_token_id),
        input_ids,
        torch.tensor(-1, device=device, dtype=dtype),
    )

    # find the change idx
    first = torch.tensor([True], device=device, dtype=torch.bool)
    change = torch.cat([first, code[1:] != code[:-1]])
    change_idx = torch.nonzero(change, as_tuple=False).flatten()

    # only keep the change of image_token_id/video_token_id
    keep = code[change_idx] > 0
    starts = change_idx[keep]

    # last change position is input_ids.numel()
    next_change = torch.cat(
        [
            change_idx[1:],
            torch.tensor([input_ids.numel()], device=device, dtype=change_idx.dtype),
        ]
    )
    ends = next_change[keep]

    vals = code[starts]
    starts_cpu = starts.tolist()
    ends_cpu = ends.tolist()
    vals_cpu = vals.tolist()
    return [(int(s), int(e), int(v)) for s, e, v in zip(starts_cpu, ends_cpu, vals_cpu)]


def reorganize_inputs(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor = None,
    pixel_values_videos: torch.Tensor = None,
    image_grid_thw: torch.Tensor = None,
    video_grid_thw: torch.Tensor = None,
    image_input_mask: torch.Tensor = None,
    video_input_mask: torch.Tensor = None,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    square_merge_size: int = 4,
):
    """
    Reorganize the inputs for Qwen3.5 vision model.
    """
    if pixel_values is None:
        if video_input_mask is None and pixel_values_videos is not None:
            video_input_mask = (input_ids == video_token_id).contiguous()
        return pixel_values_videos, video_grid_thw, video_input_mask

    if pixel_values_videos is None:
        if image_input_mask is None and pixel_values is not None:
            image_input_mask = (input_ids == image_token_id).contiguous()
        return pixel_values, image_grid_thw, image_input_mask

    image_thw_cpu = image_grid_thw.tolist()
    video_thw_cpu = video_grid_thw.tolist()
    vision_indexs = find_vision_id_index(input_ids.view(-1), image_token_id, video_token_id)
    len_split = sum([thw[0] for thw in image_thw_cpu])
    len_split += sum([thw[0] for thw in video_thw_cpu])
    assert len_split == len(vision_indexs)

    vision_values = []
    vision_grid_thw = []
    idx = 0
    video_idx = 0
    image_idx = 0
    video_seqlen = 0
    image_seqlen = 0
    while idx < len(vision_indexs):
        start, end, token_id = vision_indexs[idx]
        if token_id == image_token_id:
            seqlen = 0
            thw = image_thw_cpu[image_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == image_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values[image_seqlen : (image_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            image_idx += 1
            idx += thw[0]
            image_seqlen += seqlen
        elif token_id == video_token_id:
            seqlen = 0
            thw = video_thw_cpu[video_idx]
            for i in range(thw[0]):
                start, end, token_id = vision_indexs[idx + i]
                assert token_id == video_token_id
                seqlen += (end - start) * square_merge_size
            assert seqlen == thw[0] * thw[1] * thw[2]
            vision_values.append(pixel_values_videos[video_seqlen : (video_seqlen + seqlen)])
            vision_grid_thw.append(thw)

            video_idx += 1
            idx += thw[0]
            video_seqlen += seqlen
        else:
            raise AssertionError(f"should not have {token_id=}")

    if video_input_mask is None:
        video_input_mask = input_ids == video_token_id

    if image_input_mask is None:
        image_input_mask = input_ids == image_token_id

    vision_values = torch.cat(vision_values)
    vision_grid_thw = torch.tensor(vision_grid_thw, device=image_grid_thw.device, dtype=image_grid_thw.dtype)
    vision_input_mask = video_input_mask | image_input_mask

    return vision_values, vision_grid_thw, vision_input_mask


# reference: megatron/training/utils.py get_batch_on_this_cp_rank
def split_data_cp_rank(val: torch.Tensor, cp_size: int, seq_dim: int, cp_rank: int = None):
    """
    Split the data by CP rank for Qwen3.5 vision model, using zigzag pattern.
    """
    assert cp_size > 1
    assert 0 == val.shape[seq_dim] % (2 * cp_size), f"{val.shape=} {cp_size=}"
    assert cp_rank is not None
    if val is None:
        return val

    val = val.view(
        *val.shape[0:seq_dim],
        2 * cp_size,
        val.shape[seq_dim] // (2 * cp_size),
        *val.shape[(seq_dim + 1) :],
    )

    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
    val = val.index_select(seq_dim, index)
    val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])

    return val


def expand_thw(thw: torch.Tensor) -> torch.Tensor:
    """
    Expand the THW for Qwen3.5 vision model.
    """
    assert thw.dim() == 2
    repeats = thw[:, 0].to(torch.long)
    assert torch.all(repeats > 0), "thw[:,0] must be > 0"

    idx = torch.arange(thw.size(0), device=thw.device).repeat_interleave(repeats)
    out = thw[idx].clone()
    out[:, 0] = 1
    return out


def collapse_thw(expanded: torch.Tensor) -> torch.Tensor:
    """
    Collapse the THW for Qwen3.5 vision model.
    """
    assert expanded.dim() == 2
    assert expanded.size(1) >= 2
    if expanded.shape[0] < 2:
        return expanded

    # find the diff
    other = expanded[:, 1:]
    prev = torch.cat([other[:1], other[:-1]], dim=0)
    change = (other != prev).any(dim=1)
    # the index0 must be now row
    change[0] = True

    # find the diff
    starts = torch.nonzero(change, as_tuple=False).squeeze(1)
    ends = torch.cat([starts[1:], torch.tensor([other.size(0)], device=other.device)]) - 1
    counts = ends - starts + 1

    rows_other = other[starts]
    result_first_col = counts.to(expanded.dtype).unsqueeze(1)
    result = torch.cat([result_first_col, rows_other], dim=1)
    return result


def pad_and_split(
    cp_size: int,
    hw_factor: int,
    pixel_values: list[torch.Tensor],
    image_grid_thws: list[torch.Tensor],
):
    """
    Split the pixel values and image grid thws for Qwen3.5 vision model.
    """
    assert len(pixel_values) == len(image_grid_thws)
    # split the pixel_values
    split_pixel_values = []
    split_image_grid_thws = []
    for pixel_value, image_grid_thw in zip(pixel_values, image_grid_thws):
        split_image_grid_thw = list(torch.split(image_grid_thw, 1, dim=0))
        split_image_grid_thws.extend(split_image_grid_thw)
        slice_begin = 0
        for ele in split_image_grid_thw:
            slice_end = slice_begin + ele.prod().item()
            split_pixel_values.append(pixel_value[slice_begin:slice_end].clone())
            slice_begin = slice_end

    pixel_values = split_pixel_values
    image_grid_thws = split_image_grid_thws
    img_num = len(image_grid_thws)

    img_num_per_rank = img_num // cp_size
    img_num_remain = img_num % cp_size
    cp_img_num = []
    for i in range(cp_size):
        cp_img_num.append(img_num_per_rank)
        if i < img_num_remain:
            cp_img_num[i] += 1

    img_idx = 0
    new_pixel_values = []
    new_image_grid_thws = []
    images_padded = []
    for i in range(cp_size):
        seq_len = 0
        img_begin_idx = img_idx
        img_end_idx = img_begin_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        for j in range(img_begin_idx, img_end_idx):
            seq_len += pixel_values[j].size(0)
            new_pixel_values.append(pixel_values[j])
            new_image_grid_thws.append(image_grid_thws[j])

        image_padded = 0 != seq_len % hw_factor
        if image_padded:
            padded_seqlen = (seq_len + hw_factor - 1) // hw_factor * hw_factor - seq_len
            assert padded_seqlen > 0 and padded_seqlen % 4 == 0
            new_pixel_values.append(
                torch.zeros(
                    [padded_seqlen, pixel_values[0].size(-1)],
                    dtype=pixel_values[0].dtype,
                    device=pixel_values[0].device,
                )
            )
            new_image_grid_thws.append(
                torch.tensor(
                    [[1, 2, padded_seqlen // 2]],
                    dtype=image_grid_thws[0].dtype,
                    device=image_grid_thws[0].device,
                )
            )
            cp_img_num[i] += 1
        images_padded.append(int(image_padded))

    return new_pixel_values, new_image_grid_thws, cp_img_num, images_padded


@torch.no_grad
def cp_split(
    cp_size: int,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
):
    """
    Split the pixel values and image grid thws for Qwen3.5 vision model.
    """
    assert cp_size > 1
    if pixel_values is None:
        assert image_grid_thw is None
        return None, None, None, None

    assert not pixel_values.requires_grad
    assert not image_grid_thw.requires_grad
    # expand video thw
    image_grid_thw = expand_thw(image_grid_thw)

    hw_factor = 4
    new_pixel_values, new_image_grid_thws, cp_img_num, images_padded = pad_and_split(
        cp_size,
        hw_factor,
        [pixel_values],
        [image_grid_thw],
    )
    for image_padded in images_padded:
        assert not image_padded, "qwen3.5 vit not support sp now, no need to paded"

    pixel_values = torch.cat(new_pixel_values, dim=0)
    image_grid_thw = torch.cat(new_image_grid_thws, dim=0)
    return pixel_values, image_grid_thw, cp_img_num, images_padded


def get_vision_cp_data(
    vision_data: torch.Tensor,
    vision_grid_thw: torch.Tensor,
    square_merge_size: int,
    cp_img_num: list[int],
    images_padded: list[bool],
    cp_rank: int,
    cp_size: int,
):
    """Get vision data and grid_thw for context parallelism.
    Returns:
        vision_data (torch.Tensor): Vision data of shape [total_thw_size, n_features].
        vision_grid_thw (torch.Tensor): Vision grid_thw of shape [total_thw_size, 3].
        seqlens_list (list of torch.Tensor): List of seqlens of the vision data in each context parallel rank,
                                             for the all gather after vision encoder.
    """
    # we use the context parallelism size and context parallel group of LLM for vision model.
    # we only divide the number of images in each context parallel rank.
    assert cp_size == len(cp_img_num)

    seqlens = torch.repeat_interleave(vision_grid_thw[:, 1] * vision_grid_thw[:, 2], vision_grid_thw[:, 0])
    vision_grid_thw_list = []
    vision_data_list = []
    seqlens_list = []
    img_idx = 0
    for i in range(cp_size):
        start_idx = img_idx
        end_idx = start_idx + cp_img_num[i]
        img_idx += cp_img_num[i]

        vision_grid_thw_list.append(vision_grid_thw[start_idx:end_idx])
        if images_padded[i]:
            seqlens_list.append(seqlens[start_idx : end_idx - 1])
        else:
            seqlens_list.append(seqlens[start_idx:end_idx])
        data_start_idx = seqlens[:start_idx].sum()
        data_end_idx = seqlens[:end_idx].sum()
        vision_data_list.append(vision_data[data_start_idx:data_end_idx])
    new_vision_grid_thw = vision_grid_thw_list[cp_rank]
    new_vision_data = vision_data_list[cp_rank]
    new_seqlens_list = [t // square_merge_size for t in seqlens_list]
    return new_vision_data, new_vision_grid_thw, new_seqlens_list


class AllGatherVisionEmbeddings(torch.autograd.Function):
    """
    AllGatherVisionEmbeddings for Qwen3.5 vision model.
    """

    @staticmethod
    def forward(ctx, input, seqlens_on_cp_ranks, cp_group: torch.distributed.ProcessGroup):
        """
        Forward pass for AllGatherVisionEmbeddings.
        """
        outputs = []
        for i in range(len(seqlens_on_cp_ranks)):
            o = torch.zeros(
                (seqlens_on_cp_ranks[i].sum(), *input.shape[1:]),
                device=input.device,
                dtype=input.dtype,
                layout=input.layout,
            )
            outputs.append(o)
        torch.distributed.all_gather(outputs, input, group=cp_group)
        ctx.cp_rank = torch.distributed.get_rank(group=cp_group)
        ctx.save_for_backward(*seqlens_on_cp_ranks)

        output = torch.cat(outputs, dim=0)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for AllGatherVisionEmbeddings.
        """
        cp_rank = ctx.cp_rank
        seqlens_on_cp_ranks = ctx.saved_tensors
        start_idx = torch.cat(seqlens_on_cp_ranks[:cp_rank]).sum() if cp_rank != 0 else 0
        end_idx = start_idx + seqlens_on_cp_ranks[cp_rank].sum()
        grad_output = grad_output[start_idx:end_idx]
        return grad_output, None, None


def preprocess_packed_seqs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pre_process: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> tuple[torch.Tensor, PackedSeqParams]:
    """
    Preprocess packed sequences
    CP splits sequence into CP*2 chunks, and each GPU gets 2 chunks (GPU0 gets first and last chunks, GPU1
    gets second and second last chunks, and so on), this is for load balancing with causal masking.
    See https://github.com/NVIDIA/TransformerEngine/issues/1368
    """
    batch_size = input_ids.shape[0]

    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    if pg_collection is not None:
        tp_size = pg_collection.tp.size()
        cp_size = pg_collection.cp.size()
        cp_rank = pg_collection.cp.rank()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        cp_size = mpu.get_context_parallel_world_size()
        cp_rank = mpu.get_context_parallel_rank()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_size = (align_size - seqlens_in_batch % align_size) % align_size
    seqlens_in_batch_padded = seqlens_in_batch + pad_size

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
    cu_seqlens_padded = torch.zeros(batch_size + 1, dtype=torch.int32, device=input_ids.device)
    cu_seqlens_padded[1:] = torch.cumsum(seqlens_in_batch_padded, dim=0)

    # ----------------------------------------------------------------------------
    # Move the index information needed in the subsequent loop to the CPU at once,
    # to avoid frequent .item() calls in the loop that cause D2H synchronization
    # ----------------------------------------------------------------------------
    seqlens_in_batch_cpu: list[int] = seqlens_in_batch.tolist()  # original valid lengths
    seqlens_in_batch_padded_cpu: list[int] = seqlens_in_batch_padded.tolist()  # lengths after padding
    cu_seqlens_padded_cpu: list[int] = cu_seqlens_padded.tolist()  # start positions (after padding)

    # Pure Python int calculation to avoid further synchronization
    max_seqlen_in_batch = max(seqlens_in_batch_padded_cpu)

    shape = list(input_ids.shape[1:])
    shape[0] = sum(seqlens_in_batch_padded_cpu) // cp_size
    if pre_process:
        input_ids_rmpad = torch.zeros(shape, dtype=input_ids.dtype, device=input_ids.device)
        for i in range(batch_size):
            # Use Python int, so no GPU→CPU sync in the loop
            if cp_size <= 1:
                seqlen = seqlens_in_batch_cpu[i]
                start_idx = cu_seqlens_padded_cpu[i]
                input_ids_rmpad[start_idx : start_idx + seqlen] = input_ids[i, attention_mask[i]]
                continue

            seqlen_padded_i = seqlens_in_batch_padded_cpu[i]
            seqlen = seqlen_padded_i // cp_size
            half_seqlen = seqlen // 2
            start_idx = cu_seqlens_padded_cpu[i] // cp_size
            # split to 2 chunks
            d = input_ids[i, attention_mask[i]]
            input_ids_rmpad[start_idx : start_idx + half_seqlen] = d[
                half_seqlen * cp_rank : half_seqlen * (cp_rank + 1)
            ]

            remain_start = seqlen_padded_i - half_seqlen * (cp_rank + 1)
            remain_end = seqlen_padded_i - half_seqlen * cp_rank
            remain_end = min(remain_end, d.shape[0])
            remain_len = remain_end - remain_start
            if remain_len > 0:
                input_ids_rmpad[start_idx + half_seqlen : start_idx + half_seqlen + remain_len] = d[
                    remain_start:remain_end
                ]

    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        max_seqlen_q=max_seqlen_in_batch,
        cu_seqlens_kv=cu_seqlens_padded,
        max_seqlen_kv=max_seqlen_in_batch,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
    )

    if pre_process:
        return input_ids_rmpad.unsqueeze(0), packed_seq_params
    else:
        return input_ids, packed_seq_params


class CudaGraphScope(enum.Enum):
    """Cuda Graph Scope - defines which parts of the model to capture."""

    full_iteration = 1  # Captures the entire training iteration
    attn = 2  # Captures attention layers
    mlp = 3  # Captures MLP layers (dense layers only)
    moe = 4  # Captures MoE layers (drop-and-pad MoE layers only)
    moe_router = 5  # Captures MoE router part
    moe_preprocess = 6  # Captures MoE preprocessing part (requires moe_router)
    mamba = 7  # Captures Mamba layers
    full_iteration_inference = 8  # Captures the entire inference iteration


def is_using_quantization_scales(config):
    """Returns whether the model is using quantization scales based on the config."""
    return getattr(config, "fp8", False) or getattr(config, "fp4", False)


class MTPLossLoggingHelper:
    """Helper class for logging MTP losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: Optional[torch.distributed.ProcessGroup] = None,
        avg_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        """Save the mtp loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip mtp loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=torch.cuda.current_device())
        tracker["values"][layer_number] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        """Clear the mtp losses."""
        tracker = MTPLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        """Collect and reduce the mtp losses across ranks."""
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Reduce mtp losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(
                values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG
            )

    def track_mtp_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the Multi-Token Prediction (MTP) metrics for logging."""
        MTPLossLoggingHelper.reduce_loss_in_tracker()
        tracker = MTPLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        mtp_losses = tracker["values"] * loss_scale
        mtp_num_layers = mtp_losses.shape[0]
        for i in range(mtp_num_layers):
            name = f"mtp_{i + 1} loss"
            loss = mtp_losses[i]
            if total_loss_dict is not None:
                if name in total_loss_dict:
                    total_loss_dict[name] += loss
                else:
                    total_loss_dict[name] = loss
            if writer is not None:
                writer.add_scalar(name, loss, iteration)
            if wandb_writer is not None:
                wandb_writer.log({f"{name}": loss}, iteration)

        MTPLossLoggingHelper.clean_loss_in_tracker()


def process_mtp_loss(
    hidden_states: Tensor,
    labels: Tensor,
    loss_mask: Optional[Tensor],
    output_layer: Callable,
    output_weight: Optional[Tensor],
    runtime_gather_output: Optional[bool],
    is_training: bool,
    compute_language_model_loss: Callable,
    config: TransformerConfig,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    packed_seq_params: Optional[PackedSeqParams] = None,
    scale_logits_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Process Multi-Token Prediction (MTP) loss computation.

    This is a standalone function that handles MTP loss computation. It's used on the
    post_process rank to split concatenated hidden states and compute MTP losses.

    Args:
        hidden_states (Tensor): Hidden states tensor (concatenated with MTP outputs).
        labels (Tensor): Ground truth labels.
        loss_mask (Optional[Tensor]): Mask for loss computation. If None, uses all ones.
        output_layer (Callable): Output layer method to compute logits.
        output_weight (Optional[Tensor]): Optional output weight for shared embeddings.
        runtime_gather_output (Optional[bool]): Whether to gather output at runtime.
        is_training (bool): Whether the model is in training mode.
        compute_language_model_loss (Callable): Method to compute language model loss.
        config (TransformerConfig): Model configuration containing mtp_num_layers etc.
        cp_group (Optional[ProcessGroup]): Context parallelism process group.
        packed_seq_params (Optional[PackedSeqParams]): Packed sequence parameters.
        scale_logits_fn (Optional[Callable[[Tensor], Tensor]]): Optional function to
            scale logits before loss computation (e.g., MuP output scaling).

    Returns:
        Tensor: Updated hidden states after MTP loss processing (first chunk only).
    """
    hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
    hidden_states = hidden_states_list[0]

    if labels is None:
        return hidden_states

    mtp_labels = labels.clone()
    if loss_mask is None:
        loss_mask = torch.ones_like(mtp_labels)

    # Store the original number of tokens before rolling for proper normalization
    # when calculate_per_token_loss is enabled. This ensures MTP gradients are
    # correctly scaled relative to the main loss gradients in finalize_model_grads.
    original_num_tokens = loss_mask.sum()

    for mtp_layer_number in range(config.mtp_num_layers):
        mtp_logits, _ = output_layer(
            hidden_states_list[mtp_layer_number + 1],
            weight=output_weight,
            runtime_gather_output=runtime_gather_output,
        )
        if scale_logits_fn is not None:
            mtp_logits = scale_logits_fn(mtp_logits)
        mtp_labels, _ = roll_tensor(
            mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        loss_mask, num_tokens = roll_tensor(
            loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
        )
        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
        mtp_loss = loss_mask * mtp_loss
        if is_training:
            mtp_loss_for_log = (
                torch.sum(mtp_loss) / num_tokens if num_tokens > 0 else mtp_loss.new_tensor(0.0)
            )
            MTPLossLoggingHelper.save_loss_to_tracker(
                mtp_loss_for_log,
                mtp_layer_number,
                config.mtp_num_layers,
                avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
            )
        mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
        if config.calculate_per_token_loss:
            # When calculate_per_token_loss is enabled, finalize_model_grads will
            # divide all gradients by total_num_tokens (from main loss).
            # However, MTP has fewer valid tokens due to rolling. To ensure correct
            # per-token gradient weighting, we normalize by the rolled token count
            # and re-scale by the original token count.
            # Avoid division by zero
            num_tokens_safe = torch.clamp(num_tokens, min=1)
            mtp_loss_normalized = (
                mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe)
            )
            hidden_states = MTPLossAutoScaler.apply(hidden_states, mtp_loss_normalized)
        else:
            safe_num_tokens = num_tokens.clamp(min=1)
            hidden_states = MTPLossAutoScaler.apply(
                hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
            )

    return hidden_states
