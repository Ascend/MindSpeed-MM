import logging

import torch
import torch.nn.functional as F

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss import chunk_loss, calculate_lm_loss, fixed_cross_entropy
from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_fused import chunk_loss_cce_fused
from mindspeed_mm.fsdp.utils.constants import AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import split_forward_gather_backward_with_cp


logger = logging.getLogger(__name__)


def calculate_chunk_size(batch_size: int, total_size: int) -> int:
    """
    Calculate dynamic Chunk Size to ensure batch_size * chunk_size ≤ total size,
    where chunk_size is the largest power of two not exceeding the theoretical maximum value.

    Args:
        batch_size (int): Input batch size

        total_size (int): Upper limit of total tokens (batch_size * chunk_size),
            typically configured as the maximum token capacity of the device (e.g., 4096/8192 tokens).

    Returns:
        int: Dynamic Chunk Size that meets the requirements, returns 1 by default (when input is invalid)
    """
    if batch_size <= 0 or total_size <= 0:
        print_rank(logger.info, f'Batch size={batch_size} or total size={total_size} must be a positive integer!')
        return 1
    if batch_size >= total_size:
        print_rank(logger.info, f'Batch size={batch_size} exceeds total size={total_size}!')
        return 1

    max_possible_chunk_size = total_size // batch_size

    if max_possible_chunk_size == 0:
        print_rank(logger.info, f'No valid Chunk Size for batch size batch_size={batch_size}!')
        return 1

    max_power_of_two_chunk_size = 1 << (max_possible_chunk_size.bit_length() - 1)

    if max_power_of_two_chunk_size > max_possible_chunk_size:
        max_power_of_two_chunk_size = max_power_of_two_chunk_size >> 1  # Right shift by 1 bit = divide by 2

    return max_power_of_two_chunk_size


def get_loss_func_params(
    labels,
    loss_type,
    ignore_index=-100,
    chunk_size=1024,
    **kwargs
):
    bs = labels.shape[0]
    total_chunk_size = kwargs.get('total_chunk_size', None)
    if total_chunk_size:
        chunk_size = calculate_chunk_size(bs, total_chunk_size)
    labels = F.pad(labels, (0, 1), value=ignore_index)
    # Shift labels to match the input sequence for next-token prediction.
    shift_labels = labels[..., 1:].contiguous()

    # Create a mask to identify valid tokens (typically > -1 means non-special tokens)
    loss_mask = shift_labels > -1

    # Retrieve loss_type arguments to determine loss reduction behavior.
    if loss_type == "per_sample_loss":
        if "cu_seqlens" in kwargs and kwargs.get("cu_seqlens", None) is not None:
            cu_seqlens = kwargs.get("cu_seqlens")
            lengths = cu_seqlens[:, 1:] - cu_seqlens[:, :-1]  # [1, num_samples]
            total_seq_len = loss_mask.size(1)
            positions = torch.arange(total_seq_len, device=loss_mask.device)[None, :]  # [1, T]
            sample_ids = (positions.unsqueeze(1) >= cu_seqlens[:, 1:].unsqueeze(2)).sum(dim=1)  # [1, T]
            valid_per_sample = torch.zeros_like(lengths, dtype=torch.int32)
            for i in range(lengths.size(1)):
                mask = (sample_ids == i)
                valid_per_sample[:, i] = loss_mask[mask].sum()
            result = torch.repeat_interleave(valid_per_sample, lengths[0], dim=1).float()  # [1, total_seq_len]
            alpha = torch.nn.functional.pad(result, (0, max(0, total_seq_len - result.size(1))), value=1 / lengths.size(1)) * lengths.size(1)

            ps = get_parallel_state()
            if ps.is_cp_enable():
                alpha = split_forward_gather_backward_with_cp(alpha, dim=1)
        else:
            # Compute per-sample loss: alpha scales each sample by total valid tokens in the batch.
            alpha = loss_mask.sum(1) * loss_mask.shape[0]  # shape: [batch_size]
        reduction = "none"  # Keep per-token losses for sample-wise aggregation.
    elif loss_type == "per_token_loss":
        # Use raw sum loss without normalization here;
        avg_per_step_token_num = kwargs.get(AVG_PER_STEP_TOKEN_NUM, None)
        if avg_per_step_token_num is None:
            raise KeyError(f"per_token_loss must use PrefetchGradAccDataLoader")
        torch.distributed.all_reduce(avg_per_step_token_num, op=torch.distributed.ReduceOp.AVG)
        alpha = avg_per_step_token_num
        reduction = "sum"
    elif loss_type == "default":
        # Default: normalize loss by total number of valid tokens in the batch.
        alpha = loss_mask.sum()  # scalar
        reduction = "sum"
    else:
        raise NotImplementedError(f"{loss_type} is not implemented!")

    ps = get_parallel_state()
    if ps.is_cp_enable():
        shift_labels = split_forward_gather_backward_with_cp(shift_labels, dim=1)

    if chunk_size:
        # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
        bs = shift_labels.shape[0]
        chunk_labels = torch.split(shift_labels, chunk_size, dim=1)

        # Each token has its own coefficient.
        if alpha.ndim >= 2 and alpha.shape[1] > 1:
            alpha = torch.split(alpha.view(bs, -1), chunk_size, dim=1)

        # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
        loss_func_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": ignore_index,
                "reduction": reduction,
                "alpha": alpha[i].view(-1) if isinstance(alpha, (list, tuple)) else alpha,
                "chunk_size": chunk_size,
            }
            for i in range(len(chunk_labels))
        ]
        return loss_func_kwargs

    loss_func_kwargs = [
        {
            "shift_labels": shift_labels,
            "ignore_index": ignore_index,
            "reduction": reduction,
            "alpha": alpha,
            "chunk_size": chunk_size,
        }
    ]

    return loss_func_kwargs


def build_loss_func(
    loss_type,
    ignore_index=-100,
    chunk_size=None,
    vocab_tile_size=None,
    **kwargs
):
    outer_labels = kwargs.get("labels", None)
    _kwargs = {}
    _kwargs[AVG_PER_STEP_TOKEN_NUM] = kwargs.get(AVG_PER_STEP_TOKEN_NUM, None)
    _kwargs['total_chunk_size'] = kwargs.get('total_chunk_size', None)
    _kwargs['cu_seqlens'] = kwargs.get('cu_seqlens', None)
    if vocab_tile_size is not None:
        # CCE path: vocab-tile streaming + 2D tile fused kernel + dual-stream pipelining
        # Supported reductions: 'default' and 'per_token_loss' (both scalar-sum + alpha
        # normalization). 'per_sample_loss' needs per-sample reduction and is not yet
        # supported (CCE's autograd Function currently returns a single scalar sum).
        if loss_type not in ("default", "per_token_loss"):
            raise NotImplementedError(
                f"CCE loss currently only supports loss_type='default'/'per_token_loss', got '{loss_type}'"
            )
        if kwargs.get('total_chunk_size', None):
            raise ValueError(
                "chunkloss_plan.impl_type='cce' is mutually exclusive with enable_dynamic_chunk_loss: "
                "please enable only one chunk loss implementation."
            )

        # CCE does token chunking via vocab tiles internally, so it must never re-enter
        # get_loss_func_params' token-chunk branch. Other loss paths (e.g. dynamic chunk
        # loss) may have injected total_chunk_size / cu_seqlens into _kwargs; strip them
        # so they cannot override the chunk_size=None passed below.
        _cce_kwargs = {k: v for k, v in _kwargs.items() if k not in ("total_chunk_size", "cu_seqlens")}

        def loss_func(hidden_states, head_weight, head_bias, labels=None):
            if head_bias is not None:
                raise NotImplementedError(f"head_bias is not supported in CCE loss")
            labels = labels if labels is not None else outer_labels
            if labels is None:
                raise ValueError("labels must be provided either in build_loss_func or in loss_func call.")
            # Reuse get_loss_func_params to get shift_labels/alpha/ignore_index
            # Pass chunk_size=None to avoid token chunking (CCE uses vocab tile internally)
            loss_func_kwargs = get_loss_func_params(
                labels, loss_type, ignore_index, chunk_size=None, **_cce_kwargs,
            )
            shift_labels = loss_func_kwargs[0]["shift_labels"]
            alpha = loss_func_kwargs[0]["alpha"]  # scalar (loss_mask.sum() or avg tokens)
            loss = chunk_loss_cce_fused(
                hidden_states, head_weight, shift_labels,
                vt=vocab_tile_size, ignore_index=ignore_index,
                seq_chunk_size=chunk_size,
            )
            # Alpha normalization (CCE forward outputs unnormalized sum loss)
            # Autograd automatically handles gradient scaling for / alpha during backward
            return loss / alpha

    elif chunk_size or kwargs.get('total_chunk_size', None):
        # Return a closure that computes the chunked language modeling loss using the prepared config.

        def loss_func(hidden_states, head_weight, head_bias, labels=None):
            labels = labels if labels is not None else outer_labels
            if labels is None:
                raise ValueError("labels must be provided either in build_loss_func or in loss_func call.")
            loss_func_kwargs = get_loss_func_params(
                labels,
                loss_type,
                ignore_index,
                chunk_size,
                **_kwargs,
            )

            return chunk_loss(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward=calculate_lm_loss,
                loss_kwargs_chunks=loss_func_kwargs,
                chunk_size=loss_func_kwargs[0]["chunk_size"],
            )

    else:
        def loss_func(logits, labels=None, vocab_size=None, **kwargs):
            labels = labels if labels is not None else outer_labels
            if labels is None:
                raise ValueError("labels must be provided either in build_loss_func or in loss_func call.")
            loss_func_kwargs = get_loss_func_params(
                labels,
                loss_type,
                ignore_index,
                chunk_size,
                **_kwargs,
            )
            shift_labels = loss_func_kwargs[0]["shift_labels"]
            reduction = loss_func_kwargs[0]["reduction"]
            alpha = loss_func_kwargs[0]["alpha"]

            logits = logits.view(-1, logits.shape[-1]).contiguous().float()
            labels = shift_labels.view(-1)
            return fixed_cross_entropy(
                logits, labels,
                ignore_index=ignore_index,
                alpha=alpha,
                reduction=reduction
            )

    return loss_func
