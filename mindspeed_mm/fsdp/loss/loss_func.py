import torch
import torch.nn.functional as F

from mindspeed.fsdp.memory.chunk_loss.chunk_loss import chunk_loss, calculate_lm_loss, fixed_cross_entropy
from mindspeed_mm.fsdp.utils.constants import AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.communication import split_forward_gather_backward_with_cp


def build_loss_func(
    loss_type,
    ignore_index=-100,
    chunk_size=1024,
    **kwargs
):
    labels = kwargs.get('labels', None)
    if labels is None:
        raise ValueError("labels are missing.")
    bs = labels.shape[0]
    labels = F.pad(labels, (0, 1), value=ignore_index)
    # Shift labels to match the input sequence for next-token prediction.
    shift_labels = labels[..., 1:].contiguous()

    # Create a mask to identify valid tokens (typically > -1 means non-special tokens)
    loss_mask = shift_labels > -1

    # Retrieve loss_type arguments to determine loss reduction behavior.
    if loss_type == "per_sample_loss":
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
    elif loss_type == "token_loss":
        alpha = loss_mask.sum()
        torch.distributed.all_reduce(alpha, op=torch.distributed.ReduceOp.AVG)
        reduction = "none"
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
        chunk_labels = torch.split(shift_labels, chunk_size, dim=1)

        if loss_type == "square_loss":
            alpha = torch.split(alpha.view(bs, -1), chunk_size, dim=1)

        # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
        loss_func_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": ignore_index,
                "reduction": reduction,
                "alpha": alpha[i].view(-1) if isinstance(alpha, (list, tuple)) else alpha,
            }
            for i in range(len(chunk_labels))
        ]

        # Return a closure that computes the chunked language modeling loss using the prepared config.
        def loss_func(hidden_states, head_weight, head_bias):
            return chunk_loss(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward=calculate_lm_loss,
                loss_kwargs_chunks=loss_func_kwargs,
                chunk_size=chunk_size
            )

    else:
        def loss_func(logits):
            logits = logits.view(-1, logits.shape[-1])
            labels = shift_labels.view(-1)
            return fixed_cross_entropy(
                logits, labels,
                alpha=alpha,
                reduction=reduction
            )

    return loss_func, loss_mask