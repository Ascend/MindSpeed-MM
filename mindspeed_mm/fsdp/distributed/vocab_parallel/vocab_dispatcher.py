from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist

from . import vocab_stream
from .comm import all_to_all_single, differentiable_all_to_all


def get_num_tokens_per_owner(owners: torch.Tensor, vp_size: int) -> torch.Tensor:
    """Count how many ids this rank sends to each vocab owner."""
    return torch.histc(owners, bins=vp_size, min=0, max=vp_size).to(dtype=torch.int64)


def dispatch_preprocess(
    owners: torch.Tensor,
    vp_size: int,
    vocab_group,
) -> Tuple[List[int], List[int]]:
    """Return the (input_splits, output_splits) of both all-to-alls."""
    counts = get_num_tokens_per_owner(owners, vp_size)
    # Even all-to-all of the counts: rank j receives the column of what every rank sends to it.
    return counts.tolist(), all_to_all_single(counts, vocab_group).tolist()


class _ExpandRows(torch.autograd.Function):
    """Expand deduplicated rows back to one row per lookup.

    Its backward is the aggregation the owners need: duplicates of one row are summed here, on the
    requesting side, so the reverse all-to-all and the row ledger carry distinct rows only. The sum runs
    in FP32 even when the rows are BF16, which is where a sparse gradient loses precision first.
    """

    @staticmethod
    def forward(ctx, rows: torch.Tensor, inverse: torch.Tensor, num_unique: int):
        ctx.save_for_backward(inverse)
        ctx.row_dim = rows.shape[-1]
        ctx.num_unique = num_unique
        return rows.index_select(0, inverse)

    @staticmethod
    def backward(ctx, grad_output):
        (inverse,) = ctx.saved_tensors
        merged = torch.zeros((ctx.num_unique, ctx.row_dim), dtype=torch.float32,
                             device=grad_output.device)
        merged.index_add_(0, inverse, grad_output.reshape(-1, ctx.row_dim).float())
        return merged.to(grad_output.dtype), None, None


@dataclass
class PendingLookup:
    """A lookup whose rows have been requested but not yet handed to the layer that consumes them.

    Produced by :func:`vocab_alltoall_prepare` ahead of the decoder layers and completed by
    :func:`vocab_alltoall_finish` inside the layer, so the host-to-device copy has the layers in
    between to run under (see ``VocabPlanConfig.prefetch_split``).
    """

    rows: torch.Tensor                      # owner-local rows, copy possibly still in flight
    ready: object                           # event from the side stream, None when the copy blocked
    group: object
    input_splits: List[int]                 # what this rank sends back
    output_splits: List[int]                # what this rank receives back
    inverse_permutation: torch.Tensor
    inverse: torch.Tensor | None            # lookup -> unique row; None when dedup is off
    num_unique: int
    shape: Tuple[int, ...]                  # ngram id shape, for the final view
    batch_size: int
    output_dtype: object
    embedding_dim: int
    gradient_divide_factor: float           # divides the row gradient in the return all-to-all
    dispatcher: str                         # which backend completes it


def vocab_alltoall_prepare(module: torch.nn.Embedding, ngram_ids: torch.Tensor, vocab_group, plan,
                           async_copy: bool = True) -> PendingLookup:
    """Route the ids to their owners and start fetching the rows.

    Stops after the host-to-device copy, which is what lets the copy overlap the layers in between
    when ``async_copy`` is set. Without it this is just the first half of
    :func:`vocab_alltoall_forward`.
    """
    vp_size = 1 if vocab_group is None else dist.get_world_size(vocab_group)
    if module.num_embeddings % vp_size != 0:
        raise ValueError(
            f"num_embeddings ({module.num_embeddings}) must be divisible by vocab parallel size ({vp_size})."
        )

    window = module.num_embeddings // vp_size
    flat_ids = ngram_ids.reshape(-1).long()
    # Dedup first: ids repeat across tokens, so both all-to-alls and the host fetch carry distinct
    # rows and ``finish`` expands them back (see VocabPlanConfig.dedup). int32 halves the AI-CPU traffic.
    inverse = None
    unique_ids = flat_ids
    if plan.dedup:
        dedup_ids = flat_ids
        if module.num_embeddings <= torch.iinfo(torch.int32).max:
            dedup_ids = flat_ids.to(torch.int32)
        unique_ids, inverse = torch.unique(dedup_ids, sorted=True, return_inverse=True)
        unique_ids = unique_ids.to(flat_ids.dtype)
    owners = torch.div(unique_ids, window, rounding_mode="floor").to(torch.int32)

    input_splits, output_splits = dispatch_preprocess(owners, vp_size, vocab_group)

    # Integer argsort falls back to AI CPU on NPU, so sort float32 keys (exact while vp_size <= 2**24);
    # any order inside one owner's block is valid, and the inverse permutation restores input order.
    sort_keys = owners
    if owners.device.type == "npu" and vp_size <= 2 ** 24:
        sort_keys = owners.to(torch.float32)
    permutation = torch.argsort(sort_keys)
    sorted_local_ids = unique_ids[permutation] - owners[permutation].long() * window

    received_ids = all_to_all_single(sorted_local_ids, vocab_group, input_splits, output_splits)

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(permutation.numel(), device=permutation.device)

    rows, event = module._vp_fetcher.fetch(module, received_ids, plan, async_copy=async_copy)
    return PendingLookup(
        rows=rows,
        ready=event,
        group=vocab_group,
        input_splits=input_splits,
        output_splits=output_splits,
        inverse_permutation=inverse_permutation,
        inverse=inverse,
        num_unique=int(unique_ids.numel()),
        shape=tuple(ngram_ids.shape),
        batch_size=int(ngram_ids.shape[0]),
        output_dtype=getattr(module, "_vp_output_dtype", None),
        gradient_divide_factor=float(getattr(module, "_vp_gradient_divide_factor", 1.0)),
        embedding_dim=module.embedding_dim,
        dispatcher=plan.dispatcher,
    )


def vocab_alltoall_finish(module: torch.nn.Embedding, pending: PendingLookup) -> torch.Tensor:
    """Hand the fetched rows to the layer: cast, send them back, restore the input order."""
    # Waiting here is the whole point of the split: the copy was issued before the decoder layers, so
    # by now it is normally already done and the wait costs nothing.
    vocab_stream.wait(pending.ready)
    local_embeddings = pending.rows
    if pending.output_dtype is not None and local_embeddings.dtype != pending.output_dtype:
        local_embeddings = local_embeddings.to(dtype=pending.output_dtype)

    sorted_embeddings = differentiable_all_to_all(
        local_embeddings,
        pending.group,
        pending.output_splits,
        pending.input_splits,
        gradient_divide_factor=pending.gradient_divide_factor,
    )
    if pending.inverse is None:
        return sorted_embeddings[pending.inverse_permutation].view(*pending.shape, pending.embedding_dim)
    unique_rows = sorted_embeddings[pending.inverse_permutation].view(pending.num_unique,
                                                                     pending.embedding_dim)
    expanded = _ExpandRows.apply(unique_rows, pending.inverse, pending.num_unique)
    return expanded.view(*pending.shape, pending.embedding_dim)


def vocab_alltoall_forward(module: torch.nn.Embedding, ngram_ids: torch.Tensor, vocab_group, plan) -> torch.Tensor:
    """Route ids to their row owner and fetch the rows back (uneven all-to-all, MoE-dispatch style)."""
    pending = vocab_alltoall_prepare(module, ngram_ids, vocab_group, plan, async_copy=False)
    return vocab_alltoall_finish(module, pending)


# The split form of each backend: (prepare, finish). A backend that cannot be split simply does not
# register here, and VocabPlanConfig.prefetch_split has nothing to fall back to.
VOCAB_SPLIT_DISPATCHERS: Dict[str, Tuple[Callable, Callable]] = {
    "alltoall": (vocab_alltoall_prepare, vocab_alltoall_finish),
}




def vp_lookup(module, input_ids, past_key_values, vp_group, vp_plan) -> torch.Tensor:
    """Look this layer's rows up now.

    The model normally prefetches ahead and hands the rows in as an input (see
    ``vocab_parallel.vp_prefetch``); this is the path used when it does not, e.g. for a layer outside
    the prefetch scope or with prefetching disabled.
    """
    ngram_ids = module._compute_ngram_ids(input_ids, past_key_values)
    prepare, finish = get_vocab_split_dispatcher(vp_plan.dispatcher)
    pending = prepare(module.ngram_embedding, ngram_ids, vp_group, vp_plan)
    return finish(module.ngram_embedding, pending).flatten(-2)


def get_vocab_split_dispatcher(name: str) -> Tuple[Callable, Callable]:
    pair = VOCAB_SPLIT_DISPATCHERS.get(name)
    if pair is None:
        raise ValueError(f"vocab dispatcher {name} has no split form; disable prefetch_split.")
    return pair
