import logging
import types
from functools import partial

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from mindspeed_mm.fsdp import envs
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.log import print_rank
from mindspeed_mm.fsdp.params.parallel_args import VocabPlanConfig
from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.utils.str_match import module_name_match
from mindspeed_mm.fsdp.utils.utils import is_host_resident_param

from . import vocab_stream
from .constants import HOST_RESIDENT_HINT
from .vocab_dispatcher import get_vocab_split_dispatcher, vp_lookup
from .vocab_row_fetch import DeviceRowFetcher, HostRowFetcher
from .vocab_sparse_grad import VocabRowGrad


logger = logging.getLogger(__name__)


def set_host_resident(model: torch.nn.Module, host_resident: bool, hint: str = HOST_RESIDENT_HINT) -> list:
    """Remove the n-gram embedding from ``model._no_placement_params``, or leave it declared.

    Qwen3.8 declares ``ple.ple_embedding.ngram_embedding.weight`` there, which keeps the embedding in cpu
    memory. With ``host_resident=False`` that entry is dropped, so the embedding is no longer pinned to the
    cpu and its placement follows the rest of the model; ``True`` is a no-op. Returns the entries found,
    which the caller uses to log the embedding; ``hint`` is the substring that finds them.
    """
    declared = [pattern for pattern in getattr(model, "_no_placement_params", None) or []
                if hint in pattern]
    if not host_resident and declared:
        model._no_placement_params = [pattern for pattern in model._no_placement_params
                                      if pattern not in declared]
    return declared


def get_vp_modules(model: torch.nn.Module, plan: VocabPlanConfig):
    vp_modules = []
    seen = set()
    for pattern in plan.apply_modules:
        matched = []
        for name, module in model.named_modules():
            if module_name_match(pattern, name):
                if not (callable(getattr(module, "vp_forward", None))
                        or callable(getattr(module, "_compute_ngram_ids", None))):
                    raise TypeError(
                        f"[Vocabulary Parallel] Module <{name}> must implement `_compute_ngram_ids` "
                        f"(or a full `vp_forward`)."
                    )
                matched.append((name, module))
        if not matched:
            print_rank(logger.warning, f"[Vocabulary Parallel] No module matched pattern <{pattern}>.")
        for name, module in matched:
            if id(module) in seen:
                continue
            seen.add(id(module))
            print_rank(logger.info, f"[Vocabulary Parallel]: Apply VP to module <{name}>")
            vp_modules.append(module)
    return vp_modules


def prefetched_rows(module, prefetched, input_ids):
    """The rows of a prefetched lookup when it belongs to this batch, else ``None``.

    A prefetch built for another batch (``chunk_mbs`` slices the batch inside the decoder stack) is
    not this layer's, so the caller looks its own rows up instead. Split prefetches arrive as a pending
    record and are finished here.
    """
    if prefetched is None:
        return None
    from .vocab_dispatcher import PendingLookup, get_vocab_split_dispatcher

    batch_size = (
        prefetched.batch_size if isinstance(prefetched, PendingLookup) else prefetched.shape[0]
    )
    if batch_size != input_ids.shape[0]:
        return None
    if isinstance(prefetched, PendingLookup):
        _, finish = get_vocab_split_dispatcher(prefetched.dispatcher)
        return finish(module.ngram_embedding, prefetched).flatten(-2)
    return prefetched


def get_vp_forward_fn(vp_group, vp_plan):
    """Forward for a module that only hashes, mirroring ``expert_parallel``'s experts_forward_fn.

    It also takes the prefetched rows when the model hands them in, so a model without its own
    ``vp_forward`` gets the same recomputation-saving path.
    """

    def vp_forward(self, input_ids, past_key_values=None, prefetched=None, **kwargs):
        rows = prefetched_rows(self, prefetched, input_ids)
        if rows is not None:
            return rows
        return vp_lookup(self, input_ids, past_key_values, vp_group, vp_plan)

    return vp_forward


def vp_prefetch(module, input_ids, past_key_values, vp_group, vp_plan):
    """Look this layer's rows up ahead of the layer and hand them back as a value.

    The model passes the result into the decoder layer, so it becomes one of the checkpointed
    function's inputs: the recomputation pass reuses it instead of hashing and routing the ids again,
    which keeps a non-reentrant checkpoint's saved-tensor set identical between the two passes.

    ``chunk_mbs`` is not supported: it slices the batch inside the decoder stack, where this
    ahead-of-the-stack call cannot see the micro-batches.
    """
    if not vp_plan.prefetch:
        return None
    ngram_ids = module._compute_ngram_ids(input_ids, past_key_values)
    prepare, finish = get_vocab_split_dispatcher(vp_plan.dispatcher)
    pending = prepare(module.ngram_embedding, ngram_ids, vp_group, vp_plan)
    if vp_plan.prefetch_split:
        # The layer finishes the lookup, so the copy overlaps the layers in between as before.
        return pending
    return finish(module.ngram_embedding, pending).flatten(-2)


def bind_vp_forward(module, vp_plan, vp_group, name: str) -> None:
    """Route the layer's lookup through the vocabulary-parallel forward (mirrors expert_parallel.py).

    A module may provide ``vp_forward`` itself; otherwise it only has to hash and the lookup is
    installed around ``_compute_ngram_ids``. ``vp_prefetch`` is what the model calls ahead of the
    decoder stack; its result travels into the layer as an input (see ``vp_prefetch``).
    """
    if callable(getattr(module, "vp_forward", None)):
        module.forward = partial(module.vp_forward, vp_group=vp_group, vp_plan=vp_plan)
    elif callable(getattr(module, "_compute_ngram_ids", None)):
        module.forward = types.MethodType(get_vp_forward_fn(vp_group, vp_plan), module)
    else:
        raise TypeError(
            f"[Vocabulary Parallel] module <{name}> must implement `_compute_ngram_ids` (or a full "
            f"`vp_forward`) so the lookup knows which rows to fetch."
        )
    # module goes in positionally: a keyword binding would collide with the call's first argument.
    module.vp_prefetch = partial(vp_prefetch, module, vp_group=vp_group, vp_plan=vp_plan)


def _setup_device_embedding(model, module, embedding, plan, fsdp_plan, vp_mesh, *, output_dtype, name,
                        weight_name, vp_rank, vp_size) -> None:
    """Keep the embedding on the accelerator, sharded over the vocabulary group.

    This is the layout of the device-resident baseline: an ordinary vocabulary-parallel parameter whose
    gradient comes back dense from the return all-to-all. Nothing accumulates rows on the cpu, so the
    sparse ledger, its cpu merge, the prefetch and the deduplication have nothing to act on and stay out
    of the way (see ``get_vocab_embeddings``, which only reports embeddings that own a ledger).
    """
    cpu_only = [flag for flag in ("dedup", "prefetch", "prefetch_split") if getattr(plan, flag)]
    if cpu_only:
        raise ValueError(
            f"[Vocabulary Parallel] vocab_plan.host_by_cpu=false keeps the embedding on the accelerator and "
            f"follows the device-resident baseline, which has no cpu-side machinery: {cpu_only} belong "
            f"to the cpu-resident path and must be off (dedup=false, prefetch=false, "
            f"prefetch_split=false)."
        )
    parallel_state = get_parallel_state()
    if embedding.num_embeddings % vp_size != 0:
        raise ValueError(
            f"num_embeddings ({embedding.num_embeddings}) must be divisible by vocab parallel size "
            f"({vp_size})."
        )

    weight = embedding.weight
    if isinstance(weight, DTensor):
        raise RuntimeError("[Vocabulary Parallel] Embedding weight is already distributed.")
    window = embedding.num_embeddings // vp_size
    # Allocate this rank's window directly: the model is built on meta, so this real allocation is what
    # the weight loader fills and what FSDP leaves alone. Zeros keeps both placements bit-identical.
    local_weight = torch.zeros(window, embedding.embedding_dim, dtype=weight.dtype,
                               device=get_device_type())
    # The placement names both dims: the window is sharded over vp and replicated over vdp, and a single
    # Shard(0) would claim a full embedding and hide the replica dim from the checkpoint.
    vocab_mesh = getattr(parallel_state, "vocab_device_mesh", None)
    if vocab_mesh is not None and vocab_mesh.ndim > 1:
        placements, mesh = [Replicate(), Shard(0)], vocab_mesh
    else:
        placements, mesh = [Shard(0)], vp_mesh
    distributed = DTensor.from_local(
        local_weight,
        mesh,
        placements,
        shape=torch.Size((embedding.num_embeddings, embedding.embedding_dim)),
        stride=torch.Size((embedding.embedding_dim, 1)),
        run_check=False,
    )
    embedding.register_parameter(
        "weight", torch.nn.Parameter(distributed, requires_grad=weight.requires_grad)
    )
    embedding._vp_vocab_start_index = vp_rank * window
    embedding._vp_vocab_end_index = (vp_rank + 1) * window
    # The division by the ranks the loss is averaged over lives in the return all-to-all, as in the
    # reference's pure-VP path (see comm.differentiable_all_to_all).
    embedding._vp_gradient_divide_factor = float(parallel_state.fully_shard_parallel_size)
    embedding._vp_output_dtype = output_dtype
    embedding._vp_fetcher = DeviceRowFetcher()

    # The embedding is an accelerator parameter now: it has to be visible to the optimizer and to FSDP,
    # while the ancestor FSDP units must not take it over (it is already sharded over the vocab group).
    set_host_resident(model, host_resident=False)
    if fsdp_plan is not None:
        patterns = list(getattr(fsdp_plan, "ignored_modules", None) or [])
        if f"{name}.ngram_embedding" not in patterns:
            patterns.append(f"{name}.ngram_embedding")
        fsdp_plan.ignored_modules = patterns
    print_rank(
        logger.info,
        f"[Vocabulary Parallel] embedding on device: <{weight_name}> sharded over the vocabulary group "
        f"({window} rows/rank, {weight.dtype}), dense row gradient, no host ledger",
    )


def _shard_embedding_weight(embedding: torch.nn.Embedding, vp_rank: int, vp_size: int) -> int:
    """Replace ``embedding.weight`` with this rank's contiguous row window; returns the window size."""
    num_embeddings = embedding.num_embeddings
    if num_embeddings % vp_size != 0:
        raise ValueError(
            f"num_embeddings ({num_embeddings}) must be divisible by vocab parallel size ({vp_size})."
        )
    window = num_embeddings // vp_size
    weight = embedding.weight
    if weight.shape[0] == window:
        return window

    if not weight.is_meta:
        raise RuntimeError(
            "[Vocabulary Parallel] expects a meta embedding weight: VP must run before "
            "to_empty_if_needed so that only this rank's window is ever allocated. "
            "Set training.init_model_with_meta_device: true; otherwise every rank "
            "materializes the whole embedding (and any already-loaded values are discarded here)."
        )
    new_weight = torch.empty(window, embedding.embedding_dim, dtype=weight.dtype, device=weight.device)

    embedding.register_parameter("weight", torch.nn.Parameter(new_weight, requires_grad=weight.requires_grad))
    return window

def vocab_parallelize_modules(
    model: torch.nn.Module,
    vp_mesh: DeviceMesh,
    plan: VocabPlanConfig,
    output_dtype: torch.dtype | None = None,
    fsdp_plan=None,
) -> torch.nn.Module:
    vp_modules = get_vp_modules(model, plan)
    if not vp_modules:
        raise ValueError(
            "[Vocabulary Parallel] vocab_plan.apply_modules matched no module: check the pattern(s) "
            f"{list(plan.apply_modules)}."
        )
    module_names = {id(module): name for name, module in model.named_modules()}

    host_embeddings, device_embeddings, host_modules = [], [], []
    # Only the backward host<->device copies use the side stream (see vocab_stream).
    vocab_stream.configure(plan.async_offload)
    vp_group = vp_mesh.get_group()
    vp_rank = torch.distributed.get_rank(vp_group)
    vp_size = torch.distributed.get_world_size(vp_group)

    for module in vp_modules:
        name = module_names.get(id(module), module.__class__.__name__)
        embedding = module.ngram_embedding
        weight_name = f"{name}.ngram_embedding.weight"
        if embedding.max_norm is not None:
            raise ValueError("[Vocabulary Parallel] nn.Embedding.max_norm is not supported.")
        if embedding.scale_grad_by_freq:
            raise ValueError("[Vocabulary Parallel] nn.Embedding.scale_grad_by_freq is not supported.")
        if embedding.sparse:
            raise ValueError(
                "[Vocabulary Parallel] nn.Embedding(sparse=True) is not used here: the row-sparse "
                "gradient is assembled by the dispatcher from the touched rows (see VocabRowGrad), "
                "which is the layout the sparse optimizer expects. Remove the flag."
            )
        if embedding.padding_idx is not None:
            raise ValueError("[Vocabulary Parallel] nn.Embedding.padding_idx is not supported.")

        if not plan.host_by_cpu:
            _setup_device_embedding(model, module, embedding, plan, fsdp_plan, vp_mesh,
                                output_dtype=output_dtype, name=name, weight_name=weight_name,
                                vp_rank=vp_rank, vp_size=vp_size)
            bind_vp_forward(module, plan, vp_group, name)
            device_embeddings.append((f"{name}.ngram_embedding", embedding))
            continue

        if not is_host_resident_param(model, weight_name):
            # Vocabulary parallelism owns this parameter's placement, so it declares the pattern here:
            # every model would otherwise have to list it (Qwen3.8 does, new models need not).
            model._no_placement_params = list(getattr(model, "_no_placement_params", None) or []) + [
                weight_name
            ]
            print_rank(
                logger.info,
                f"[Vocabulary Parallel] declaring <{weight_name}> host-resident in "
                f"model._no_placement_params",
            )

        window = _shard_embedding_weight(embedding, vp_rank, vp_size)
        embedding._vp_vocab_start_index = vp_rank * window
        embedding._vp_vocab_end_index = (vp_rank + 1) * window
        if embedding.weight.dtype != torch.float32:
            print_rank(
                logger.warning,
                f"[Vocabulary Parallel] <{weight_name}> is {embedding.weight.dtype}, not fp32: "
                f"SparseAdam keeps its moments in the embedding's dtype, so the update loses precision "
                f"(the reference keeps an FP32 master). Leave model.dtype unset or set it to fp32.",
            )
        # The embedding is not FSDP-managed, so its gradient is reduced here: the row-window replicas are
        # summed in the ledger, then divided by the number of ranks the loss is averaged over (the
        # return all-to-all, as in the reference).
        parallel_state = get_parallel_state()
        embedding._vp_gradient_divide_factor = float(parallel_state.fully_shard_parallel_size)
        embedding._vp_row_grad = VocabRowGrad(
            shape=embedding.weight.shape,
            group=parallel_state.get_vdp_group(),
            device=torch.device(get_device_type()),
            device_sort=envs.MM_VP_DEVICE_SORT,
            device_merge=envs.MM_VP_DEVICE_MERGE,
        )
        embedding._vp_fetcher = HostRowFetcher()
        # The lookup reads fp32 rows out of host memory; the layer around it runs in the mixed
        # precision dtype, which an on-device (FSDP-cast) embedding would have produced.
        embedding._vp_output_dtype = output_dtype
        bind_vp_forward(module, plan, vp_group, name)
        host_embeddings.append((f"{name}.ngram_embedding", embedding))
        host_modules.append(module)

        # The model calls ``vp_prefetch`` ahead of the PLE layer, so a recomputation reuses the rows
        # instead of looking them up again (see VocabPlanConfig.prefetch).
        # Only the backward host<->device copies use the side stream (see vocab_stream).
        transfer = "overlap the compute stream" if vocab_stream.enabled() else "block the compute stream"
        split = "split (the copy runs under the layers in between)" if plan.prefetch_split else "single call"
        dedup = "on (repeated ids collapse before routing)" if plan.dedup else "off (every lookup is routed)"
        if plan.prefetch:
            print_rank(logger.info, "[Vocabulary Parallel] prefetch on: each PLE layer's rows are looked "
                                    f"up before the decoder layers, so recomputation reuses them; "
                                    f"{split}; dedup {dedup}; row gradient transfers {transfer}")
        else:
            print_rank(logger.info, "[Vocabulary Parallel] prefetch off: each PLE layer looks its rows up "
                                    f"inside itself, so recomputation repeats the lookup; row gradient "
                                    f"transfers {transfer}; dedup {dedup}")

        print_rank(
            logger.debug,
            f"[Vocabulary Parallel]: rank={vp_rank}/{vp_size}, window={window}, "
            f"vocab range=[{embedding._vp_vocab_start_index}, {embedding._vp_vocab_end_index})",
        )

    # The per-step paths (optimizer member, clipping) read this instead of walking the module tree.
    model._vocab_embeddings = {"host": host_embeddings, "device": device_embeddings}
    return model
