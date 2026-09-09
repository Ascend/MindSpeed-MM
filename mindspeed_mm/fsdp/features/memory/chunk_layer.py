import logging
from dataclasses import dataclass
from functools import wraps

import torch

from mindspeed.fsdp.utils.log import print_rank
from mindspeed_mm.fsdp.features.memory.swap_core import SwapCache


logger = logging.getLogger(__name__)


class _ChunkOutputCopy(torch.autograd.Function):
    """Assemble chunk outputs without cloning the full gradient per write."""

    @staticmethod
    def forward(ctx, output_buffer, chunk_output, start, end):
        ctx.start = int(start)
        ctx.end = int(end)
        ctx.mark_dirty(output_buffer)
        output_buffer[:, ctx.start:ctx.end, :].copy_(chunk_output)
        return output_buffer

    @staticmethod
    def backward(ctx, grad_output):
        # Writes are disjoint, so earlier writes can share the full gradient.
        # Clone only the chunk gradient because offload may resize its storage.
        grad_chunk = grad_output[:, ctx.start:ctx.end, :].clone()
        return grad_output, grad_chunk, None, None


@dataclass(frozen=True)
class ChunkLayerPlan:
    """Immutable chunk boundaries shared by one decoder-layer forward."""

    original_cu_seqlens: torch.Tensor | None
    layout: str
    batch_size: int
    sequence_length: int
    num_chunks: int
    cache_key: str
    forward_id: int
    chunk_infos: tuple[dict, ...]


def _build_chunk_bounds(
    sequence_length: int,
    num_chunks: int,
) -> tuple[tuple[int, int], ...]:
    """Split a sequence into exactly ``num_chunks`` contiguous ranges."""
    if num_chunks < 1:
        raise ValueError("chunk_layer_chunks must be greater than zero")
    if sequence_length < num_chunks:
        raise ValueError(
            f"sequence length {sequence_length} must be at least "
            f"chunk_layer_chunks={num_chunks}"
        )

    chunk_size, _ = divmod(sequence_length, num_chunks)
    return tuple(
        (
            chunk_idx * chunk_size,
            (chunk_idx + 1) * chunk_size
            if chunk_idx < num_chunks - 1
            else sequence_length,
        )
        for chunk_idx in range(num_chunks)
    )


def _build_chunk_infos(
    original_cu_seqlens: torch.Tensor | None,
    layout: str,
    batch_size: int,
    sequence_length: int,
    num_chunks: int,
) -> tuple[dict, ...]:
    """Precompute immutable per-chunk sequence metadata once per layer forward."""
    chunk_bounds = _build_chunk_bounds(sequence_length, num_chunks)
    if layout == "bsnd":
        return tuple(
            {
                "start": chunk_start,
                "end": chunk_end,
                "length": chunk_end - chunk_start,
                "doc_ranges": (),
                "doc_ids": tuple(range(batch_size)),
                "local_cu_seqlens": None,
                "max_local_seqlen": chunk_end - chunk_start,
                "carries_last_doc": chunk_idx < num_chunks - 1,
                "state_count": batch_size,
            }
            for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds)
        )

    boundaries = original_cu_seqlens.detach().cpu().tolist()
    doc_ranges = tuple(
        (int(start), int(end))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )
    chunk_infos = []
    for chunk_idx, (chunk_start, chunk_end) in enumerate(chunk_bounds):
        doc_ids = tuple(
            doc_id
            for doc_id, (doc_start, doc_end) in enumerate(doc_ranges)
            if max(doc_start, chunk_start) < min(doc_end, chunk_end)
        )
        if not doc_ids:
            raise ValueError(f"chunk {chunk_idx} overlaps no packed sequence")
        lengths = [
            min(doc_ranges[doc_id][1], chunk_end)
            - max(doc_ranges[doc_id][0], chunk_start)
            for doc_id in doc_ids
        ]
        local_cu_seqlens = torch.tensor(
            [0, *lengths],
            dtype=torch.int64,
            device=original_cu_seqlens.device,
        ).cumsum(0)
        chunk_infos.append(
            {
                "start": chunk_start,
                "end": chunk_end,
                "length": chunk_end - chunk_start,
                "doc_ranges": doc_ranges,
                "doc_ids": doc_ids,
                "local_cu_seqlens": local_cu_seqlens,
                "max_local_seqlen": max(lengths),
                "carries_last_doc": doc_ranges[doc_ids[-1]][1] > chunk_end,
                "state_count": len(doc_ids),
            }
        )
    return tuple(chunk_infos)


def get_chunk_layer_modules(model):
    """Find decoder-like modules that expose the Qwen chunk unit implementation."""
    modules = []
    for name, module in model.named_modules():
        chunks = int(getattr(module, "chunk_layer_chunks", 1))
        layer_type = getattr(module, "layer_type", None)
        layer_types = tuple(getattr(module, "chunk_layer_types", ()))
        if (
            chunks > 1
            and layer_type in layer_types
            and callable(getattr(module, "_forward_chunk_once", None))
        ):
            modules.append((name, module))
    return modules


def _make_chunk_decoder_forward(module):
    """Adapt the model-specific chunk unit to the feature wrapper interface."""
    chunk_forward_once = module._forward_chunk_once

    @wraps(chunk_forward_once)
    def chunk_decoder_forward(
        hidden_states,
        history_cache_1=None,
        history_cache_2=None,
        *,
        position_cos,
        position_sin,
        plan,
        layer_cache,
        chunk_idx,
        position_ids=None,
        cache_position=None,
        base_kwargs=None,
        past_key_values=None,
        **_unused_kwargs,
    ):
        if past_key_values is not None:
            raise ValueError("chunk_layer is a training path and cannot use past_key_values")
        return chunk_forward_once(
            hidden_states,
            (position_cos, position_sin),
            plan,
            layer_cache,
            chunk_idx,
            (history_cache_1, history_cache_2),
            position_ids,
            cache_position,
            {} if base_kwargs is None else base_kwargs,
        )

    return chunk_decoder_forward


def prepare_chunk_layer_modules(module):
    """Install the single-chunk decoder before recompute/offload wrappers are applied."""
    chunk_layer_modules = get_chunk_layer_modules(module)
    for name, module in chunk_layer_modules:
        if getattr(module, "_chunk_layer_patch_phase", None) is not None:
            raise RuntimeError(f"chunk_layer has already patched module {name}")
        print_rank(logger.info, f"Preparing chunk decoder module: {name}")
        module._chunk_layer_original_forward = module.forward
        module.forward = _make_chunk_decoder_forward(module)
        module._chunk_layer_patch_phase = "prepared"


def _make_chunk_scheduler(module, chunk_decoder_forward, swap_cache: SwapCache):
    """Build the outer loop that runs one decoder layer chunk by chunk."""

    @wraps(module._chunk_layer_original_forward)
    def chunk_layer_forward(
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del attention_mask
        if past_key_values is not None:
            raise ValueError("chunk_layer is a training path and cannot use past_key_values")

        # Keep model-specific parallel state optional when chunk_layer is disabled.
        from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state

        parallel_state = get_parallel_state()
        if parallel_state.ulysses_parallel_size > 1 or parallel_state.ring_attention_size > 1:
            raise ValueError("chunk_layer currently requires context/sequence parallel size 1")

        batch_size, seq_len, _ = hidden_states.shape
        num_chunks = int(module.chunk_layer_chunks)

        original_cu = kwargs.get("cu_seq_lens_q")
        if original_cu is None:
            original_cu = kwargs.get("cu_seqlens")
        layout = "tnd" if original_cu is not None else "bsnd"
        if layout == "tnd":
            if batch_size != 1:
                raise ValueError("packed TND chunk_layer requires micro_batch_size=1")
            if not isinstance(original_cu, torch.Tensor):
                original_cu = torch.tensor(
                    original_cu, dtype=torch.int64, device=hidden_states.device
                )
            else:
                original_cu = original_cu.to(device=hidden_states.device, dtype=torch.int64)
            if int(original_cu[-1]) != seq_len:
                raise ValueError(
                    f"last cu_seqlens value {int(original_cu[-1])} must equal sequence length {seq_len}"
                )

        forward_id = int(getattr(module, "_chunk_layer_forward_id", 0))
        module._chunk_layer_forward_id = forward_id + 1
        cache_key = f"{id(module)}:{forward_id}"
        plan = ChunkLayerPlan(
            original_cu_seqlens=original_cu,
            layout=layout,
            batch_size=batch_size,
            sequence_length=seq_len,
            num_chunks=num_chunks,
            cache_key=cache_key,
            forward_id=forward_id,
            chunk_infos=_build_chunk_infos(
                original_cu,
                layout,
                batch_size,
                seq_len,
                num_chunks,
            ),
        )
        from mindspeed_mm.fsdp.features.memory.chunk_layer_cache import (
            CacheOffloadManager,
            mark_cache_backward_exit,
            FACache,
            GDNCache,
        )

        manager = CacheOffloadManager()
        if manager.tensor_swap_cache is None:
            manager.tensor_swap_cache = swap_cache
        layer_cache = FACache(plan, module.layer_idx) if module.layer_type == "full_attention" else GDNCache(plan, module.layer_idx)
        # Always offload chunk caches between forward and backward.
        layer_cache.offload = True
        manager.register(layer_cache)
        hidden_states = mark_cache_backward_exit(hidden_states, cache_key)

        output_buffer = torch.empty_like(hidden_states)
        history_cache_1 = history_cache_2 = None
        chunk_base_kwargs = kwargs.copy()

        for chunk_idx, chunk_info in enumerate(plan.chunk_infos):
            start = chunk_info["start"]
            end = chunk_info["end"]
            hidden_chunk_view = hidden_states[:, start:end, :]
            chunk_position_embeddings = tuple(
                tensor[:, start:end, ...] for tensor in position_embeddings
            )
            chunk_position_ids = (
                position_ids[:, start:end] if position_ids is not None else None
            )
            chunk_cache_position = (
                cache_position[start:end] if cache_position is not None else None
            )

            # Give offload an owned storage that it can release after D2H.
            chunk_hidden_states = hidden_chunk_view.clone()

            chunk_output, history_cache_1, history_cache_2 = chunk_decoder_forward(
                chunk_hidden_states,
                history_cache_1,
                history_cache_2,
                position_cos=chunk_position_embeddings[0],
                position_sin=chunk_position_embeddings[1],
                plan=plan,
                layer_cache=layer_cache,
                chunk_idx=chunk_idx,
                position_ids=chunk_position_ids,
                cache_position=chunk_cache_position,
                base_kwargs=chunk_base_kwargs,
                past_key_values=None,
            )
            # Release the previous carry after this chunk has consumed it.
            cache = manager.get_cache(cache_key)
            if cache is not None:
                cache.complete_chunk(chunk_idx)
            # Assemble one output buffer without CopySlices gradient duplication.
            output_buffer = _ChunkOutputCopy.apply(
                output_buffer,
                chunk_output,
                start,
                end,
            )
            del chunk_output

        cache_keys = manager.finalize(cache_key)
        if cache_keys is not None:
            from mindspeed_mm.fsdp.features.memory.chunk_layer_cache import (
                mark_cache_backward_enter,
            )

            current_key, previous_key = cache_keys
            output_buffer = mark_cache_backward_enter(
                output_buffer,
                current_key,
                previous_key,
            )
        return output_buffer

    return chunk_layer_forward


def finalize_chunk_layer_modules(module, swap_cache: SwapCache):
    """Make chunk scheduling outermost after recompute/offload have wrapped the unit."""
    chunk_layer_modules = get_chunk_layer_modules(module)
    for name, module in chunk_layer_modules:
        if getattr(module, "_chunk_layer_patch_phase", None) != "prepared":
            raise RuntimeError(f"chunk_layer module {name} was not prepared before finalization")
        print_rank(logger.info, f"Finalizing chunk decoder module: {name}")
        module._chunk_layer_decoder_forward = module.forward
        module.forward = _make_chunk_scheduler(module, module._chunk_layer_decoder_forward, swap_cache)
        module._chunk_layer_patch_phase = "finalized"
