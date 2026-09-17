"""MagiHuman gated-MLP checkpoint layout conversions.

The upstream checkpoint stores every gated ``up_gate_proj`` in GPT-OSS style::

    [gate_0, linear_0, gate_1, linear_1, ...]

That layout forces ``swiglu7`` to materialize two stride-2 views on every
forward/recompute/backward pass.  The NPU training layout groups the same rows
into two contiguous halves instead::

    [gate_0, gate_1, ..., linear_0, linear_1, ...]

Only the output rows of gated ``up_gate_proj.weight`` change.  Parameter names,
shapes, dtypes, ``down_proj`` weights, and non-gated GELU7 layers stay intact.
The functions here are intentionally NPU-independent so runtime loading and the
HF/DCP converter share one, exactly inverse implementation.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping

import torch


_UP_GATE_KEY_RE = re.compile(r"(?:^|\.)block\.layers\.(\d+)\.mlp\.up_gate_proj\.weight$")
_FIRST_GATED_LAYER = 4
_LAST_GATED_LAYER = 39
_FIRST_MULTIMODAL_GATED_LAYER = 36
_NUM_MODALITIES = 3
EXPECTED_GATED_UP_GATE_WEIGHTS = _LAST_GATED_LAYER - _FIRST_GATED_LAYER + 1


def gated_up_gate_layer_index(key: str) -> int | None:
    """Return the gated DiT layer index for an up-gate weight key, else ``None``."""

    match = _UP_GATE_KEY_RE.search(key)
    if match is None:
        return None
    layer_idx = int(match.group(1))
    if _FIRST_GATED_LAYER <= layer_idx <= _LAST_GATED_LAYER:
        return layer_idx
    return None


def gated_layer_num_experts(layer_idx: int) -> int:
    """Return the number of contiguous modality blocks in a gated layer."""

    if not _FIRST_GATED_LAYER <= layer_idx <= _LAST_GATED_LAYER:
        raise ValueError(f"layer {layer_idx} is not a MagiHuman gated MLP layer")
    return _NUM_MODALITIES if layer_idx >= _FIRST_MULTIMODAL_GATED_LAYER else 1


def _validate_up_gate_weight(weight: torch.Tensor, num_experts: int) -> int:
    if weight.ndim < 1:
        raise ValueError("up_gate_proj weight must have at least one dimension")
    if num_experts < 1:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    divisor = 2 * num_experts
    if weight.shape[0] % divisor:
        raise ValueError(
            f"up_gate_proj dim0={weight.shape[0]} is not divisible by "
            f"2 * num_experts ({divisor})"
        )
    return weight.shape[0] // divisor


def deinterleave_up_gate_weight(weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert upstream interleaved rows to contiguous gate/linear halves.

    Multi-modality layers contain ``num_experts`` contiguous expert blocks; the
    conversion is performed independently inside every block.
    """

    intermediate = _validate_up_gate_weight(weight, num_experts)
    tail = tuple(weight.shape[1:])
    # [E, I, pair, ...] -> [E, pair, I, ...]
    return (
        weight.reshape(num_experts, intermediate, 2, *tail)
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def interleave_up_gate_weight(weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    """Convert internal contiguous halves back to the upstream HF row layout."""

    intermediate = _validate_up_gate_weight(weight, num_experts)
    tail = tuple(weight.shape[1:])
    # [E, pair, I, ...] -> [E, I, pair, ...]
    return (
        weight.reshape(num_experts, 2, intermediate, *tail)
        .transpose(1, 2)
        .reshape_as(weight)
        .contiguous()
    )


def convert_state_dict_up_gate_layout_(
    state_dict: MutableMapping[str, torch.Tensor], *, to_internal: bool
) -> int:
    """Convert all MagiHuman gated up-gate weights in ``state_dict`` in place.

    Keys may carry prefixes such as ``dit.`` or ``model.dit.``.  Returns the
    number of converted tensors and rejects malformed target tensors loudly.
    """

    converted = 0
    for key in list(state_dict):
        original = state_dict[key]
        converted_weight = convert_up_gate_checkpoint_tensor(
            key, original, to_internal=to_internal
        )
        if converted_weight is original:
            continue
        state_dict[key] = converted_weight
        converted += 1
    return converted


def convert_up_gate_checkpoint_tensor(
    key: str, weight: torch.Tensor, *, to_internal: bool
) -> torch.Tensor:
    """Convert one checkpoint tensor when ``key`` targets a gated up-gate.

    Non-target tensors are returned unchanged.  This small hook lets the
    framework's online HF loader apply the same exact conversion as the
    offline converter before it shards a full tensor into FSDP2 DTensors.
    """

    layer_idx = gated_up_gate_layer_index(key)
    if layer_idx is None:
        return weight
    convert = deinterleave_up_gate_weight if to_internal else interleave_up_gate_weight
    return convert(weight, gated_layer_num_experts(layer_idx))


@torch.no_grad()
def deinterleave_gated_mlp_weights_(dit: torch.nn.Module) -> int:
    """Convert a loaded DiT in place before FSDP2 shards it.

    The dimension predicate protects the non-gated GELU7 layers and avoids
    depending on an ``MLPConfig`` that upstream does not retain on the module.
    """

    converted = 0
    for _, module in dit.named_modules():
        up = getattr(module, "up_gate_proj", None)
        down = getattr(module, "down_proj", None)
        if up is None or down is None:
            continue
        if not hasattr(up, "out_features") or not hasattr(down, "in_features"):
            continue
        if up.out_features != 2 * down.in_features:
            continue
        up.weight.copy_(deinterleave_up_gate_weight(up.weight, getattr(up, "num_experts", 1)))
        converted += 1
    return converted
