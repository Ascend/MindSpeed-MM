from abc import ABC, abstractmethod
import re
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig

from mindspeed_mm.fsdp.utils.constants import (
    DCP_DOWN_PROJ_SUFFIX,
    DCP_GATE_UP_PROJ_SUFFIX,
    HF_GATE_PROJ_SUFFIX,
    HF_UP_PROJ_SUFFIX,
)


def permute_moe_expert(
    key: str,
    tensor: torch.Tensor,
    patterns: List[str],
) -> torch.Tensor:
    """Permute MoE expert weights.

    - gate_up_proj: (num_experts, 2 * intermediate, hidden) <-> (num_experts, hidden, 2 * intermediate)
    - down_proj:    (num_experts, hidden, intermediate) <-> (num_experts, intermediate, hidden)

    Returns the tensor unchanged if the key does not match any pattern.
    """
    for pattern in patterns:
        if re.fullmatch(pattern, key):
            return tensor.permute(0, 2, 1).contiguous()
    return tensor


def reshape_fused_linear(
    key: str,
    tensor: torch.Tensor,
    fused_linear_names: Optional[List[str]],
) -> torch.Tensor:
    """Reshape fused-linear weights.

    (expert_num, input_dim, output_dim) -> (expert_num * input_dim, output_dim)

    No-op when ``fused_linear_names`` is None or empty.
    """
    if not fused_linear_names:
        return tensor
    if any(name in key for name in fused_linear_names):
        return tensor.view(-1, tensor.shape[-1])
    return tensor


def rename_key(key: str, hf_prefix: str, dcp_prefix: str) -> str:
    """Apply the (hf_prefix -> dcp_prefix) rename.
    """
    if len(hf_prefix) > 0:
        return key.replace(hf_prefix, dcp_prefix, 1)
    return f"{dcp_prefix}{key}"


def convert_diffusers_key(
    key: str,
    convert_mapping: Dict[str, str],
    str_replace_mapping: Dict[str, str],
) -> Optional[str]:
    """Convert one diffusers key to a native key using the supplied exact
    ``convert_mapping`` and ordered ``str_replace_mapping``.

    Returns ``None`` for keys to skip (e.g. ``_extra_state``).
    """
    # Skip diffusers internal keys
    if key.endswith("_extra_state"):
        return None

    # Exact key rename
    if key in convert_mapping:
        return convert_mapping[key]

    # Sub-string replacements (applied in order; single-character patterns are
    # skipped deliberately to avoid destructive replaces)
    for old_str, new_str in str_replace_mapping.items():
        if len(old_str) > 1:
            key = key.replace(old_str, new_str)

    return key


def merge_moe_expert_weights(
    state_dict: Dict[str, torch.Tensor],
    hf_keys: Tuple[str, ...],
    dcp_key: str,
    dcp_down_proj_suffix: str = DCP_DOWN_PROJ_SUFFIX,
    dcp_gate_up_proj_suffix: str = DCP_GATE_UP_PROJ_SUFFIX,
    hf_gate_proj_suffix: str = HF_GATE_PROJ_SUFFIX,
    hf_up_proj_suffix: str = HF_UP_PROJ_SUFFIX,
    transpose: bool = True,
) -> Optional[Tuple[str, torch.Tensor]]:
    """Merge a complete projection group from buffered MTP experts.
    """
    if not all(key in state_dict for key in hf_keys):
        return None
    if dcp_key.endswith(dcp_down_proj_suffix):
        down_proj_weights = [state_dict.pop(key) for key in hf_keys]
        if transpose:
            down_proj_weights = [weight.T for weight in down_proj_weights]
        return dcp_key, torch.stack(down_proj_weights)

    if dcp_key.endswith(dcp_gate_up_proj_suffix):
        gate_up_proj_weights = []
        # hf_keys is built as (gate, up) pairs for each expert, so even/odd slices align.
        for gate_proj, up_proj in zip(hf_keys[::2], hf_keys[1::2]):
            gate_prefix = gate_proj.removesuffix(hf_gate_proj_suffix)
            up_prefix = up_proj.removesuffix(hf_up_proj_suffix)
            if gate_prefix != up_prefix:
                raise ValueError(f"Mismatched gate/up expert keys: {gate_proj}, {up_proj}")
            weight = torch.concat(
                [state_dict.pop(gate_proj), state_dict.pop(up_proj)]
            )
            gate_up_proj_weights.append(weight.T if transpose else weight)
        return dcp_key, torch.stack(gate_up_proj_weights)

    raise ValueError(f"Unsupported DCP expert weight key: {dcp_key}")


def split_moe_expert_weights(
    tensor: torch.Tensor,
    hf_keys: Tuple[str, ...],
    dcp_key: str,
    dcp_down_proj_suffix: str = DCP_DOWN_PROJ_SUFFIX,
    dcp_gate_up_proj_suffix: str = DCP_GATE_UP_PROJ_SUFFIX,
    hf_gate_proj_suffix: str = HF_GATE_PROJ_SUFFIX,
    hf_up_proj_suffix: str = HF_UP_PROJ_SUFFIX,
    transpose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Split a merged MTP expert tensor into individual HF tensors.
    """
    result = {}
    if dcp_key.endswith(dcp_gate_up_proj_suffix):
        # hf_keys preserves the (gate, up) order created during mapping, so even/odd slices align.
        for expert_tensor, gate_key, up_key in zip(
            tensor.unbind(), hf_keys[::2], hf_keys[1::2]
        ):
            gate_prefix = gate_key.removesuffix(hf_gate_proj_suffix)
            up_prefix = up_key.removesuffix(hf_up_proj_suffix)
            if gate_prefix != up_prefix:
                raise ValueError(f"Mismatched gate/up expert keys: {gate_key}, {up_key}")
            if transpose:
                expert_tensor = expert_tensor.T
            gate_proj, up_proj = torch.chunk(expert_tensor, 2, dim=0)
            result[gate_key] = gate_proj.contiguous()
            result[up_key] = up_proj.contiguous()
    elif dcp_key.endswith(dcp_down_proj_suffix):
        for expert_tensor, hf_key in zip(tensor.unbind(), hf_keys):
            if transpose:
                expert_tensor = expert_tensor.T
            result[hf_key] = expert_tensor.contiguous()
    else:
        raise ValueError(f"Unsupported DCP expert weight key: {dcp_key}")
    return result


def build_weight_transform(model_id: str, hf_dir: str) -> Optional["WeightTransformPipeline"]:
    transform_cls = WEIGHT_TRANSFORM_PIPELINES.get(model_id)
    return transform_cls(hf_dir=hf_dir) if transform_cls is not None else None


class WeightTransformPipeline(ABC):
    """Base pipeline for weight format conversion.
    """

    def __init__(self) -> None:
        self.hf_to_dcp_mapping: Dict[Tuple[str, ...], str] = {}
        self.dcp_to_hf_mapping: Dict[str, Tuple[str, ...]] = {}

    @abstractmethod
    def hf_to_dcp(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Convert one HuggingFace tensor into a DCP tensor.

        Returns ``None`` when the input tensor has been consumed or buffered but
        does not produce an immediate DCP entry. For example, a pipeline may
        buffer several expert tensors and emit their merged tensor only after
        the complete group has arrived.
        """
        pass

    @abstractmethod
    def dcp_to_hf(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert one DCP tensor into one or more HuggingFace tensors.

        Multiple tensors are returned when a DCP tensor stores weights that
        were merged during HF-to-DCP conversion. For example, a merged MoE
        expert tensor is split back into the individual ``gate_proj``,
        ``up_proj`` and/or ``down_proj`` HF expert weights.
        """
        pass


class Qwen35WeightTransformPipeline(WeightTransformPipeline):
    """Weight transform pipeline for Qwen3.5.
    """
    def __init__(
        self,
        hf_dir: str,
        mtp_num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
        text_config = getattr(config, "text_config", config)
        self.expert_weight_patterns = [
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.down_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.down_proj",
        ]
        self._buffered_weights: Dict[str, torch.Tensor] = {}

        num_experts = getattr(text_config, "num_experts", 0)
        mtp_num_hidden_layers = getattr(text_config, "mtp_num_hidden_layers", 0)
        if mtp_num_layers is not None:
            mtp_num_hidden_layers = min(mtp_num_hidden_layers, mtp_num_layers)
        for layer in range(mtp_num_hidden_layers if num_experts > 0 else 0):
            weight_path = f"mtp.layers.{layer}.mlp.experts"
            gate_up_hf_keys = tuple(
                f"{weight_path}.{expert}.{projection}.weight"
                for expert in range(num_experts)
                for projection in ("gate_proj", "up_proj")
            )
            down_hf_keys = tuple(
                f"{weight_path}.{expert}.down_proj.weight"
                for expert in range(num_experts)
            )
            self.hf_to_dcp_mapping[gate_up_hf_keys] = f"{weight_path}.gate_up_proj"
            self.hf_to_dcp_mapping[down_hf_keys] = f"{weight_path}.down_proj"

        self.dcp_to_hf_mapping = {
            dcp_key: hf_keys
            for hf_keys, dcp_key in self.hf_to_dcp_mapping.items()
        }

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Convert a Huggingface weight to DCP format.

        Shape symbols:
            e: Expert index.
            E: Number of experts.
            H: Hidden size.
            I: Intermediate size.
        """
        # Merge per-expert MTP weights:
        # experts.{e}.{gate,up}_proj.weight [I, H] -> experts.gate_up_proj [E, H, 2I]
        # experts.{e}.down_proj.weight [H, I] -> experts.down_proj [E, I, H]
        for hf_keys, dcp_key in self.hf_to_dcp_mapping.items():
            if key not in hf_keys:
                continue
            self._buffered_weights[key] = tensor
            return merge_moe_expert_weights(
                state_dict=self._buffered_weights,
                hf_keys=hf_keys,
                dcp_key=dcp_key,
            )

        # Permute fused MoE weights:
        # experts.gate_up_proj [E, 2I, H] -> [E, H, 2I]
        # experts.down_proj [E, H, I] -> [E, I, H]
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return key, tensor

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Convert a DCP weight to Huggingface format.

        Shape symbols:
            e: Expert index.
            E: Number of experts.
            H: Hidden size.
            I: Intermediate size.
        """
        # Split merged MTP weights:
        # experts.gate_up_proj [E, H, 2I] -> experts.{e}.{gate,up}_proj.weight [I, H]
        # experts.down_proj [E, I, H] -> experts.{e}.down_proj.weight [H, I]
        for dcp_key, hf_keys in self.dcp_to_hf_mapping.items():
            if key != dcp_key:
                continue
            return split_moe_expert_weights(
                tensor=tensor,
                hf_keys=hf_keys,
                dcp_key=dcp_key,
            )

        # Permute fused MoE weights:
        # experts.gate_up_proj [E, H, 2I] -> [E, 2I, H]
        # experts.down_proj [E, I, H] -> [E, H, I]
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return {key: tensor}


class DiffusersKeyMapTransformPipeline(WeightTransformPipeline):
    """Generic diffusers -> native per-tensor transform pipeline driven by
    subclass key maps (``CONVERT_MAPPING`` / ``STR_REPLACE_MAPPING``).

    New models only ship the mapping tables and register a subclass in
    ``WEIGHT_TRANSFORM_PIPELINES`` (e.g. ``Wan22DiffusersTransformPipeline``);
    the checkpoint layer carries no model-specific knowledge. Saving back to
    diffusers format is not supported.
    """

    CONVERT_MAPPING: Dict[str, str] = {}
    STR_REPLACE_MAPPING: Dict[str, str] = {}

    def __init__(self, hf_dir: Optional[str] = None, **kwargs) -> None:
        # ``hf_dir``/``mtp_num_layers`` are accepted (and ignored) so the class
        # can be built through ``build_weight_transform`` / ``HFCheckpointer``
        # with the same call signature as config-driven pipelines.
        super().__init__()

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Optional[Tuple[str, torch.Tensor]]:
        """Rename one diffusers key to the native layout; returns ``None`` for
        dropped tensors (e.g. diffusers-internal ``_extra_state`` entries)."""
        key = convert_diffusers_key(key, self.CONVERT_MAPPING, self.STR_REPLACE_MAPPING)
        if key is None:
            return None
        return key, tensor

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError(
            "Saving weights in diffusers format is not supported."
        )


# Wan2.2 diffusers -> native WanModel key mapping tables.

# Exact key renames (old_key -> new_key)
WAN22_DIFFUSERS_CONVERT_MAPPING = {
    "condition_embedder.text_embedder.linear_1.bias": "text_embedding.0.bias",
    "condition_embedder.text_embedder.linear_1.weight": "text_embedding.0.weight",
    "condition_embedder.text_embedder.linear_2.bias": "text_embedding.2.bias",
    "condition_embedder.text_embedder.linear_2.weight": "text_embedding.2.weight",
    "condition_embedder.time_embedder.linear_1.bias": "time_embedding.0.bias",
    "condition_embedder.time_embedder.linear_1.weight": "time_embedding.0.weight",
    "condition_embedder.time_embedder.linear_2.bias": "time_embedding.2.bias",
    "condition_embedder.time_embedder.linear_2.weight": "time_embedding.2.weight",
    "condition_embedder.time_proj.bias": "time_projection.1.bias",
    "condition_embedder.time_proj.weight": "time_projection.1.weight",
    "scale_shift_table": "head.modulation",
    "proj_out.bias": "head.head.bias",
    "proj_out.weight": "head.head.weight",
}

# Sub-string replacements applied to every key
WAN22_DIFFUSERS_STR_REPLACE_MAPPING = {
    "attn1.norm_q": "self_attn.norm_q",
    "attn1.norm_k": "self_attn.norm_k",
    "attn2.norm_q": "cross_attn.norm_q",
    "attn2.norm_k": "cross_attn.norm_k",
    "attn1.to_q.": "self_attn.q.",
    "attn1.to_k.": "self_attn.k.",
    "attn1.to_v.": "self_attn.v.",
    "attn1.to_out.0.": "self_attn.o.",
    "attn2.to_q.": "cross_attn.q.",
    "attn2.to_k.": "cross_attn.k.",
    "attn2.to_v.": "cross_attn.v.",
    "attn2.to_out.0.": "cross_attn.o.",
    ".ffn.net.0.proj.": ".ffn.0.",
    ".ffn.net.2.": ".ffn.2.",
    "scale_shift_table": "modulation",
    ".norm2.": ".norm3.",
}


class Wan22DiffusersTransformPipeline(DiffusersKeyMapTransformPipeline):
    """Wan2.2 diffusers -> native transform pipeline, driven by the tables above."""

    CONVERT_MAPPING = WAN22_DIFFUSERS_CONVERT_MAPPING
    STR_REPLACE_MAPPING = WAN22_DIFFUSERS_STR_REPLACE_MAPPING


WEIGHT_TRANSFORM_PIPELINES = {
    "qwen3_5_moe": Qwen35WeightTransformPipeline,
    "wan2_2": Wan22DiffusersTransformPipeline,
}
# Note: ``DiffusersKeyMapTransformPipeline`` is a base class with empty mapping
# tables and must NOT be registered directly; new diffusers-source models add a
# concrete subclass with their key tables and register it here keyed by model_id.
