from abc import ABC, abstractmethod
import re
from typing import List, Optional, Tuple

import torch


def permute_moe_expert(
    key: str,
    tensor: torch.Tensor,
    patterns: List[str],
) -> torch.Tensor:
    """Permute MoE expert weights.

    - gate_up_proj: (num_experts, 2 * intermediate, hidden) <-> (num_experts, hidden, 2 * intermediate)
    - down_proj:    (num_experts, 2 * intermediate, hidden) <-> (num_experts, hidden, 2 * intermediate)

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


class WeightTransformPipeline(ABC):
    """Base pipeline for weight format conversion
    """

    @abstractmethod
    def hf_to_dcp(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        """Convert a weight tensor from HuggingFace format to DCP format."""
        pass

    @abstractmethod
    def dcp_to_hf(
        self,
        key: str,
        tensor: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        """Convert a weight tensor from DCP format back to HuggingFace format."""
        pass


class Qwen35WeightTransformPipeline(WeightTransformPipeline):
    """Per-tensor transform pipeline for qwen3.5.
    """
    def __init__(
        self,
    ) -> None:
        self.expert_weight_patterns = [
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"model\.language_model\.layers\.\d+\.mlp\.experts\.down_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.gate_up_proj",
            r"mtp\.layers\.\d+\.mlp\.experts\.down_proj",
        ]

    def hf_to_dcp(
        self, key: str, tensor: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return key, tensor

    def dcp_to_hf(
        self, key: str, tensor: torch.Tensor
    ) -> Tuple[str, torch.Tensor]:
        tensor = permute_moe_expert(key, tensor, self.expert_weight_patterns)
        return key, tensor


WEIGHT_TRANSFORM_PIPELINES = {
    "qwen3_5": Qwen35WeightTransformPipeline,
    "qwen3_5_moe": Qwen35WeightTransformPipeline,
}
