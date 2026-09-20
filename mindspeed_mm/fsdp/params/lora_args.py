# Copyright 2025 Huawei Technologies Co., Ltd. All rights reserved.
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

"""LoRA configuration arguments for FSDP2 training.

This module defines the dataclass for LoRA-specific configuration
parameters used in FSDP2 distributed training.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

from mindspeed_mm.config.arguments.base_args import BaseArguments


class LoraArguments(BaseArguments):
    """Configuration arguments for LoRA (Low-Rank Adaptation) training.

    This class contains all parameters needed to configure LoRA adapters
    for efficient fine-tuning of large models.

    Attributes:
        enable: Whether to enable LoRA fine-tuning.
        lora_save_only: Whether to export only LoRA weights instead of a full checkpoint.
        rank: Rank of the low-rank matrices.
        alpha: Scaling factor for LoRA weights.
        target_modules: List of target module names/patterns for LoRA.
        dropout: Dropout rate for LoRA layers.
        pretrained_lora_path: Path to pretrained LoRA weights (optional).
    """
    enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA fine-tuning."},
    )
    lora_save_only: bool = field(
        default=False,
        metadata={
            "help": "When LoRA is enabled, export only adapter safetensors and its config."
        },
    )
    rank: int = field(
        default=8,
        metadata={"help": "Rank of the low-rank matrices."},
    )
    alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA weights."},
    )
    target_modules: Union[str, List[str]] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        metadata={
            "help": "Target module names/patterns for LoRA. Supports wildcard patterns "
            "(e.g., 'language_model.layers.{*}.q_proj'), the special keyword "
            "'all-linear' (auto-expand all nn.Linear leaves), or plain strings. "
            "When 'all-linear' is used, model.freeze patterns exclude components "
            "(e.g. ViT / aligner) from LoRA injection."
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for LoRA layers."},
    )
    pretrained_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained LoRA weights to load."},
    )
    disable_peft_moe_conversion: bool = field(
        default=True,
        metadata={
            "help": "Whether to disable PEFT's automatic MoE target_modules → target_parameters "
            "conversion (triggered by config.model_type for MoE models like qwen3_5_moe). "
            "When True (default), gate_proj/up_proj/down_proj stay as target_modules and LoRA "
            "is applied to nn.Linear layers (e.g. shared_expert). "
            "When False, PEFT converts them to target_parameters (gate_up_proj, down_proj), "
            "which targets nn.Parameter objects in the routed experts. Use False if you want "
            "LoRA on the experts module instead of shared_expert."
        },
    )

    def model_post_init(self, __context):
        """Validate LoRA configuration after initialization."""
        # Normalize target_modules: wrap bare strings (e.g. 'all-linear' or 'q_proj')
        # into single-element lists for consistent iteration in downstream logic.
        if isinstance(self.target_modules, str):
            self.target_modules = [self.target_modules]

        if self.enable:
            if self.rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.rank}")

            if self.alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")

            if not self.target_modules:
                raise ValueError("target_modules cannot be empty when LoRA is enabled")

            if not 0.0 <= self.dropout < 1.0:
                raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
