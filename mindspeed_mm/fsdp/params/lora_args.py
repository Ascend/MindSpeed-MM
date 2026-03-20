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
from typing import List, Literal, Optional

from mindspeed_mm.fsdp.params.utils import allow_extra_fields


@allow_extra_fields
@dataclass
class LoraArguments:
    """Configuration arguments for LoRA (Low-Rank Adaptation) training.
    
    This class contains all parameters needed to configure LoRA adapters
    for efficient fine-tuning of large models.
    
    Attributes:
        enable: Whether to enable LoRA fine-tuning.
        rank: Rank of the low-rank matrices.
        alpha: Scaling factor for LoRA weights.
        target_modules: List of target module names/patterns for LoRA.
        dropout: Dropout rate for LoRA layers.
        init_lora_weights: Weight initialization method.
        pretrained_lora_path: Path to pretrained LoRA weights (optional).
        save_mode: Mode for saving LoRA weights.
        lora_target_modules_support: List of supported module types.
    """
    enable: bool = field(
        default=False,
        metadata={"help": "Enable LoRA fine-tuning."},
    )
    rank: int = field(
        default=8,
        metadata={"help": "Rank of the low-rank matrices."},
    )
    alpha: int = field(
        default=16,
        metadata={"help": "Scaling factor for LoRA weights."},
    )
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj"],
        metadata={
            "help": "List of target module names/patterns for LoRA. "
            "Supports wildcard patterns (e.g., 'language_model.layers.{*}.q_proj')."
        },
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for LoRA layers."},
    )
    init_lora_weights: str = field(
        default="kaiming",
        metadata={
            "help": "Weight initialization method for LoRA. "
            "Options: 'true', 'false', 'gaussian', 'kaiming', 'loftq'."
        },
    )
    pretrained_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained LoRA weights to load."},
    )
    save_mode: Literal["lora_only", "full_model"] = field(
        default="lora_only",
        metadata={
            "help": "Mode for saving LoRA weights. "
            "'lora_only': Save only LoRA adapter weights. "
            "'full_model': Save complete model with LoRA."
        },
    )
    lora_target_modules_support: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of supported module types for validation. "
            "If None, validation is skipped."
        },
    )
    
    def __post_init__(self) -> None:
        """Validate LoRA configuration after initialization."""
        if self.enable:
            if self.rank <= 0:
                raise ValueError(f"LoRA rank must be positive, got {self.rank}")
            
            if self.alpha <= 0:
                raise ValueError(f"LoRA alpha must be positive, got {self.alpha}")
            
            if not self.target_modules:
                raise ValueError("target_modules cannot be empty when LoRA is enabled")
            
            if not 0.0 <= self.dropout < 1.0:
                raise ValueError(f"LoRA dropout must be in [0, 1), got {self.dropout}")
            
            valid_init_methods = ["true", "false", "gaussian", "kaiming", "loftq"]
            if self.init_lora_weights.lower() not in valid_init_methods:
                raise ValueError(
                    f"init_lora_weights must be one of {valid_init_methods}, "
                    f"got {self.init_lora_weights}"
                )
            
            if self.save_mode not in ["lora_only", "full_model"]:
                raise ValueError(
                    f"save_mode must be 'lora_only' or 'full_model', "
                    f"got {self.save_mode}"
                )