# Copyright 2026 Huawei Technologies Co., Ltd.
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

"""MindSpeed-MM FSDP2 adapter for Kimi-K3.1-Pro."""

from transformers import AutoConfig

from mindspeed_mm.fsdp.models.kimi_k3 import (
    KimiK3ForConditionalGeneration as _KimiK3ForConditionalGeneration,
)
from mindspeed_mm.fsdp.utils.register import model_register

from .configuration_kimi_k3 import KimiK3Config
from .modeling_kimi_linear import apply_kimi_k31_dsa_patch


class KimiK31ForConditionalGeneration(_KimiK3ForConditionalGeneration):
    @staticmethod
    def overwrite_transformer_config(transformer_config, model_args, feature_args):
        transformer_config = _KimiK3ForConditionalGeneration.overwrite_transformer_config(
            transformer_config, model_args, feature_args
        )
        dsa_implementation = getattr(model_args, "dsa_implementation", "eager")
        if dsa_implementation not in ("eager", "ascendc"):
            raise ValueError(
                f"Unsupported dsa_implementation: {dsa_implementation}. "
                "Expected 'eager' or 'ascendc'."
            )
        transformer_config.text_config.dsa_implementation = dsa_implementation
        return transformer_config


try:
    AutoConfig.register("kimi_k31", KimiK3Config)
except ValueError:
    pass

apply_kimi_k31_dsa_patch()
KimiK31ForConditionalGeneration = model_register.register("kimi_k31_pro")(
    KimiK31ForConditionalGeneration
)

__all__ = ["KimiK31ForConditionalGeneration"]
