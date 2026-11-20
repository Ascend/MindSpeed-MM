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

"""Quantizer plan configs: the config-layer view of the payload quantizers
registered in features/memory/quantizer.py.

Each quantizer type gets one plan class declaring only its own knobs; the
QuantizerPlan discriminated union dispatches on `name` at parse time, so
invalid knob/quantizer combinations are unrepresentable instead of being
ruled out by pairing validators. Consumers resolve a plan to a quantizer
instance via plan.resolve() and hand it to SwapCache.put(quantizer=...);
the swap core never imports this module.
"""
from typing import Annotated, Literal, Union

from pydantic import Field, model_validator

from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.features.memory.quantizer import (
    Fp8BlockwiseQuantizer, Int8PerTokenQuantizer, PayloadQuantizer)
from mindspeed_mm.fsdp.utils.device import is_arch35


def _coerce_quantizer_field(data):
    """Normalize bare quantizer names and reject the removed flat FP8 option."""
    if not isinstance(data, dict):
        return data
    if 'fp8_block_size' in data:
        raise ValueError(
            "'fp8_block_size' is no longer supported; nest it as "
            "quantizer: {name: fp8_blockwise, block_size: <64|128|192|256>}")
    if isinstance(data.get('quantizer'), str):
        data = dict(data)
        data['quantizer'] = {'name': data['quantizer']}
    return data


class Int8PerTokenQuantizerPlan(BaseArguments):
    """int8 per-token payload quantization (see features/memory/quantizer.py)."""
    class Config:
        extra = 'forbid'  # BaseArguments defaults to extra='allow', which would silently swallow a misspelled knob

    name: Literal['int8_pertoken'] = 'int8_pertoken'

    def resolve(self) -> PayloadQuantizer:
        return Int8PerTokenQuantizer.instance()


class Fp8BlockwiseQuantizerPlan(BaseArguments):
    """fp8-e4m3 block-wise payload quantization (Ascend 950PR&950DT products only)."""
    class Config:
        extra = 'forbid'

    name: Literal['fp8_blockwise'] = 'fp8_blockwise'
    block_size: Literal[64, 128, 192, 256] = Fp8BlockwiseQuantizer.DEFAULT_BLOCK_SIZE

    @model_validator(mode='after')
    def _check_arch(self):
        if not is_arch35():
            raise ValueError("'fp8_blockwise' requires Ascend 950PR&950DT products")
        return self

    def resolve(self) -> PayloadQuantizer:
        return Fp8BlockwiseQuantizer.instance(self.block_size)


QuantizerPlan = Annotated[
    Union[Int8PerTokenQuantizerPlan, Fp8BlockwiseQuantizerPlan],
    Field(discriminator='name'),
]
