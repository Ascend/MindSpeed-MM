# Copyright 2024 Huawei Technologies Co., Ltd.
# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
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
from .coefficients import resolve_ns_coefficients
from .newton_schulz import (
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_per_expert,
    zeropower_via_newtonschulz5_per_head,
)
from .optimizer import Muon, adjust_lr_wd_for_muon
from .qk_clip import qk_clip_, qk_clip_mla_


__all__ = [
    "Muon",
    "adjust_lr_wd_for_muon",
    "qk_clip_",
    "qk_clip_mla_",
    "resolve_ns_coefficients",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_per_expert",
    "zeropower_via_newtonschulz5_per_head",
]
