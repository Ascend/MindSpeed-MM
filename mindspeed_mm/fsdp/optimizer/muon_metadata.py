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

"""Muon layout declarations."""


def _validate_muon_metadata(parameter, metadata):
    """Validate against the global parameter shape."""
    if not isinstance(metadata, dict):
        raise TypeError("Muon parameter metadata must be a dictionary")
    mode = metadata.get("muon_mode")
    if mode == "per_head":
        count_key, ndim = "muon_num_heads", 2
    elif mode == "per_expert":
        count_key, ndim = "muon_num_experts", 3
    else:
        raise ValueError(f"unsupported Muon annotation mode: {mode!r}")
    if metadata.keys() - {"muon_mode", count_key}:
        raise ValueError("Muon layout annotations contain unsupported fields")
    count = metadata.get(count_key)
    if type(count) is not int or count <= 0 or parameter.ndim != ndim:
        raise ValueError(f"{mode} Muon requires a {ndim}-D weight and a positive integer {count_key}")
    if min(parameter.shape) <= 0:
        raise ValueError("Muon annotations require nonempty matrices")
    if mode == "per_head" and parameter.size(0) % count:
        raise ValueError("per-head Muon requires complete head row blocks")
    if mode == "per_expert" and parameter.size(0) != count:
        raise ValueError("per-expert Muon requires a [num_experts, rows, columns] weight")


def register_muon_layout(owner, parameter_name="weight", *, mode, num_heads=None, num_experts=None):
    """Keep layout metadata on the module across parameter replacement."""
    if not isinstance(parameter_name, str) or not parameter_name or "." in parameter_name:
        raise ValueError("Muon layouts must name a parameter directly owned by the module")
    metadata = {"muon_mode": mode}
    if num_heads is not None:
        metadata["muon_num_heads"] = num_heads
    if num_experts is not None:
        metadata["muon_num_experts"] = num_experts
    _validate_muon_metadata(owner.get_parameter(parameter_name), metadata)
    layouts = owner.__dict__.setdefault("_muon_layouts", {})
    if parameter_name in layouts and layouts[parameter_name] != metadata:
        raise ValueError(f"Conflicting Muon layout for {parameter_name}")
    layouts[parameter_name] = metadata
