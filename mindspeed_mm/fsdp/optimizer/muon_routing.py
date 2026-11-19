# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from .muon_metadata import _validate_muon_metadata

logger = logging.getLogger(f"{__package__}.optimizer")
DEFAULT_MUON_FALLBACK_PARAM_KEYWORDS = ("embedding", "embed_tokens", "output_layer", "lm_head")


def _adamw_fallback_params(
    model: "torch.nn.Module", fallback_param_keywords: Optional[Iterable[str]] = None,
) -> set[torch.nn.Parameter]:
    """Return the parameters that always use the AdamW fallback.

    These are the parameters of ``nn.Embedding`` modules and of modules named
    ``lm_head`` or ``output_layer``, every bias, and every parameter whose name
    contains one of ``DEFAULT_MUON_FALLBACK_PARAM_KEYWORDS`` or the extra
    ``fallback_param_keywords`` (case-insensitive).
    """
    adamw_fallback: set[torch.nn.Parameter] = set()
    for module_path, module in model.named_modules():
        role = module_path.rsplit(".", 1)[-1]
        if isinstance(module, torch.nn.Embedding) or role in {"lm_head", "output_layer"}:
            adamw_fallback.update(module.parameters(recurse=False))
    keywords = list(DEFAULT_MUON_FALLBACK_PARAM_KEYWORDS)
    if isinstance(fallback_param_keywords, str):
        keywords.append(fallback_param_keywords)
    elif fallback_param_keywords:
        keywords.extend(fallback_param_keywords)
    keywords = tuple(keyword.lower() for keyword in keywords if keyword)
    adamw_fallback.update(parameter for name, parameter in model.named_parameters()
                          if name.lower().endswith(".bias")
                          or any(keyword in name.lower() for keyword in keywords))
    return adamw_fallback


def _is_muon_eligible(
    param: torch.nn.Parameter,
    adamw_fallback_params: set[torch.nn.Parameter],
) -> bool:
    # A 2-D weight with a dimension of size 1 is effectively a vector:
    # orthogonalizing it only normalizes it, so it uses the AdamW fallback.
    return (
        param.ndim == 2
        and min(param.shape) > 1
        and param not in adamw_fallback_params
    )


def _collect_muon_metadata(model, *, per_head, per_expert, adamw_fallback_params):
    """Resolve active declarations across the whole model."""
    annotations = {}
    if not (per_head or per_expert):
        return annotations
    for module_path, module in model.named_modules():
        for name, specification in module.__dict__.get("_muon_layouts", {}).items():
            mode = specification["muon_mode"]
            if mode not in {"per_head", "per_expert"}:
                raise ValueError(f"unsupported Muon annotation mode: {mode!r}")
            if not (per_head if mode == "per_head" else per_expert):
                continue
            qualified_name = f"{module_path}.{name}" if module_path else name
            try:
                parameter = module.get_parameter(name)
            except AttributeError as error:
                raise ValueError(f"Muon annotation refers to missing parameter {qualified_name}") from error
            if not parameter.requires_grad or parameter in adamw_fallback_params:
                continue
            _validate_muon_metadata(parameter, specification)
            if parameter in annotations and annotations[parameter] != specification:
                raise ValueError(f"Conflicting Muon annotations for {qualified_name}")
            annotations[parameter] = dict(specification)
    for enabled, mode in ((per_head, "per_head"), (per_expert, "per_expert")):
        if enabled and not any(entry["muon_mode"] == mode for entry in annotations.values()):
            raise ValueError(f"{mode} Muon requested, but no eligible trainable annotations were found")
    return annotations


def _configure_muon_qk_clip(model, *, enabled):
    """Bind current Q/K parameters and reset model-side recording when disabled."""
    bindings = {}
    for module in model.modules():
        configure = getattr(module, "configure_muon_qk_clip", None)
        if configure is None:
            continue
        for name, metadata in configure(enabled=enabled).items():
            parameter = module.get_parameter(name)
            if not enabled or parameter in bindings:
                raise ValueError("Unexpected or duplicate QK-Clip annotation")
            if any(not key.startswith("qk_clip_") for key in metadata):
                raise ValueError("QK-Clip annotations contain unsupported fields")
            params = metadata.get("qk_clip_params")
            if not isinstance(params, (tuple, list)) or len(params) != 2 or params[0] is not parameter:
                raise ValueError("QK-Clip annotations must bind the current query and key parameters")
            if not callable(metadata.get("qk_clip_metric_provider")):
                raise ValueError("QK-Clip annotations require a metric provider")
            bindings[parameter] = dict(metadata)
    return bindings


def _log_muon_group_routing(
    model: "torch.nn.Module",
    param_groups: Sequence[Dict[str, Any]],
    label: str,
    default_lr: float,
    muon_fallback_param_keywords: Optional[Iterable[str]] = None,
) -> None:
    """Log each group's mode and size, and the names forced to the AdamW fallback."""
    name_by_param = {parameter: name for name, parameter in model.named_parameters()}
    adamw_fallback = _adamw_fallback_params(model, muon_fallback_param_keywords)
    adamw_fallback_names = sorted(
        name_by_param[parameter]
        for group in param_groups
        if not group.get("use_muon", False)
        for parameter in group["params"]
        if parameter in adamw_fallback and parameter in name_by_param
    )
    summary = [
        {
            "mode": group.get(
                "muon_mode", "muon" if group.get("use_muon") else "adamw_fallback"
            ),
            "lr": group.get("lr", default_lr),
            "parameters": len(group["params"]),
            "elements": sum(parameter.numel() for parameter in group["params"]),
        }
        for group in param_groups
    ]
    logger.info("Muon parameter groups (%s): %s", label, summary)
    logger.info("Muon AdamW fallback parameters by name or type (%s): %s", label, adamw_fallback_names)


def _mark_muon_param_groups(
    model: "torch.nn.Module",
    param_groups: Sequence[Dict[str, Any]],
    muon_per_head: bool = False,
    muon_per_expert: bool = False,
    qk_clip_threshold: Optional[float] = None,
    qk_clip_alpha: float = 0.5,
    muon_fallback_param_keywords: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    """Split parameter groups into Muon groups and AdamW fallback groups.

    Each input group becomes: one full-matrix Muon group, one group per
    per-head or per-expert weight (with its head or expert count and any
    QK-Clip settings), and one AdamW fallback group. Other group settings such
    as ``lr`` and ``weight_decay`` are copied into every split group.
    """
    if qk_clip_threshold is not None and not muon_per_head:
        raise ValueError("QK-Clip requires muon_per_head")
    adamw_fallback_params = _adamw_fallback_params(model, muon_fallback_param_keywords)
    annotations = _collect_muon_metadata(
        model, per_head=muon_per_head, per_expert=muon_per_expert,
        adamw_fallback_params=adamw_fallback_params,
    )
    qk_params = _configure_muon_qk_clip(model, enabled=qk_clip_threshold is not None)
    if qk_clip_threshold is not None:
        if not qk_params:
            raise ValueError("QK-Clip requested, but no compatible model annotations were provided")
        if any(annotations.get(parameter, {}).get("muon_mode") != "per_head" for parameter in qk_params):
            raise ValueError("QK-Clip requires eligible trainable per-head query annotations")
    per_head_params = {
        parameter: metadata["muon_num_heads"] for parameter, metadata in annotations.items()
        if metadata["muon_mode"] == "per_head"
    }
    per_expert_params = {
        parameter: metadata["muon_num_experts"] for parameter, metadata in annotations.items()
        if metadata["muon_mode"] == "per_expert"
    }
    marked_groups: List[Dict[str, Any]] = []

    for group in param_groups:
        params = group.get("params", [])
        muon_params = []
        fallback_params = []
        per_head_groups = []
        per_expert_groups = []

        for p in params:
            if not p.requires_grad:
                continue
            if p in adamw_fallback_params:
                fallback_params.append(p)
            elif p in per_head_params:
                per_head_groups.append((p, per_head_params[p]))
            elif p in per_expert_params:
                per_expert_groups.append((p, per_expert_params[p]))
            elif _is_muon_eligible(p, adamw_fallback_params):
                muon_params.append(p)
            else:
                fallback_params.append(p)

        group_base = {k: v for k, v in group.items() if k != "params"}
        if muon_params:
            marked_groups.append({**group_base, "params": muon_params, "use_muon": True})
        for param, num_heads in per_head_groups:
            qk_metadata = qk_params.get(param, {})
            qk_controls = (
                {"qk_clip_threshold": qk_clip_threshold, "qk_clip_alpha": qk_clip_alpha}
                if qk_metadata
                else {}
            )
            marked_groups.append(
                {
                    **group_base,
                    "params": [param],
                    "use_muon": True,
                    "muon_mode": "per_head",
                    "muon_num_heads": num_heads,
                    **qk_metadata,
                    **qk_controls,
                }
            )
        for param, num_experts in per_expert_groups:
            marked_groups.append(
                {
                    **group_base,
                    "params": [param],
                    "use_muon": True,
                    "muon_mode": "per_expert",
                    "muon_num_experts": num_experts,
                }
            )
        if fallback_params:
            marked_groups.append({**group_base, "params": fallback_params, "use_muon": False})

    names = {parameter: name for name, parameter in model.named_parameters()}
    for group in marked_groups:
        # Parameter names, used by the cross-rank consistency check in
        # optimizer/muon/owner_plan.py and by the routing log.
        group["param_names"] = [names.get(parameter) for parameter in group["params"]]
    return marked_groups
