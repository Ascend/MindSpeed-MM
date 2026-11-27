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

"""Configurable pre-norm decoder for MTP outside a model's residual chain."""

import copy
import importlib
import inspect
from collections.abc import Mapping

import torch
from torch import nn


def _component_class(value):
    if isinstance(value, str):
        module_name, separator, class_name = value.rpartition(".")
        if not separator:
            raise ValueError("MTP component classes must use dotted import paths")
        value = getattr(importlib.import_module(module_name), class_name)
    if not inspect.isclass(value) or not issubclass(value, nn.Module):
        raise ValueError("MTP component must be an nn.Module class")
    return value


def _build_component(component_cls, config, layer_idx):
    parameters = inspect.signature(component_cls).parameters
    kwargs = {}
    for name in ("layer_idx", "layer_index", "layer_number"):
        if name in parameters:
            kwargs[name] = layer_idx
            break
    if "is_mtp" in parameters:
        kwargs["is_mtp"] = True
    return component_cls(config, **kwargs)


class PlainResidualDecoder(nn.Module):
    """Compose existing attention/MLP/norm classes without model-native wiring.

    Parameters retain the conventional self_attn/mlp/input_layernorm/
    post_attention_layernorm names. The decoder owns its residuals and uses
    an independent configuration, so backbone residual chains and selective
    recompute settings do not need changes in model implementations.
    """

    def __init__(
        self,
        config,
        layer_idx,
        attention_class,
        mlp_class,
        norm_class,
        config_overrides=None,
        is_mtp=True,
    ):
        super().__init__()
        if config_overrides is None:
            config_overrides = {}
        if not isinstance(config_overrides, Mapping):
            raise ValueError("MTP decoder config_overrides must be a mapping")
        self.config = copy.copy(config)
        for name, value in config_overrides.items():
            if not isinstance(name, str):
                raise ValueError("MTP decoder configuration keys must be strings")
            setattr(self.config, name, value)
        self.layer_idx = layer_idx
        self.is_mtp = is_mtp
        self.self_attn = _build_component(_component_class(attention_class), self.config, layer_idx)
        self.mlp = _build_component(_component_class(mlp_class), self.config, layer_idx)
        norm_cls = _component_class(norm_class)
        norm_parameters = inspect.signature(norm_cls).parameters
        eps = getattr(self.config, "rms_norm_eps", getattr(self.config, "layer_norm_epsilon", 1e-5))
        norm_kwargs = {}
        if "eps" in norm_parameters:
            norm_kwargs["eps"] = eps
        elif "variance_epsilon" in norm_parameters:
            norm_kwargs["variance_epsilon"] = eps
        self.input_layernorm = norm_cls(self.config.hidden_size, **norm_kwargs)
        self.post_attention_layernorm = norm_cls(self.config.hidden_size, **norm_kwargs)
        parameters = inspect.signature(self.self_attn.forward).parameters
        self._attention_forward_names = set(parameters)
        self._attention_accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()
        )

    def forward(self, hidden_states, position_embeddings=None, **kwargs):
        # NoPE components do not need the shared MTP block's rotary argument.
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if not self._attention_accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in self._attention_forward_names}
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states), **kwargs)
        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        if not isinstance(hidden_states, torch.Tensor):
            raise ValueError("MTP attention must return a tensor or a tuple starting with a tensor")
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
