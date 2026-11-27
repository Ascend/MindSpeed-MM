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

"""Attach the FSDP2 MTP auxiliary head without changing model implementations.

The adapter is installed before LoRA, feature hooks and fully_shard, so its
parameters retain ordinary ``mtp.*`` checkpoint keys and enter the optimizer.
Model specific differences are module paths in ``model.mtp_adapter``.
"""

import copy
import importlib
import inspect
from collections.abc import Mapping

import torch
from torch import nn

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.distributed.context_parallel.utils import cal_split_sizes
from mindspeed_mm.fsdp.models.mtp import MultiTokenPredictionBlock


def _resolve_module(model: nn.Module, path: str) -> nn.Module:
    try:
        module = model.get_submodule(path)
    except AttributeError as exc:
        raise ValueError(f"MTP module path '{path}' does not exist") from exc
    return module


def _import_class(path: str):
    module_name, separator, class_name = path.rpartition(".")
    if not separator:
        raise ValueError(f"MTP class must be a dotted import path: {path}")
    return getattr(importlib.import_module(module_name), class_name)


def _find_backbone(model: nn.Module) -> str:
    # The common HF text-model layouts. Ambiguous or unusual VLMs should set
    # hidden_state_module explicitly so the final normalized states are used.
    for path in ("model.language_model", "language_model.model", "model"):
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if hasattr(module, "layers") and hasattr(module, "embed_tokens"):
            return path
    raise ValueError("Cannot find the text backbone for MTP; set mtp_adapter.hidden_state_module")


def _norm_factory(norm: nn.Module):
    norm_cls = type(norm)
    signature = inspect.signature(norm_cls)

    def build(hidden_size, eps):
        kwargs = {}
        if "eps" in signature.parameters:
            kwargs["eps"] = eps
        elif "variance_epsilon" in signature.parameters:
            kwargs["variance_epsilon"] = eps
        return norm_cls(hidden_size, **kwargs)

    return build


def _layer_factory(layer_cls, layer_kwargs, layer_index):
    signature = inspect.signature(layer_cls)
    params = signature.parameters

    def build(config, _index, is_mtp=True):
        kwargs = dict(layer_kwargs)
        if "is_mtp" in params:
            kwargs.setdefault("is_mtp", is_mtp)
        if "layer_idx" in params:
            kwargs.setdefault("layer_idx", layer_index + _index)
        elif "layer_index" in params:
            kwargs.setdefault("layer_index", layer_index + _index)
        elif "layer_number" in params:
            kwargs.setdefault("layer_number", layer_index + _index)
        try:
            return layer_cls(config, **kwargs)
        except TypeError as exc:
            raise ValueError(
                f"Cannot construct MTP decoder {layer_cls.__name__}; "
                "set mtp_adapter.decoder_class and layer_kwargs"
            ) from exc

    return build


def _hidden_state(output):
    if isinstance(output, torch.Tensor):
        return output
    hidden = getattr(output, "last_hidden_state", None)
    if isinstance(hidden, torch.Tensor):
        return hidden
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError("MTP hidden_state_module must return a tensor or last_hidden_state")


def attach_mtp_adapter(model: nn.Module, model_args, feature_args) -> nn.Module:
    """Install one generic MTP block and two forward hooks on a built model.

    An enabled MTP uses configured module paths or common text-backbone
    layouts. A model with its own MTP block cannot use both implementations.
    """
    plan = getattr(model_args, "mtp_adapter", None)
    num_layers = getattr(model_args, "mtp_num_layers", 0)
    if not num_layers:
        return model
    if plan is None:
        plan = {}
    if not isinstance(plan, Mapping):
        raise ValueError("model.mtp_adapter must be a mapping of module paths")
    if num_layers < 0:
        raise ValueError("model.mtp_num_layers must be non-negative")
    if getattr(model, "mtp", None) is not None:
        raise ValueError("Model already has an MTP block; remove mtp_adapter or disable model-native MTP")

    backbone_path = plan.get("hidden_state_module") or _find_backbone(model)
    backbone = _resolve_module(model, backbone_path)
    layers = _resolve_module(model, plan.get("decoder_layers", f"{backbone_path}.layers"))
    if not isinstance(layers, (nn.ModuleList, nn.Sequential)) or not layers:
        raise ValueError("MTP decoder_layers must be a nonempty ModuleList or Sequential")
    embed = _resolve_module(model, plan.get("embed_tokens", f"{backbone_path}.embed_tokens"))
    norm = _resolve_module(model, plan.get("final_norm", f"{backbone_path}.norm"))
    head_path = plan.get("lm_head")
    if head_path is None:
        for candidate in ("lm_head", "language_model.lm_head"):
            try:
                _resolve_module(model, candidate)
                head_path = candidate
                break
            except ValueError:
                pass
    if head_path is None:
        raise ValueError("Cannot find the output head for MTP; set mtp_adapter.lm_head")
    head = _resolve_module(model, head_path)
    rotary_path = plan.get("rotary_emb", f"{backbone_path}.rotary_emb")
    try:
        rotary = _resolve_module(model, rotary_path)
    except ValueError:
        if "rotary_emb" in plan:
            raise
        rotary = None

    root_config = getattr(model, "config", None)
    config_path = plan.get("config_path")
    if config_path:
        source_config = model
        try:
            for part in config_path.split("."):
                source_config = getattr(source_config, part)
        except AttributeError as exc:
            raise ValueError(f"MTP config_path '{config_path}' does not exist") from exc
    else:
        source_config = getattr(backbone, "config", None)
        if source_config is None and root_config is not None:
            source_config = getattr(root_config, "text_config", root_config)
    if source_config is None:
        raise ValueError("MTP text backbone needs a config with hidden_size and vocab_size")
    config = copy.copy(source_config)
    for field in ("hidden_size", "vocab_size"):
        if not hasattr(config, field):
            raise ValueError(f"MTP backbone config has no {field}")
    config.mtp_num_layers = num_layers
    config.enable_chunk_loss = bool(getattr(feature_args, "enable_chunk_loss", False))
    config.enable_dynamic_chunk_loss = bool(getattr(feature_args, "enable_dynamic_chunk_loss", False))

    layer_cls = _import_class(plan["decoder_class"]) if "decoder_class" in plan else type(layers[-1])
    layer_index = plan.get("layer_index", 0)
    if not isinstance(layer_index, int) or layer_index < 0:
        raise ValueError("mtp_adapter.layer_index must be a non-negative integer")
    layer_kwargs = plan.get("layer_kwargs", {})
    if not isinstance(layer_kwargs, Mapping):
        raise ValueError("mtp_adapter.layer_kwargs must be a mapping")
    mask_builder_path = plan.get("causal_mask_builder")
    if mask_builder_path is None:
        mtp_kwargs = {}
    else:
        causal_mask_builder = None if mask_builder_path == "none" else _import_class(mask_builder_path)
        mtp_kwargs = {"causal_mask_builder": causal_mask_builder}
    mtp = MultiTokenPredictionBlock(
        config,
        _layer_factory(layer_cls, layer_kwargs, layer_index),
        _norm_factory(norm),
        position_ids_mode=plan.get("position_ids_mode", "text"),
        share_decoder=plan.get("share_decoder", True),
        **mtp_kwargs,
    )
    if not any(parameter.is_meta for parameter in mtp.parameters()):
        reference = next(embed.parameters(), None)
        if reference is not None:
            mtp.to(device=reference.device, dtype=reference.dtype)
    # Ownership metadata is not a parameter or a checkpoint key.
    mtp._ms_mm_mtp_adapter_created = True
    model.add_module("mtp", mtp)

    captured = {}

    def start_capture(_module, _args, kwargs):
        captured.clear()
        if model.training and kwargs.get("labels") is not None and kwargs.get("input_ids") is not None:
            captured["active"] = True

    def capture_hidden(_module, _args, kwargs, output):
        if not captured.get("active"):
            return
        captured["output"] = output
        captured["kwargs"] = kwargs

    def append_loss(_module, args, kwargs, output):
        try:
            labels = kwargs.get("labels")
            input_ids = kwargs.get("input_ids")
            if not model.training or labels is None or input_ids is None:
                return output
            if isinstance(output, tuple):
                raise ValueError("MTP requires return_dict=True from the model")
            if getattr(output, "loss", None) is None:
                raise ValueError("MTP requires a model output with a base loss")
            if "output" not in captured:
                raise ValueError(f"MTP hidden_state_module '{backbone_path}' was not called")
            backbone_output = captured["output"]
            hidden = _hidden_state(backbone_output)
            # Keep token IDs global: the MTP block shifts them before splitting
            # embeddings to match the local hidden states.
            expected_seq_len = input_ids.shape[1]
            ps = get_parallel_state()
            if ps.is_ulysses_enable():
                # Mirror split_forward_gather_backward_with_cp: ring first,
                # then the rank-specific (possibly uneven) Ulysses split.
                if ps.is_ring_enable():
                    ring_size = ps.get_ring_group_size()
                    if expected_seq_len % (2 * ring_size):
                        raise ValueError("MTP sequence length must be a multiple of 2 * ring_size")
                    expected_seq_len //= ring_size
                expected_seq_len = cal_split_sizes(
                    expected_seq_len, ps.get_ulysses_group_size()
                )[ps.get_ulysses_rank()]
            expected_shape = (input_ids.shape[0], expected_seq_len)
            if hidden.ndim != 3 or hidden.shape[:2] != expected_shape:
                raise ValueError(
                    "MTP input_ids and text hidden states have different batch/sequence shapes; "
                    f"expected hidden batch/sequence {expected_shape}, got {tuple(hidden.shape[:2])}; "
                    "configure an aligned text backbone and global token IDs"
                )
            if labels.shape != input_ids.shape:
                raise ValueError("MTP labels and input_ids must have the same batch/sequence shape")
            backbone_kwargs = captured["kwargs"]
            position_ids = getattr(backbone_output, "position_ids", None)
            if position_ids is None:
                position_ids = backbone_kwargs.get("position_ids", kwargs.get("position_ids"))
            loss_function = getattr(model, "mtp_loss_function", None) or getattr(model, "loss_function", None)
            if loss_function is None:
                raise ValueError("MTP loss function was not installed by the training engine")
            extra = {key: kwargs[key] for key in ("cu_seqlens",) if key in kwargs}
            mtp_loss = mtp(
                hidden,
                input_ids=input_ids,
                labels=labels,
                attention_mask=kwargs.get("attention_mask"),
                position_ids=position_ids,
                embed_tokens=embed,
                rotary_emb=rotary,
                output_layer=head,
                loss_function=loss_function,
                logits_to_keep=kwargs.get("logits_to_keep", 0),
                seq_len=input_ids.shape[1],
                **extra,
            )
            if isinstance(output, dict):
                output["mtp_loss"] = mtp_loss
            else:
                setattr(output, "mtp_loss", mtp_loss)
            return output
        finally:
            captured.clear()

    model.register_forward_pre_hook(start_capture, with_kwargs=True)
    backbone.register_forward_hook(capture_hidden, with_kwargs=True)
    model.register_forward_hook(append_loss, with_kwargs=True)
    return model


@torch.no_grad()
def initialize_mtp_parameters(model: nn.Module, seed: int) -> None:
    """Initialize adapter-created MTP shards before training or a partial load.

    Base HF checkpoints normally lack ``mtp.*``. Meta-device parameters have
    no valid value after ``to_empty``; a non-strict load leaves them untouched.
    Initialize only the adapter-owned auxiliary block so base weights and
    model-native MTP initialization/RNG behavior are not modified.
    """
    mtp = getattr(model, "mtp", None)
    if mtp is None or not getattr(mtp, "_ms_mm_mtp_adapter_created", False):
        return
    from mindspeed_mm.fsdp.utils.utils import _precreate_dtensor_rng_tracker

    _precreate_dtensor_rng_tracker(mtp, seed)
    hf_initializer = getattr(model, "_initialize_weights", None)
    for module in mtp.modules():
        module._is_hf_initialized = False
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
        else:
            # Some custom norms define parameters directly, but their
            # HF _init_weights omits those parameters.
            # Give every direct parameter a valid value before model-specific
            # initialization overrides it.
            for name, parameter in module.named_parameters(recurse=False):
                if "norm" in type(module).__name__.lower() and name == "weight":
                    torch.nn.init.ones_(parameter)
                elif parameter.ndim >= 2 and parameter.is_floating_point():
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
                else:
                    torch.nn.init.zeros_(parameter)
        if callable(hf_initializer):
            hf_initializer(module)
    model._ms_mm_mtp_preinitialized = True
