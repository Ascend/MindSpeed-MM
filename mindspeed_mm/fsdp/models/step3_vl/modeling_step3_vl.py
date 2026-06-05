from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

import torch
import yaml
from transformers import AutoConfig
from transformers.cache_utils import Cache
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from mindspeed_mm.fsdp.utils.register import model_register


DEFAULT_STEP3_TEXT_LAYERS = 36
DEFAULT_STEP3_VISION_LAYERS = 47


def _get_config_value(config, name, default):
    value = getattr(config, name, None)
    return default if value is None else value


def _fix_step3_config(config):
    text_layers = _get_config_value(config.text_config, "num_hidden_layers", DEFAULT_STEP3_TEXT_LAYERS)
    vision_layers = _get_config_value(config.vision_config, "layers", DEFAULT_STEP3_VISION_LAYERS)

    # Step3 remote code reads these fields while constructing the module tree, but some released
    # configs omit FSDP2/meta-init compatible defaults. Prefer the checkpoint config and only fill
    # missing or inconsistent fields so the adapter does not hard-code one parameter scale.
    config.text_config.max_window_layers = text_layers
    config.vision_config.layers = vision_layers
    config.vocab_size = config.text_config.vocab_size
    config.use_cache = False
    config.text_config.use_cache = False
    return config


def _load_step3_base_class(model_path: str):
    return get_class_from_dynamic_module(
        "modeling_step_vl.Step3VL10BForCausalLM",
        model_path,
        local_files_only=True,
    )


def _build_key_mapping():
    return {
        r"^vision_model": "model.vision_model",
        r"^model(?!\.(language_model|vision_model))": "model.language_model",
        r"^vit_large_projector": "model.vit_large_projector",
    }


def _load_step3_model_path_from_config():
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        raise RuntimeError("Step3-VL plugin requires a YAML config file as the first argument.")

    with Path(sys.argv[1]).open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    model_path = config.get("model", {}).get("model_name_or_path")
    if not model_path:
        raise RuntimeError("Step3-VL config must set `model.model_name_or_path`.")
    return model_path


_Step3BaseForCausalLM = _load_step3_base_class(_load_step3_model_path_from_config())
_Step3CausalLMOutput = _Step3BaseForCausalLM.forward.__globals__["StepVLCausalLMOutputWithPast"]


@model_register.register("step3_vl")
class Step3VLFSDPForCausalLM(_Step3BaseForCausalLM):
    @classmethod
    def overwrite_transformer_config(cls, transformer_config, model_args):
        return _fix_step3_config(transformer_config)

    @classmethod
    def _from_config(cls, config):
        return cls(_fix_step3_config(config))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, config=None, **kwargs):
        model_path = Path(pretrained_model_name_or_path)
        if config is None:
            config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True,
            )

        config = _fix_step3_config(config)
        kwargs.setdefault("key_mapping", _build_key_mapping())
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("local_files_only", True)
        return super().from_pretrained(model_path, *model_args, config=config, **kwargs)

    @property
    def visual(self):
        return self.model.vision_model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        num_patches=None,
        patch_pixel_values=None,
        patch_newline_mask=None,
        pixel_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if isinstance(num_patches, torch.Tensor):
            num_patches = num_patches.detach().cpu().tolist()

        outputs = self.model(
            input_ids=input_ids,
            num_patches=num_patches,
            pixel_values=pixel_values,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if use_cache is None else use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
            )

        return _Step3CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=getattr(outputs, "past_key_values", None),
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
