# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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


from functools import wraps
import megatron
import megatron.core
from megatron.core.enums import ModelType
import megatron.core.transformer
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from .utils import is_enable_lora, merge_dicts, modify_keys_with_dict


def unrap_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model, module_instances=None):
        if not module_instances:
            module_instances = megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES
        return fn(model, module_instances)

    return wrapper


def model_provider_func_wrapper(model_provider_func):
    @wraps(model_provider_func)
    def wrapper(*args, **kwargs):
        model = model_provider_func(*args, **kwargs)
        args = get_args()
        if is_enable_lora():
            from peft import LoraConfig, get_peft_model, PeftModel, LoraModel
            from peft.tuners.tuners_utils import check_target_module_exists
            config = core_transformer_config_from_args(args)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                megatron_config=config,
                megatron_core="megatron.core",
            )

            freeze_params = [name for name, param in model.named_parameters() if not param.requires_grad]
            trainable_target_modules = []
            for module_name, _ in model.named_modules():
                if not check_target_module_exists(lora_config, module_name):
                    continue
                if not any(param_name.startswith(module_name) for param_name in freeze_params):
                    trainable_target_modules.append(module_name)

            args.lora_trainable_target_modules = trainable_target_modules
            if not trainable_target_modules:
                return model
            lora_config.target_modules = trainable_target_modules

            model = get_peft_model(model, lora_config)
            model.add_module('module', model.get_base_model())
            model.print_trainable_parameters()
            megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES = tuple(
                list(megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES) + [PeftModel, LoraModel]
            )

        return model

    return wrapper


def _load_base_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        state_dict, checkpoint_name, release = fn(*args, **kwargs)

        if is_enable_lora() and state_dict is not None:
            args = get_args()
            exclude_words = ['base_layer', 'lora_', 'norm']
            state_dict['model'] = modify_keys_with_dict(state_dict['model'], exclude_words)

            if args.lora_load is not None:
                state_dict_lora, checkpoint_name_lora, release_lora = fn(args.lora_load, **kwargs)
                merge_dicts(state_dict, state_dict_lora)
        return state_dict, checkpoint_name, release

    return wrapper


def load_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if is_enable_lora():
            strict = False
            kwargs['strict'] = strict

        return fn(*args, **kwargs)

    return wrapper


def state_dict_for_save_checkpoint(state_dict):
    state_dict_ = dict()
    for key in state_dict:
        if 'lora' in key:
            state_dict_[key] = state_dict[key]
    return state_dict_


def state_dict_for_save_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(self, prefix='', keep_vars=False):
        if is_enable_lora():
            return state_dict_for_save_checkpoint(self.state_dict(prefix=prefix, keep_vars=keep_vars))
        return fn(self, prefix='', keep_vars=False)

    return wrapper


def get_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
        model_provider_func = model_provider_func_wrapper(model_provider_func)
        model = fn(model_provider_func, model_type, wrap_with_ddp)
        return model

    return wrapper


def apply_patches():
    megatron.training.training.load_checkpoint = load_checkpoint_wrapper(
        megatron.training.checkpointing.load_checkpoint)
    megatron.training.checkpointing._load_base_checkpoint = _load_base_checkpoint_wrapper(
        megatron.training.checkpointing._load_base_checkpoint)
    megatron.training.training.get_model = get_model_wrapper(megatron.training.training.get_model)
    megatron.legacy.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.legacy.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint)
    megatron.core.transformer.module.MegatronModule.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.core.transformer.module.MegatronModule.state_dict_for_save_checkpoint)
    megatron.training.checkpointing.unwrap_model = unrap_model_wrapper(megatron.training.checkpointing.unwrap_model)
    megatron.training.training.unwrap_model = unrap_model_wrapper(megatron.training.training.unwrap_model)
