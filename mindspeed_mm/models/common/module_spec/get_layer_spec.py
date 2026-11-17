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

from mindspeed_mm.models.common.module_spec.qwen2vl_layer_spec import get_qwen2vl_layer_spec, get_mlp_module_spec, \
    get_qwen2vl_llm_layer_spec, get_qwen2_5_vit_layer_spec
from mindspeed_mm.models.common.module_spec.qwen3vl_layer_spec import get_qwen3vl_layer_spec
from mindspeed_mm.models.common.module_spec.qwen2_5omni_layer_spec import get_qwen_omni_audio_layer_spec
from mindspeed_mm.models.common.module_spec.internvl_layer_spec import get_language_layer_spec
from mindspeed_mm.models.common.module_spec.deepseekvl_layer_spec import get_deepseekvl_model_spec
from mindspeed_mm.models.common.module_spec.qwen3vl_layer_spec import get_qwen3vl_llm_layer_local_spec
from mindspeed_mm.models.common.module_spec.glm4v_layer_spec import get_glm4v_layer_spec, get_glm4v_vit_layer_spec
from mindspeed_mm.models.common.module_spec.videoalign_layer_spec import get_videoalign_layer_spec, get_videoalign_llm_layer_spec

audio_layer_specs = {'qwen_omni': get_qwen_omni_audio_layer_spec}

vit_layer_specs = {'qwen2vit': get_qwen2vl_layer_spec,
                  'qwen3vit': get_qwen3vl_layer_spec,
                   'qwen2_5_vit': get_qwen2_5_vit_layer_spec,
                   'glm4v_vit': get_glm4v_vit_layer_spec,
                   'videoalign_vit': get_videoalign_layer_spec}

llm_layer_specs = {'qwen2lm': get_qwen2vl_llm_layer_spec,
                   'qwen2_5_lm': get_qwen2vl_llm_layer_spec,
                   'qwen2_5_omni_thinker': get_qwen2vl_llm_layer_spec,
                   'internllm': get_language_layer_spec,
                   'deepseek': get_deepseekvl_model_spec,
                   "qwen3_lm": get_qwen3vl_llm_layer_local_spec,
                   "glm4v_lm": get_glm4v_layer_spec,
                   "videoalign_lm": get_videoalign_llm_layer_spec}

projector_layer_specs = {'lnmlp': get_mlp_module_spec, 'mlp': get_mlp_module_spec}


def get_vit_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in vit_layer_specs:
            return vit_layer_specs[config.model_id](config, is_vit=True)
    return None


def get_audio_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in audio_layer_specs:
            return audio_layer_specs[config.model_id](config, is_vit=True)
    return None


def get_llm_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in llm_layer_specs:
            return llm_layer_specs[config.model_id](config, is_vit=False)
    return None


def get_projector_layer_spec(config, *args, **kwargs):
    if getattr(config, 'model_id', None) is not None:
        if config.model_id in projector_layer_specs:
            return projector_layer_specs[config.model_id](config, use_te=False).submodules
    return None
