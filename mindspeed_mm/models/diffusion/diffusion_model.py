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

from .wan_flow_match_scheduler import WanFlowMatchScheduler
from .diffusers_scheduler import DiffusersScheduler
from .hunyuan_video_15_scheduler import Hunyuan_15_FlowMatchDiscreteScheduler


DIFFUSION_MODEL_MAPPINGS = {
    "wan_flow_match_scheduler": WanFlowMatchScheduler,
    "hunyuanvideo_15_diffusion": Hunyuan_15_FlowMatchDiscreteScheduler,
}


class DiffusionModel:
    """
    Factory class for all customized diffusion models and diffusers schedulers.
    Args:
        config:
        {
            "model_id": "ddpm",
            "num_timesteps": 1000,
            "beta_schedule": "linear",
            ...
        }
    """

    def __init__(self, config):
        if config.model_id in DIFFUSION_MODEL_MAPPINGS:
            model_cls = DIFFUSION_MODEL_MAPPINGS[config.model_id]
            self.diffusion = model_cls(**config.to_dict())
        else:
            self.diffusion = DiffusersScheduler(config.to_dict())

    def get_model(self):
        return self.diffusion
