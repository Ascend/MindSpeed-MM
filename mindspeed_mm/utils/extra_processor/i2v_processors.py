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

from .cogvideox_i2v_processor import CogVideoXI2VProcessor
from .hunyuanvideo_i2v_processor import HunyuanVideoI2VProcessor
from .opensoraplan_i2v_processor import OpenSoraPlanI2VProcessor
from .wan_i2v_processor import WanVideoI2VProcessor
from .wan_ti2v_processor import WanVideoTI2VProcessor
from .stepvideo_i2v_processor import StepVideoI2VProcessor
from .hunyuanvideo_15_i2v_processor import HunyuanVideo15I2VProcessor

I2V_PROCESSOR_MAPPINGS = {
    "cogvideox_i2v_processor": CogVideoXI2VProcessor,
    "opensoraplan_i2v_processor": OpenSoraPlanI2VProcessor,
    "wan_i2v_processor": WanVideoI2VProcessor,
    "wan_ti2v_processor": WanVideoTI2VProcessor,
    "hunyuanvideo_i2v_processor": HunyuanVideoI2VProcessor,
    "stepvideo_i2v_processor": StepVideoI2VProcessor,
    "hunyuanvideo_15_i2v_processor": HunyuanVideo15I2VProcessor
}


class I2VProcessor:
    """
    The extra processor of the image to video task
    I2VProcessor is the factory class for all i2v_processor

    Args:
        config (dict): for Instantiating an atomic methods
    """

    def __init__(self, config):
        super().__init__()
        i2v_processor_cls = I2V_PROCESSOR_MAPPINGS[config["processor_id"]]
        self.processor = i2v_processor_cls(config)

    def get_processor(self):
        return self.processor
