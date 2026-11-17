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

import torch

from mindspeed_mm.data.data_utils.constants import INPUT_MASK, MASKED_VIDEO
from mindspeed_mm.utils.mask_utils import MaskCompressor


class OpenSoraPlanI2VProcessor:
    """

    Args:
        config (dict): the processor config
        {
            "vae_scale_factor": [4, 8 ,8]
        }
    """

    def __init__(self, config):
        vae_scale_factor = config.get("vae_scale_factor", [4, 8, 8])
        ae_stride_t, ae_stride_h, ae_stride_w = vae_scale_factor
        self.mask_compressor = MaskCompressor(ae_stride_h=ae_stride_h, ae_stride_w=ae_stride_w, ae_stride_t=ae_stride_t)

    def __call__(self, vae_model, videos, video_latents, **kwargs):
        masked_video = kwargs.get(MASKED_VIDEO, None).to(videos.dtype)
        mask = kwargs.get(INPUT_MASK, None).to(videos.dtype)
        masked_video = vae_model.encode(masked_video)
        mask = self.mask_compressor(mask)

        return {MASKED_VIDEO: masked_video, INPUT_MASK: mask}
