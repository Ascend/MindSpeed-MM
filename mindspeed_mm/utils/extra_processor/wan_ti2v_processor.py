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

from mindspeed_mm.utils.extra_processor.wan_i2v_processor import WanVideoI2VProcessor


class WanVideoTI2VProcessor(WanVideoI2VProcessor):
    def __call__(self, vae_model, videos, first_frame, **kwargs):
        clip_features, vae_features = None, None
        # set vae tiling mode for i2v processor
        vae_model_tiling_state = vae_model.get_tiling_state()
        if self.enable_i2v_vae_encode_tiling != "auto" and self.enable_i2v_vae_encode_tiling != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, self.enable_i2v_vae_encode_tiling)

        first_frame_latents = vae_model.encode(first_frame.unsqueeze(2))

        # back vae tiling mode for video encode
        if vae_model.get_tiling_state() != vae_model_tiling_state:
            self.set_vae_tiling_state(vae_model, vae_model_tiling_state)

        return {
            "i2v_clip_feature": clip_features,
            "i2v_vae_feature": vae_features,
            "first_frame_latents": first_frame_latents
        }
