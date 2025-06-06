# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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


import torch.nn as nn

from .vae import VideoAutoencoderKL, VideoAutoencoder3D, VideoAutoencoderKLCogVideoX
from .casualvae import CausalVAE

AE_MODEL_MAPPINGS = {"vae": VideoAutoencoderKL,
                     "vae3D": VideoAutoencoder3D,
                     "casualvae": CausalVAE,
                     "cogvideox": VideoAutoencoderKLCogVideoX}


class AEModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AE_MODEL_MAPPINGS[config.model_id](**config.to_dict())

    def get_model(self):
        return self.model

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        raise NotImplementedError("forward function is not implemented")
