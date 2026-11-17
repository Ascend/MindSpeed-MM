# Copyright 2024 Huawei Technologies Co., Ltd.
# Copyright 2024 Black Forest Labs
# SPDX-License-Identifier: Apache-2.0
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
from torch import Tensor, nn

from mindspeed_mm.models.ae.movqvae import Encoder
from mindspeed_mm.models.common.checkpoint import load_checkpoint
from mindspeed_mm.models.common.distrib import DiagonalGaussianDistribution


class FluxVae(nn.Module):
    def __init__(
        self,
        resolution=256,
        in_channels=3,
        ch=128,
        out_ch=3,
        ch_mult=None,
        num_res_blocks=2,
        z_channels=16,
        scale_factor=0.3611,
        shift_factor=0.1159,
        attn_resolutions=None,
        use_sdp_attention=True,
        from_pretrained: str = None,
        **kwargs
    ):
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4, 4]
        if attn_resolutions is None:
            attn_resolutions = [0]
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=ch,
            out_ch=out_ch,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
            attn_resolutions=attn_resolutions,
            use_sdp_attention=use_sdp_attention,
        )
        self.scale_factor = scale_factor
        self.shift_factor = shift_factor
        if from_pretrained is not None:
            load_checkpoint(self, from_pretrained)

    def encode(self, x: Tensor = None, **kwargs) -> Tensor:
        if x is None:
            x = kwargs.get('images') or kwargs.get('padded_images')
        z = self.encoder(x)
        posterior = DiagonalGaussianDistribution(z)
        z = posterior.mode()
        z = self.scale_factor * (z - self.shift_factor)
        return z
