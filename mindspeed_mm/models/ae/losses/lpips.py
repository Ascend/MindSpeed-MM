# Copyright 2026 Huawei Technologies Co., Ltd.
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros,
# Eli Shechtman, Oliver Wang. All rights reserved.
# This file is adapted from the PerceptualSimilarity (LPIPS) project:
#   https://github.com/richzhang/PerceptualSimilarity
# and is distributed under the BSD 2-Clause License:
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
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

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, perceptual_from_pretrained, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)


        self.load_state_dict(torch.load(perceptual_from_pretrained, map_location=torch.device("cpu")), strict=False)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, inputs, targets):
        in0_input, in1_input = (self.scaling_layer(inputs), self.scaling_layer(targets))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        layers = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for chn_idx in range(len(self.chns)):
            feats0[chn_idx], feats1[chn_idx] = normalize_tensor(outs0[chn_idx]), normalize_tensor(outs1[chn_idx])
            diffs[chn_idx] = (feats0[chn_idx] - feats1[chn_idx]) ** 2

        res = [spatial_average(layers[chn_idx].model(diffs[chn_idx]), keepdim=True) for chn_idx in range(len(self.chns))]

        val = res[0]
        for chn_idx in range(1, len(self.chns)):
            val += res[chn_idx]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features

        self.slice1 = self._build_slice(vgg_pretrained_features, 0, 4)
        self.slice2 = self._build_slice(vgg_pretrained_features, 4, 9)
        self.slice3 = self._build_slice(vgg_pretrained_features, 9, 16)
        self.slice4 = self._build_slice(vgg_pretrained_features, 16, 23)
        self.slice5 = self._build_slice(vgg_pretrained_features, 23, 30)

        self.N_slices = 5

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _build_slice(self, vgg_pretrained_features, start, end):
        res_slice = torch.nn.Sequential()
        for x in range(start, end):
            res_slice.add_module(str(x), vgg_pretrained_features[x])
        return res_slice

    def forward(self, X):
        h_relu1_2 = self.slice1(X)

        h_relu2_2 = self.slice2(h_relu1_2)

        h_relu3_3 = self.slice3(h_relu2_2)

        h_relu4_3 = self.slice4(h_relu3_3)

        h_relu5_3 = self.slice5(h_relu4_3)

        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
