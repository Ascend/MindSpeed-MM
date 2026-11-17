# Copyright 2026 Huawei Technologies Co., Ltd.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# get_3d_sincos_pos_embed:   https://github.com/PKU-YuanGroup/Open-Sora-Plan
# get_3d_sincos_pos_embed_from_grid: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# get_2d_sincos_pos_embed: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# get_2d_sincos_pos_embed_from_grid: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# get_1d_sincos_pos_embed: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# get_1d_sincos_pos_embed_from_grid: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# VideoPatchEmbed2D: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# OverlapPatchEmbed2D: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# OverlapPatchEmbed3D: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# PatchEmbed3D: https://github.com/hpcaitech/Open-Sora/
# PatchEmbed2DP: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# PatchEmbed2D: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# TimestepEmbedder:    https://github.com/hpcaitech/Open-Sora/
# LabelEmbbedder:   https://github.com/hpcaitech/Open-Sora/
# --------------------------------------------------------
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

__all__ = [
    "LabelEmbedder", "TimestepEmbedder", "SizeEmbedder",
    "OverlapPatchEmbed2D", "OverlapPatchEmbed3D", "PatchEmbed3D", "VideoPatchEmbed2D", "PatchEmbed2D_3DsincosPE", "PatchEmbed2D",
    "PositionEmbedding2D", "get_1d_sincos_pos_embed", "get_2d_sincos_pos_embed", "get_3d_sincos_pos_embed", "get_nd_rotary_pos_embed",
    "TextProjection", "SingleTokenRefiner",
    "TimeStepEmbedding",
    "CombinedTimestepTextProjEmbeddings",
]

from .common_embeddings import LabelEmbedder, TimestepEmbedder, SizeEmbedder
from .patch_embeddings import (
    OverlapPatchEmbed2D,
    OverlapPatchEmbed3D,
    PatchEmbed3D,
    VideoPatchEmbed2D,
    PatchEmbed2D,
)
from .pos_embeddings import (
    PositionEmbedding2D,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
    get_nd_rotary_pos_embed
)
from .pos_patch_encoding import PatchEmbed2D_3DsincosPE
from .text_embeddings import (
    TextProjection,
    SingleTokenRefiner
)
from .time_embeddings import TimeStepEmbedding
from .time_text_embeddings import CombinedTimestepTextProjEmbeddings
