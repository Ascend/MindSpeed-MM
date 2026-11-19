# Copyright 2026 Huawei Technologies Co., Ltd.
# Derived from DiffSynth-Studio commit 0ad850953156c028c6334ea8c17c50032a1ef169.
# Licensed under Apache License 2.0.
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

# MiniMax H3 integration constants, shared by the model code and the pipeline.
# ---------------------------------------------------------------------------
# Canvas / spatial policy
# ---------------------------------------------------------------------------
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32
# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2

# ---------------------------------------------------------------------------
# Condition noise augmentation
# ---------------------------------------------------------------------------
# Each value drives both the mixed condition tensor and its AdaLN timestep, so
# moving off the trained anchor makes the condition out-of-distribution.
IMGVID_COND_NOISE_AUG = 0.999
AUDIO_COND_NOISE_AUG = 1.0
