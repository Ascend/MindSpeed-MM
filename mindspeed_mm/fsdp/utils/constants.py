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

# some constants used for dataloader
# PrefetchGradAccDataLoader
GLOBAL_STEP_TOKEN_NUM = "global_step_token_num"  # tokens number per step per rank
AVG_PER_STEP_TOKEN_NUM = "avg_per_step_token_num"  # average tokens number per step per rank (gcs in consider)

# Memory report step, after optimizer state has been initialized.
MEMORY_REPORT_ITERATION = 2

# Constants used for checkpoint weight conversion.
LATEST_TXT = "latest_checkpointed_iteration.txt"
DCP_CHECKPOINT_VERSION = 3.0
DIR_MODE = 0o750
FILE_MODE = 0o640

# Projection suffixes used by MTP expert weight conversion.
HF_GATE_PROJ_SUFFIX = ".gate_proj.weight"
HF_UP_PROJ_SUFFIX = ".up_proj.weight"
DCP_DOWN_PROJ_SUFFIX = ".down_proj"
DCP_GATE_UP_PROJ_SUFFIX = ".gate_up_proj"
