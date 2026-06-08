# coding=utf-8
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
import os
from pathlib import Path
import shutil

from safetensors.torch import save_file

from checkpoint.common.converter import DcpConverter
from checkpoint.common.permissions import set_directory_permissions
from checkpoint.common.merge_dcp_to_hf import load_dcp_state_dict


class MOSSTTSConverter(DcpConverter):
    """
    A utility class to convert model checkpoints of MOSS-TTS between different formats,
    specifically between Hugging Face (HF) and torch-dcp (DCP) formats.

    Supports:
    - HF → DCP conversion
    - DCP → HF merging
    - Placeholder methods for megatron format and resharding operations.
    """

    dcp_prefix = "model."
    hf_prefix = ""

    def dcp_to_hf(
        self,
        load_dir: str = "mm_save_dir/release",     # Input: Directory containing DCP shards
        save_dir: str = "MOSS-TTS",         # Output: Directory to save merged HF model
        model_assets_dir: str = "MOSS-TTS",     # Reference: Original HF model dir (for config/tokenizer)
    ):
        """
        Merges torch-dcp shards and converts them back into standard Hugging Face format.

        This is typically used after training or inference in torch-dcp format to export
        a model that can be easily loaded with Hugging Face Transformers.
        """
        state_dict = load_dcp_state_dict(load_dir)
        shutil.copytree(model_assets_dir, save_dir, dirs_exist_ok=True)

        save_path = os.path.join(save_dir, "model.safetensors")
        save_file(state_dict, save_path)
        set_directory_permissions(Path(save_path))


    @staticmethod
    def hf_to_dcp():
        print("hf_to_dcp is not supported in MOSSTTSConverter.")
        raise NotImplementedError
