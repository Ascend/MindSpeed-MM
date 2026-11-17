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

"""MagiHuman FSDP2 data plugin package (issue #357).

Plugin root for the MagiHuman precomputed dataset. The YAML
``training.plugin`` list names this directory
(``mindspeed_mm/fsdp/data/datasets/magihuman``); ``import_plugin``
(``mindspeed_mm/fsdp/utils/register.py``) then ``walk_packages`` over it and
imports every submodule, which executes the ``@data_register.register(
"magihuman_precomputed")`` decorator in ``magihuman_precomputed_dataset.py``.

Mirrors the ``ltx2`` datasets package layout (which is importable the same way).
"""
