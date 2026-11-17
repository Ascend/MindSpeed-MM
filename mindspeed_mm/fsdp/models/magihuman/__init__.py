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

"""MagiHuman FSDP2 plugin package.

This is the plugin root for the MagiHuman 15B single-stream audio+video joint
denoiser (issue #357 migration). Layout mirrors the ``ltx2`` plugin:

    mindspeed_mm/fsdp/models/magihuman/
        magihuman_fsdp2/            # hand-written FSDP2 wrapper + patches (committed)
            modeling_magihuman.py  # @model_register.register("magihuman")
            modified.py            # FSDP2/recompute-friendly forward rewrite
            npu_patch.py           # CUDA attention / custom-op -> NPU stubs
        inference/                 # VENDORED upstream package, NOT committed (see plan)

The `training.plugin` YAML list names the directory
``mindspeed_mm/fsdp/models/magihuman/magihuman_fsdp2`` (not this root), so
`import_plugin` walks that subpackage and executes the `@model_register`
decorator in ``modeling_magihuman.py``.
"""
