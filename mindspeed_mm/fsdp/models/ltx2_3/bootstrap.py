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

"""Import aliases for LTX-2.3 source tree.

Copy ltx_core and ltx_trainer from LTX-2 repo to this directory:
  - LTX-2/packages/ltx-core/src/ltx_core -> mindspeed_mm/fsdp/models/ltx2_3/ltx_core/
  - LTX-2/packages/ltx-trainer/src/ltx_trainer -> mindspeed_mm/fsdp/models/ltx2_3/vendor/ltx_trainer/
"""

from __future__ import annotations

import importlib
import sys


def install_ltx2_3_aliases() -> None:
    """Map absolute imports (ltx_core, ltx_trainer) to vendored copies.

    LTX-2.3 source uses absolute imports. MindSpeed-MM places these under
    ``mindspeed_mm.fsdp.models.ltx2_3``, so we alias them via sys.modules.
    """
    aliases = {
        "ltx_core": ["mindspeed_mm.fsdp.models.ltx2_3.ltx_core"],
        "ltx_trainer": [
            "mindspeed_mm.fsdp.models.ltx2_3.vendor.ltx_trainer",
            "mindspeed_mm.fsdp.models.ltx2_3.vendor",
        ],
    }

    for public_name, vendored_names in aliases.items():
        if public_name in sys.modules:
            continue

        for vendored_name in vendored_names:
            try:
                sys.modules[public_name] = importlib.import_module(vendored_name)
                break
            except ImportError:
                continue
        else:
            raise ImportError(
                f"Failed to import {public_name}. Please copy it from LTX-2 repo "
                f"as per examples/ltx2_3/README.md. Expected paths: {', '.join(vendored_names)}"
            )
