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

"""Fused HC forward and backward. First-order gradients only.

Requires cann_ops_transformer_custom and its matching ACLNN package.
Forward caches and first-order backward are managed by the custom operator.
"""

# Reserve the namespace for CANN's built-in operators (e.g. QSA) first. The
# custom package then uses its existing FRAGMENT fallback instead of owning DEF.
# This also permits standalone installations without the built-in Python package.
try:
    import cann_ops_transformer.op_builder.builder  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name not in ("cann_ops_transformer", "cann_ops_transformer.op_builder",
                        "cann_ops_transformer.op_builder.builder"):
        raise


def qhc_ascendc(
    x, w_down, w_up, w_gate=None, gamma=None, channels=4, eps=1e-6,
):
    """Fused forward and backward for x/weights/optional gamma.

    Passing w_gate returns (mixed, inject); omitting it returns mixed.
    Supports first-order gradients, including losses using only one output.
    """
    if gamma is not None and (gamma.ndim != 1 or gamma.numel() != x.shape[-1]):
        raise ValueError('gamma must have shape [C*H], matching x.shape[-1]')
    from cann_ops_transformer_custom.ops.mhc.text_gated_residual import (
        text_gated_residual,
    )
    return text_gated_residual(
        x, w_down, w_up, w_gate=w_gate, gamma=gamma,
        channels=channels, eps=eps,
    )
