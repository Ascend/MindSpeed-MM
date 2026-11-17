# Copyright 2024 Huawei Technologies Co., Ltd.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

"""VAE global variables."""

_GLOBAL_AE_ARGS = None


def _ensure_var_is_initialized(var, name):
    """Make sure the input variable is not None."""
    if var is None:
        raise ValueError('{} is not initialized.'.format(name))


def _ensure_var_is_not_initialized(var, name):
    """Make sure the input variable is not None."""
    if var is not None:
        raise ValueError('{} is already initialized.'.format(name))


def set_ae_global_variables(args):
    global _GLOBAL_AE_ARGS

    _ensure_var_is_not_initialized(_GLOBAL_AE_ARGS, "args")
    _GLOBAL_AE_ARGS = args


def get_ae_args():
    """Return arguments."""
    _ensure_var_is_initialized(_GLOBAL_AE_ARGS, "args")
    return _GLOBAL_AE_ARGS
