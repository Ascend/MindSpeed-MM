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

"""Public FSDP2 environment-variable interface."""

from collections.abc import Mapping
from typing import Any

from . import envs as envs
from .envs import _manager, environment_variables
from .manager import (
    EnvironmentVariable,
    EnvironmentVariableDeprecationWarning,
    EnvironmentVariableManager,
    parse_bool,
)

_DEFAULT_NOT_PROVIDED = object()

__all__ = [
    "EnvironmentVariable",
    "EnvironmentVariableDeprecationWarning",
    "EnvironmentVariableManager",
    "descriptions",
    "environment_variables",
    "envs",
    "get",
    "get_spec",
    "is_set",
    "parse_bool",
]


def get(
    name: str,
    default: Any = _DEFAULT_NOT_PROVIDED,
    *,
    required: bool = False,
) -> Any:
    if default is _DEFAULT_NOT_PROVIDED:
        return _manager.get(name, required=required)
    return _manager.get(name, default, required=required)


def is_set(name: str, include_deprecated: bool = True) -> bool:
    return _manager.is_set(name, include_deprecated=include_deprecated)


def get_spec(name: str) -> EnvironmentVariable:
    return _manager.get_spec(name)


def descriptions() -> Mapping[str, str]:
    return _manager.descriptions()


def __getattr__(name: str) -> Any:
    if name in environment_variables:
        return _manager.get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(environment_variables))
