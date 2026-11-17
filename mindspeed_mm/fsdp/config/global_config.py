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

import os
from typing import Optional
from mindspeed_mm.fsdp.config.arguments.base_args import BaseArguments
from mindspeed_mm.fsdp.config.config_manager import ConfigManager


class _GlobalConfig:
    """Global configuration storage"""

    def __init__(self):
        self._current_config: Optional[BaseArguments] = None

    def set(self, config: BaseArguments):
        """Set global configuration"""
        self._current_config = config

    def get(self) -> Optional[BaseArguments]:
        """Get global configuration"""
        return self._current_config

    def clear(self):
        """Clear global configuration"""
        self._current_config = None


# Global configuration instance
_global_config = _GlobalConfig()


def set_global_config(config_manager: ConfigManager) -> BaseArguments:
    """
    Set global configuration

    Args:
        config_manager: Configuration manager

    Returns:
        TrainArguments: Global configuration object
    """
    # Load and parse configuration
    config = config_manager.load_and_parse()

    # Set to global configuration
    _global_config.set(config)

    if int(os.environ.get('RANK', '0')) == 0:
        print("[INFO] Global configuration has been set")

    return config


def get_args() -> Optional[BaseArguments]:
    """Get global arguments"""
    return _global_config.get()
