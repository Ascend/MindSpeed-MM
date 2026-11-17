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

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class ModelAdapter(ABC):
    @abstractmethod
    def preprocess(self, messages: List[dict]) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], generation_config) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, inputs: Dict[str, Any], outputs: Any) -> str:
        raise NotImplementedError
