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

class BaseGenEvalImpl:
    def __init__(self, dataset, pipeline, args):
        self.config = args
        self.dataset = dataset
        self.pipeline = pipeline

    def analyze_result(self):
        raise NotImplementedError("analyze_result() in BaseGenEvalImpl is required.")

    def __call__(self):
        raise NotImplementedError("__call__() in BaseGenEvalImpl is required.")
