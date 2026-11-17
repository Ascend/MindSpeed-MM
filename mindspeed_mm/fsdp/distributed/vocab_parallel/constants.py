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

"""Names shared by the vocabulary-parallel modules and the checkpoint code that stores their state."""

# Substring that identifies the host-resident embedding in ``model._no_placement_params``. Qwen3.8's entry is
# ``ple.ple_embedding.ngram_embedding.weight``; the hint is the part such entries share (see
# set_host_resident).
HOST_RESIDENT_HINT = "ngram_embedding"

# Suffix of a host embedding's checkpoint entries. The embedding is an ordinary parameter, not a DTensor, so every
# rank owns a row window of the same shape and a different content; the window index in the key keeps those
# entries apart (see vocab_checkpoint.py).
SHARD_KEY_SUFFIX = ".vp_shard_{:05d}_of_{:05d}"
