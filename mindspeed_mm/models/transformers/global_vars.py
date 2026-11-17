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

_FSDP2_EP_RANK = None
_FSDP2_EP_SIZE = None
_FSDP2_CHECK_MOE_FUNC = None
_FSDP2_EP_GROUP = None
_FSDP2_EP_FSDP_GROUP = None


def get_ep_rank():
    global _FSDP2_EP_RANK
    if _FSDP2_EP_RANK:
        return _FSDP2_EP_RANK
    else:
        return 0


def get_ep_size():
    global _FSDP2_EP_SIZE
    if _FSDP2_EP_SIZE:
        return _FSDP2_EP_SIZE
    else:
        return 1


def get_ep_group():
    global _FSDP2_EP_GROUP
    return _FSDP2_EP_GROUP


def get_ep_fsdp_group():
    global _FSDP2_EP_FSDP_GROUP
    return _FSDP2_EP_FSDP_GROUP


def get_check_moe_func():
    global _FSDP2_CHECK_MOE_FUNC
    if _FSDP2_CHECK_MOE_FUNC:
        return _FSDP2_CHECK_MOE_FUNC
    else:
        return lambda _: False


def set_ep_rank(ep_rank):
    global _FSDP2_EP_RANK
    _FSDP2_EP_RANK = ep_rank


def set_ep_size(ep_size):
    global _FSDP2_EP_SIZE
    _FSDP2_EP_SIZE = ep_size


def set_check_moe_func(fn):
    global _FSDP2_CHECK_MOE_FUNC
    _FSDP2_CHECK_MOE_FUNC = fn


def set_ep_group(group):
    global _FSDP2_EP_GROUP
    _FSDP2_EP_GROUP = group


def set_ep_fsdp_group(group):
    global _FSDP2_EP_FSDP_GROUP
    _FSDP2_EP_FSDP_GROUP = group
