# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Tuple

import verl.third_party.vllm as vllm_sleep_level


def apply_npu_plugin():
    # Please make sure to apply the patches 
    # in the correct order so that they can 
    # work properly.

    # 2: vllm sleep level
    vllm_sleep_level.VLLM_SLEEP_LEVEL = 1

    # 3:
    # In verl, the driver process aggregates the computation results of workers via Ray. 
    # Therefore, after a worker completes its computation job, it will package the output 
    # using tensordict and transfer it to the CPU. Since the `to` operation of tensordict 
    # is non-blocking, when transferring data from a device to the CPU, it is necessary to 
    # ensure that a batch of data has been completely transferred before being used on the 
    # host; otherwise, unexpected precision issues may arise. Tensordict has already noticed 
    # this problem and fixed it.
    # However, the relevant modifications only cover CUDA and MPS devices and do not take effect 
    # for third-party devices such as NPUs. This patch fixes this issue, and the relevant 
    # modifications can be removed once the fix is merged into tensordict.

    from tensordict.base import TensorDictBase

    def _sync_all_patch(self):
        import torch
        from torch._utils import _get_available_device_type, _get_device_module
        try:
            from torch.compiler import is_compiling
        except ImportError:  # torch 2.0
            from torch._dynamo import is_compiling

        device_type = _get_available_device_type()
        if device_type is None:
            return

        if device_type == "cuda":
            if not is_compiling() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
        else:
            device_module = _get_device_module(device_type)
            device_module.synchronize()

    TensorDictBase._sync_all = _sync_all_patch