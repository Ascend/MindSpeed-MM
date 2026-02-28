# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


logger = logging.getLogger(__name__)


def get_chunkloss_forward_fn():
    def chunkloss_forward(self, hidden_states: torch.Tensor, loss_func: callable):
        # Handle distributed tensor (DTensor) weights and biases by converting to local tensors.
        if isinstance(self.weight, DTensor):
            w = self.weight.to_local()
            if self.bias is not None:
                if not isinstance(self.bias, DTensor):
                    raise TypeError(
                        f"Expected bias to be a DTensor when weight is a DTensor, "
                        f"but got bias of type {type(self.bias)}."
                    )
                b = self.bias.to_local()
            else:
                b = None
        else:
            w = self.weight
            b = self.bias

        loss = loss_func(hidden_states, w, b)
        return loss
    return chunkloss_forward


def apply_chunkloss_module(module):
    chunkloss_forward = get_chunkloss_forward_fn()
    module.forward = types.MethodType(chunkloss_forward, module)


def get_chunkloss_module(modules, plan):
    chunkloss_module = getattr(modules, plan.apply_module, None)
    if chunkloss_module is None:
        raise ValueError(f"{plan.apply_module} does not exists, please set correct apply modules for chunk loss.")

    if not isinstance(chunkloss_module, torch.nn.Linear):
        raise ValueError(f"Chunk loss configuration error for module '{chunkloss_module}': "
                         f"expected torch.nn.Linear, got {repr(type(chunkloss_module))}. ")
    return chunkloss_module