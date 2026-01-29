from typing import Dict, Any
import time

import torch

from .constants import AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM
from .device import get_device_type, get_torch_device


def to_empty_if_needed(model, device: torch.device | str | int | None, recurse: bool = True):
    """Move the parameters and buffers to the specified device without copying storage if they are not already on that device.

    Args:
        module: The module whose parameters and buffers to (maybe) move.
        device: The desired device of the parameters and buffers in the module. If `None`, the default device is used.
        recurse: Whether parameters and buffers of submodules should be recursively moved to the specified device.

    Returns:
        The (maybe) moved module.
    """
    device = torch.empty((), device=device).device
    return model._apply(
        lambda t: torch.empty_like(t, device=device) if t.device != device else t,
        recurse=recurse,
    )


def move_to_device(batch: Dict[str, Any], float_dtype: str = None):
    new_batch = dict()
    for k, v in batch.items():
        if k in [AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM]:
            new_batch[k] = v.to(device=get_device_type())
        elif isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            new_batch[k] = v.to(device=get_device_type(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            new_batch[k] = [t.to(device=get_device_type(),
                            dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]
    return new_batch


def get_time(barrier=False):
    if barrier:
        torch.distributed.barrier()
    get_torch_device().synchronize()
    return time.time()