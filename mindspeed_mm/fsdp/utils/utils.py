from typing import Dict, Any
import time

import torch
from torch.distributed.tensor import DTensor, Replicate

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


def tensor_to_dtensor(t: torch.Tensor, device_mesh, placements):
    replicate = [Replicate() for _ in range(device_mesh.ndim)]
    ori_dtensor = DTensor.from_local(local_tensor=t, device_mesh=device_mesh, placements=replicate)
    new_dtensor = ori_dtensor.redistribute(device_mesh=device_mesh, placements=placements)
    return new_dtensor


def init_model_weights(model):
    post_init_modules = []

    def _pre_init_weights():
        # Find the parameters that cannot be initialized with Dtensor type, restore full_tensor, and then shard after initialization is complete
        for name, module in model.named_modules():
            setattr(module, "_is_initialized", False)
            if getattr(module, "_is_hf_initialized", False):
                module._is_hf_initialized = False
            if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                post_init_modules.append([name, module.weight.data.device_mesh, module.weight.data.placements])
                full_weight = torch.empty(module.weight.data.shape, device=module.weight.device)
                module.weight = torch.nn.Parameter(full_weight, requires_grad=module.weight.requires_grad)

    def _post_init_weights():
        if not post_init_modules:
            return

        for post_init_name, device_mesh, placements in post_init_modules:
            for name, module in model.named_modules():
                if name != post_init_name:
                    continue
                if isinstance(module, torch.nn.Embedding) and module.padding_idx is not None:
                    dtensor = tensor_to_dtensor(module.weight.data, device_mesh, placements)
                    module.weight = torch.nn.Parameter(dtensor, requires_grad=module.weight.requires_grad)

    _pre_init_weights()
    model.init_weights()
    _post_init_weights()


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


class Singleton(type):
    """Singleton metaclass to ensure only one instance of ParallelState exists."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance

        return cls._instances[cls]