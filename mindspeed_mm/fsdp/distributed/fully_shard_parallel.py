# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
from typing import Set, Any

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh

from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

from mindspeed_mm.fsdp.params.parallel_args import FSDPPlanConfig

logger = logging.getLogger(__name__)


def fully_shard_parallel_modules(model: torch.nn.Module, fsdp_mesh: DeviceMesh, fsdp_plan: FSDPPlanConfig, **kwargs):
    """
    Apply Fully Sharded Data Parallelism (FSDP) to specified modules in the model.
    
    Args:
        model: The neural network model to apply FSDP to.
        fsdp_mesh: Device mesh defining the FSDP process group.
        fsdp_plan: Configuration specifying which modules to apply FSDP to and mixed precision settings.
        **kwargs: Additional keyword arguments.
    
    Returns:
        The model with FSDP applied to specified modules.
    """
    # Get modules and parameters that should be ignored for FSDP
    ignored_modules, ignored_params = get_ignored_modules(model, fsdp_plan)
    # Get modules that should have FSDP applied
    fsdp_modules = get_fsdp_modules(model, fsdp_plan, ignored_modules)

    # Configure mixed precision if enabled
    config = {'mesh': fsdp_mesh, 'ignored_params': ignored_params}
    if fsdp_plan.enable_mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        config["mp_policy"] = mp_policy
    # Apply FSDP to specific child modules first
    for module in fsdp_modules:
        fully_shard(module, **config)
    # Apply FSDP to the entire model
    fully_shard(model, **config)

    return model


def _post_order_traverse(model: torch.nn.Module, parent_path: str = ""):
    """
    Perform post-order traversal of model submodules.
    
    Post-order traversal ensures child modules are visited before their parents,
    which is important for FSDP to properly handle nested modules.
    
    Args:
        model: The model to traverse.
        parent_path: The path to the current module in the hierarchy.
    
    Yields:
        Tuple of (module_path, module) for each module in the model.
    """
    for name, child in model.named_children():
        child_path = f"{parent_path}.{name}" if parent_path else name
        yield from _post_order_traverse(child, child_path)
    yield parent_path, model


def get_fsdp_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig, ignored_modules: Set[str]) -> dict[Any, Any]:
    fsdp_modules = []
    # Traverse all modules in the model
    if fsdp_plan.apply_modules:
        for name, module in _post_order_traverse(model):
            # Check if module matches any pattern in the FSDP plan
            for pattern in fsdp_plan.apply_modules:
                if module_name_match(pattern, name) and name not in ignored_modules:
                    print_rank(logger.debug, f'[FSDP2]: Apply fsdp2 to module <{name}>')
                    fsdp_modules.append(module)
        # Ensure at least one module matches the FSDP plan
        if len(fsdp_modules) == 0:
            raise RuntimeError(f'[FSDP2] No module named {fsdp_plan.apply_modules}.')
    return fsdp_modules


def get_ignored_modules(model: torch.nn.Module, fsdp_plan: FSDPPlanConfig):
    ignored_modules = set()
    ignored_params = set()
    if fsdp_plan.ignored_modules is None:
        return ignored_modules, ignored_params
    for name, module in model.named_modules():
        for pattern in fsdp_plan.ignored_modules:
            if module_name_match(pattern, name):
                print_rank(logger.debug, f'[FSDP2]: Ignored module to apply fsdp2 <{name}>')
                ignored_modules.add(name)
                ignored_params.update(list(module.parameters(recurse=True)))
    return ignored_modules, ignored_params