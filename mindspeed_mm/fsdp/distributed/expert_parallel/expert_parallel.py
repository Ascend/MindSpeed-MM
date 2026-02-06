# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types
from functools import partial
from typing import Callable

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, DTensor, Replicate, distribute_tensor, distribute_module

from mindspeed.fsdp.distributed.expert_parallel.dispatcher import get_experts_forward_fn
from mindspeed.fsdp.distributed.expert_parallel.dispatcher_mc2 import get_experts_forward_mc2_fn
from mindspeed.fsdp.parallel_engine_config import EPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match


logger = logging.getLogger(__name__)


def expert_parallelize_modules(modules: torch.nn.Module, ep_mesh: DeviceMesh, plan: EPPlanConfig):
    ep_modules = get_ep_modules(modules, plan)

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for module in ep_modules:
        distribute_experts_module(module, ep_mesh)

        # replace forward with ep forward
        experts_forward_fn = get_experts_forward_fn_for_qwen(ep_group)
        module.forward = types.MethodType(experts_forward_fn, module)

    return modules


def get_ep_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    ep_modules = []
    for plan_name in plan.apply_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Parallel]: Apply ep to module <{name}>')
                ep_modules.append(module)
    if len(ep_modules) == 0:
        raise RuntimeError(f'[Expert Parallel] No module named {plan} or not be ModuleList')
    return ep_modules


def prepare_distribute_input_fn(module, inputs, device_mesh):
    inputs = list(inputs)
    for idx, input_tensor in enumerate(inputs):
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (Replicate(),), run_check=False)
            inputs[idx] = input_tensor
    return *inputs,


def prepare_distribute_output_fn(module, outputs, device_mesh):
    return outputs.to_local()


def distribute_expert_weight(module_name, module, ep_mesh):
    for name, param in module.named_parameters(recurse=False):
        dist_param = torch.nn.Parameter(distribute_tensor(param, ep_mesh, [Shard(0)]))
        module.register_parameter(name, dist_param)

    for name, children_module in module.named_children():
        distribute_expert_weight(name, children_module, ep_mesh)


def distribute_experts_module(module: torch.nn.Module, ep_mesh: DeviceMesh):
    return distribute_module(module=module, device_mesh=ep_mesh, partition_fn=distribute_expert_weight,)
                             # input_fn=prepare_distribute_input_fn, output_fn=prepare_distribute_output_fn)


def get_dispatcher_fn(dispatcher, ep_group):
    forward_fn = None
    if isinstance(dispatcher, Callable):
        forward_fn = partial(dispatcher, ep_group)
    elif isinstance(dispatcher, str):
        if dispatcher == 'eager':
            forward_fn = get_experts_forward_fn(ep_group, fused=False)
        elif dispatcher == 'fused':
            forward_fn = get_experts_forward_fn(ep_group, fused=True)
        elif dispatcher == 'mc2':
            forward_fn = get_experts_forward_mc2_fn(ep_group)

    if forward_fn is None:
        raise RuntimeError(f'Unsupported dispatcher {dispatcher}.')

    return forward_fn


def get_grad_division_hook(param, ep_size):
    def hook(*unused):
        return param.grad.mul_(1 / ep_size)

    return hook


def apply_grad_division_hook(module, ep_size):
    for param in module.parameters():
        if param.requires_grad:
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(get_grad_division_hook(param, ep_size))


def get_experts_forward_fn_for_qwen(ep_group, fused=True):
    from .fused_ep import fused_ep_forward

    def experts_forward(self, hidden_states: torch.Tensor, routing_weights: torch.Tensor, router_indices: torch.Tensor):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        hidden_states = fused_ep_forward(
            self.num_experts,
            routing_weights,
            router_indices,
            hidden_states,
            fc1_weight=gate_up_proj,
            fc2_weight=down_proj,
            ep_group=ep_group
        )
        hidden_states = hidden_states.view(batch_size, -1, self.hidden_size)
        return hidden_states

    return experts_forward