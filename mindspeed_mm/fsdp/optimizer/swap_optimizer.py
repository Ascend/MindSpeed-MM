# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.optim import AdamW


class SwapOptimizerOperate():

    swap_to_device_stream = None
    swap_to_host_stream = None

    swap_to_device_events_map = {}
    swap_to_host_events_map = {}

    param_to_cpu_states_map = {}
    param_to_device_states_map = {}

    state_keys = ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']
    swap_state_keys = set()

    def __init__(self, mem_fraction_static=0.8, state_keys=None):
        if state_keys is not None:
            self.state_keys = state_keys

        if SwapOptimizerOperate.swap_to_device_stream is None:
            SwapOptimizerOperate.swap_to_device_stream = torch.accelerator.Stream()
            SwapOptimizerOperate.swap_to_host_stream = torch.accelerator.Stream()

        # create all parameters list for step
        self.param_to_group_map = {}

        for group in self.param_groups:
            for p in group['params']:
                self.param_to_group_map[p] = group

        self.opt_states_initialization()

         # predefine memory data for calculating swap_numel.
        self.mem_fraction_static = mem_fraction_static
        self.memory_data_initialization()

    def opt_states_initialization(self):
        for group in self.param_groups:
            for param in group["params"]:
                device_state_dtensor = self.state[param]
                device_state_tensor = {}
                cpu_state = {}

                amsgrad = self.param_to_group_map[param]['amsgrad']

                for key in self.state_keys:
                    if key == 'max_exp_avg_sq' and not amsgrad:
                        device_state_dtensor[key] = None
                        device_state_tensor[key] = None
                        cpu_state[key] = None
                    else:
                        self.swap_state_keys.add(key)
                        device_state_dtensor[key] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # convert dtensor to tensor
                        device_state_tensor[key] = device_state_dtensor[key].to_local()

                        cpu_state[key] = torch.empty_like(device_state_tensor[key], pin_memory=True, device='cpu')
                        cpu_state[key].copy_(device_state_tensor[key], non_blocking=True)

                        device_state_tensor[key].storage().resize_(0)

                self.param_to_device_states_map[param] = device_state_tensor
                self.param_to_cpu_states_map[param] = cpu_state

    def memory_data_initialization(self):
        params = list(self.param_to_device_states_map.keys())
        # Define how many bytes the optimizer state of each parameter occupies.
        self.byte_param = 4 if params[0].dtype == torch.float32 else 2
        self.total_memory = torch.accelerator.get_device_properties(torch.accelerator.current_device()).total_memory

    def get_swap_numel_from_unused_memory(self):
        # The available accelerator memory can be used to calculate
        # the number of parameters that can be transferred simultaneously.
        used_memory = torch.accelerator.memory_allocated()
        unused_memory = self.total_memory - used_memory

        # for example: the memory of exp_avg and exp_avg_sq of  is  (self.byte_param * 2)
        swap_numel = unused_memory * self.mem_fraction_static // (self.byte_param * len(self.swap_state_keys))
        return swap_numel

    def swap_all_to_host(self):
        for param in self.param_to_cpu_states_map.keys():
            self.swap_tensors_to_host(param)
        for param in self.param_to_cpu_states_map.keys():
            event = self.swap_to_host_events_map.get(param, None)
            if event is not None:
                torch.accelerator.current_stream().wait_event(event)
                self.swap_to_host_events_map[param] = None

    def swap_all_to_device(self):
        for param in self.param_to_cpu_states_map.keys():
            self.swap_tensors_to_device(param)
        for param in self.param_to_cpu_states_map.keys():
            event = self.swap_to_device_events_map.get(param, None)
            if event is not None:
                torch.accelerator.current_stream().wait_event(event)
                self.swap_to_device_events_map[param] = None

    def swap_tensors_to_device(self, param):
        cpu_state = self.param_to_cpu_states_map[param]

        if param in self.param_to_device_states_map:
            device_state = self.param_to_device_states_map[param]
            for key in self.state_keys:
                if device_state[key] is not None and device_state[key].storage().size() == 0:
                    device_state[key].storage().resize_(cpu_state[key].storage().size())
                    device_state[key].copy_(cpu_state[key], non_blocking=True)

        self.swap_to_device_events_map[param] = torch.accelerator.current_stream().record_event()

    def wait_swap_to_device_event(self, param):
        event = self.swap_to_device_events_map.get(param, None)
        if event is not None:
            torch.accelerator.current_stream().wait_event(event)
            self.swap_to_device_events_map[param] = None

    def swap_tensors_to_host(self, param):
        cpu_state = self.param_to_cpu_states_map[param]

        if param in self.param_to_device_states_map:
            device_state = self.param_to_device_states_map[param]
            for key in self.state_keys:
                if key in device_state and device_state[key] is not None and device_state[key].storage().size() != 0:
                    cpu_state[key].copy_(device_state[key], non_blocking=True)
                    device_state[key].storage().resize_(0)

        self.swap_to_host_events_map[param] = torch.accelerator.current_stream().record_event()

    def swap_batch_tensor_to_device(self, params_list, index):
        torch.accelerator.current_stream().wait_stream(self.swap_to_host_stream)
        swap_count = 0
        with torch.accelerator.stream(self.swap_to_device_stream):
            torch.accelerator.current_stream().wait_stream(self.swap_to_host_stream)
            self.swap_numel = self.get_swap_numel_from_unused_memory()
            while index < len(params_list) and (swap_count + params_list[index].to_local().numel() <= self.swap_numel):
                self.swap_tensors_to_device(params_list[index])
                swap_count += params_list[index].to_local().numel()
                index += 1

        if swap_count == 0:
            raise AssertionError(
                "OOM, the amount of data transferred for optimizer states "
                "from host to device is 0. You can try increasing "
                "mem_fraction_static."
            )
        return swap_count


class AdamWSwap(AdamW, SwapOptimizerOperate):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
        mem_fraction_static: Optional[float] = 0.8,
        state_keys: Optional[List[str]] = None,
    ):
        """
            This is a class that supports swapping optimizer states from the device to the host side
            to reduce peak GPU memory usage.

            During non-step phases, to avoid the additional GPU memory overhead from optimizer states,
            they are swapped from the device side to the host side. This operation is executed in the
            __init__ method of SwapOptimizerOperate.

            During the step execution, to further minimize peak memory usage, optimizer states are
            swapped from the host side back to the device side in batches, based on the available
            GPU memory. This operation is performed within the step method.

            Args:
                mem_fraction_static(float): Allocate available GPU memory * mem_fraction_static for swapping parameters
                    from host to device in batches.
                state_keys(list): Optimizer States That Need to be Swapped
                Other parameters: Refer to the documentation of the native torch.optim.AdamW parameters.

            Examples for AdamW:
                >>> AdamWSwap(params, mem_fraction_static=0.9, state_keys=['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'])
        """
        super().__init__(params,
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                foreach=False,
                maximize=maximize,
                capturable=False,
                differentiable=False,
                fused=True,)

        SwapOptimizerOperate.__init__(self, mem_fraction_static=mem_fraction_static)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if 'step' in group:
                group['step'] += 1
                if group['step'].is_cpu:
                    group['step'] = group['step'].cuda()
            else:
                group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.accelerator.current_device_index())

        swap_count = 0
        params_list = list(self.param_to_group_map.keys())

        for i, param in enumerate(params_list):
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError('AdamW does not support sparse gradients')

            group = self.param_to_group_map[param]
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            state = self.state[param]

            # State initialization
            if len(state) == 0:
                state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
            if 'max_exp_avg_sq' not in state:
                state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format) if amsgrad else None

            if swap_count == 0:
                swap_count = self.swap_batch_tensor_to_device(params_list, i)
            self.wait_swap_to_device_event(param)

            torch._fused_adamw_([param.to_local()], [param.grad.to_local()], [state['exp_avg'].to_local()], [state['exp_avg_sq'].to_local()], [state['exp_avg_sq'].to_local()] if amsgrad else [],
                                [group['step']], lr=group['lr'], beta1=beta1, beta2=beta2, weight_decay=group['weight_decay'],
                                eps=group['eps'], amsgrad=amsgrad, maximize=group['maximize'])

            working_stream = torch.accelerator.current_stream()
            with torch.accelerator.stream(self.swap_to_host_stream):
                self.swap_to_host_stream.wait_stream(working_stream)
                swap_count -= param.to_local().numel()
                self.swap_tensors_to_host(param)

        return loss
