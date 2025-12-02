# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
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
import os
import importlib
from functools import lru_cache
from typing import Type, Any, Dict, List, Optional
from einops import rearrange

import torch
import torch.distributed
import numpy as np

from megatron.core import mpu
from megatron.training import get_args


class Registry:
    """A generic class registry system that automatically uses class names as registration keys.
    
    Features:
    - Automatic registration using class names
    - Prohibition of manual name specification
    - Class name conflict detection
    """
    
    _REGISTRY: Dict[str, Type[Any]] = {}
    """Internal registry storage mapping class names to their corresponding class objects"""

    @classmethod
    def register(cls, target_class: Type[Any]) -> Type[Any]:
        """Class decorator for automatic registration using the class name.
        
        Args:
            target_class: Target class to be registered
            
        Returns:
            The original class object to preserve class definition
            
        Raises:
            ValueError: If class name is already registered
        """
        class_name = target_class.__name__
        
        if class_name in cls._REGISTRY:
            existing = cls._REGISTRY[class_name]
            raise ValueError(
                f"Class name conflict: '{class_name}' already registered by {existing}, "
                f"attempting to register: {target_class}"
            )
            
        cls._REGISTRY[class_name] = target_class
        return target_class

    @classmethod
    def get_class(cls, name: str) -> Type[Any]:
        """Retrieve a registered class by its name.
        
        Args:
            name: Name of the class to retrieve
            
        Returns:
            The registered class object
            
        Raises:
            ValueError: If the class is not found in registry
        """
        if name not in cls._REGISTRY:
            available = list(cls._REGISTRY.keys())
            raise ValueError(
                f"Class '{name}' not found in registry. Available classes: {available}"
            )
        return cls._REGISTRY[name]


@lru_cache
def is_npu_available():
    """Checks if `torch_npu` is installed and potentially if a NPU is in the environment"""
    if importlib.util.find_spec("torch_npu") is None:
        return False
    import torch_npu
    try:
        # Will raise a RuntimeError if no NPU is found
        _ = torch.npu.device_count()
        return torch.npu.is_available()
    except RuntimeError:
        return False


def get_device(device="npu"):
    """
    only support npu and cpu device, default npu.
    device format: cpu, npu, or npu:0
    """
    if isinstance(device, torch.device):
        return device
    device = device.lower().strip()
    if device == "cpu":
        return torch.device(device)

    device_infos = device.split(":")
    device_name = device_infos[0]
    if device_name == "npu":
        if is_npu_available():
            if len(device_infos) == 1:
                return torch.device(device_name)
            if len(device_infos) == 2:
                device_id = int(device_infos[1])
                num_devices = torch.npu.device_count()
                if device_id < num_devices:
                    return torch.device(f"{device_name}:{device_id}")
                else:
                    raise ValueError(f"device_id: {device_id} must less than device nums: {num_devices}")
        else:
            raise RuntimeError("NPU environment is not available")
    raise ValueError("only support npu and cpu device. device format: cpu, npu, or npu:0")


def get_dtype(dtype):
    """return torch type according to the string"""
    if isinstance(dtype, torch.dtype):
        return dtype
    dtype_mapping = {
        "int32": torch.int32,
        "float64": torch.float64,
        "float32": torch.float32,
        "float16": torch.float16,
        "fp32": torch.float32,
        "fp16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype not in dtype_mapping:
        raise ValueError("Unsupported data type")
    dtype = dtype_mapping[dtype]
    return dtype


def video_to_image(func):
    def wrapper(self, x, *args, **kwargs):
        if x.dim() == 5:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = func(self, x, *args, **kwargs)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        else:
            x = func(self, x, *args, **kwargs)
        return x
    return wrapper


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) or isinstance(t, list) else ((t,) * length)


def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(1.702 * x)


def gelu_tanh(inp: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(inp, approximate="tanh")


def set_modules_requires_grad(modules, requires_grad):
    for module in modules:
        module.requires_grad_(requires_grad)


def save_ae_checkpoint(
    epoch,
    current_step,
    optimizer_state,
    state_dict,
    scaler_state,
    sampler_state,
    checkpoint_dir,
    filename="checkpoint.ckpt",
    ema_state_dict=None,
):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "current_step": current_step,
            "optimizer_state": optimizer_state,
            "state_dict": state_dict,
            "ema_state_dict": ema_state_dict,
            "scaler_state": scaler_state,
            "sampler_state": sampler_state,
        },
        filepath,
    )
    return filepath


_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_SIZE = None


def is_context_parallel_initialized():
    if _CONTEXT_PARALLEL_GROUP is None:
        return False
    else:
        return True


def set_context_parallel_group(size, group):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE
    _CONTEXT_PARALLEL_GROUP = group
    _CONTEXT_PARALLEL_SIZE = size


def initialize_context_parallel(context_parallel_size):
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_SIZE

    if _CONTEXT_PARALLEL_GROUP is not None:
        raise AssertionError("Context parallel group is already initialized")
    _CONTEXT_PARALLEL_SIZE = context_parallel_size

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    for i in range(0, world_size, context_parallel_size):
        ranks = range(i, i + context_parallel_size)
        group = torch.distributed.new_group(ranks)
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            break


def get_context_parallel_group():
    if _CONTEXT_PARALLEL_GROUP is None:
        raise AssertionError("Context parallel group is not initialized")

    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_world_size():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    return _CONTEXT_PARALLEL_SIZE


def get_context_parallel_rank():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    rank = torch.distributed.get_rank()
    cp_rank = rank % _CONTEXT_PARALLEL_SIZE
    return cp_rank


def get_context_parallel_group_rank():
    if _CONTEXT_PARALLEL_SIZE is None:
        raise AssertionError("Context parallel size is not initialized")

    rank = torch.distributed.get_rank()
    cp_group_rank = rank // _CONTEXT_PARALLEL_SIZE

    return cp_group_rank


class IsNotValidError(Exception):
    def __init__(self, error_message=None):
        self.error_message = error_message
        super().__init__(error_message or "Expression is not valid")

    def __str__(self):
        return self.error_message or "Expression is not valid"


def ensure_valid(expression, error_message=None):
    if not expression:
        raise IsNotValidError(error_message)


def dist_sort(image_num_list):
    # calculate the average
    world_size = len(image_num_list)
    total_images = sum(image_num_list)
    avg = total_images // world_size
    remainder = total_images % world_size
    more_rank = avg + 1
    target = [avg] * world_size
    index_list = [[] for _ in range(world_size)]
    index = 0
    # when the number of images is greater than the average, as many as possible are taken as avg+1, and the rest are sent.
    for i in range(world_size):
        index_list[i].extend([j for j in range(index, index + image_num_list[i])])
        index += image_num_list[i]
    for index, image in enumerate(image_num_list):
        if remainder and image > avg:
            target[index] = more_rank
            remainder -= 1
    index = image_num_list.argsort()
    for i in range(remainder):
        target[index[i]] = more_rank
    # transfer matrix    
    transfer = np.zeros((world_size, world_size), dtype=int)
    # greedy strategy allocation
    surplus = []
    deficit = []
    for i in range(world_size):
        if image_num_list[i] > target[i]:
            surplus.append(i)
        elif image_num_list[i] < target[i]:
            deficit.append(i)
    while surplus and deficit:
        s = surplus[-1]
        d = deficit[-1]
        give = min(image_num_list[s] - target[s], target[d] - image_num_list[d])
        image_num_list[s] -= give
        image_num_list[d] += give
        transfer[s][d] += give
        if image_num_list[s] == target[s]:
            surplus.pop()
        if image_num_list[d] == target[d]:
            deficit.pop()
    return transfer, target


def unwrap_single(x: list):
    while isinstance(x, list) and len(x) == 1:
        x = x[0]
    return x


class EncoderBalanceComm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group, transfer=None, nopadding=False, skip=False):
        ctx.no_bk = transfer is None
        rank = torch.distributed.get_rank(group=group)
        ctx.shape = list(input_tensor.shape)
        if transfer is not None:
            transfer, target = transfer
            input_tensor = input_tensor[:target[rank]].contiguous() if not nopadding else input_tensor
        image_shape = input_tensor.shape
        ctx.shape[1] -= input_tensor.shape[1]
        image_num = image_shape[0]
        ishape = image_shape[1:]
        world_size = torch.distributed.get_world_size(group)
        ctx.group = group
        ctx.rank = rank
        ctx.world_size = world_size
        if transfer is None:
            shape_input = torch.tensor([image_num], dtype=torch.int8).cuda()
            shape_output = torch.empty([world_size, *shape_input.shape], dtype=shape_input.dtype).cuda()
            # gather image num
            torch.distributed._all_gather_base(shape_output, shape_input, group=group)
            image_num_list = shape_output.cpu().numpy().reshape(-1)
            transfer, target = dist_sort(image_num_list)
        ctx.transfer = [transfer.T, target]
        if skip:
            return input_tensor, [transfer.T, target]
        if np.sum(transfer) == 0:
            # do not need to balance
            if ctx.no_bk:
                return input_tensor, [transfer.T, target]
            else:
                return input_tensor
        send_img_num = sum(transfer[rank])
        # get images to comm
        send_img = list(
            torch.split(
                input_tensor[image_num - send_img_num:].contiguous(),
                transfer[rank].tolist(),
                dim=0)
            )

        output = input_tensor[:image_num - send_img_num]
        transfer = transfer.T
        recv = torch.empty_like(input_tensor).resize_([sum(transfer[rank]), *ishape])
        recv = list(torch.split(recv, transfer[rank].tolist(), dim=0))
        torch.distributed.all_to_all(recv, send_img, group=group)
        recv = torch.cat([output] + recv, dim=0)
        if not ctx.no_bk:
            return recv
        return recv, [transfer, target]
 
    @staticmethod
    def backward(ctx, grad_output):
        if ctx.no_bk or np.sum(ctx.transfer[0]) == 0:
            return grad_output, None, None, None, None
        else:
            data = EncoderBalanceComm.apply(grad_output, ctx.group, ctx.transfer, True)
            return data, None, None, None, None
    

def change_tensor_layout(tensor, src_layout, dst_layout, batch_size=None):
    """
    Transforms the input tensor from the source layout (src_layout) to the target layout (dst_layout).

    Args:
        tensor (torch.Tensor): The input tensor.
        src_layout (str): The source layout, e.g., "sbh" or "bsh".
        dst_layout (str): The target layout, e.g., "sbnd" or "tnd".
    
    Returns:
        torch.Tensor: The tensor with the transformed layout.
    """
    src_layout = src_layout.lower()
    dst_layout = dst_layout.lower()
    
    if src_layout == dst_layout:
        return tensor
    key = (src_layout, dst_layout)
    layout_mappings = {
        # input layout change to `sbh`
        ("bsh", "sbh"): lambda x: rearrange(x, "b s h -> s b h"),
        # flash attention input layout change
        ("sbnd", "sbh"): lambda x: rearrange(x, "s b n d -> s b (n d)"),
        ("sbnd", "bsnd"): lambda x: rearrange(x, "s b n d -> b s n d"),
        ("sbnd", "bnsd"): lambda x: rearrange(x, "s b n d -> b n s d"),
        ("sbnd", "tnd"): lambda x: rearrange(x, "s b n d -> (s b) n d"),
        # output layout change to `sbh`
        ("bsnd", "sbh"): lambda x: rearrange(x, "b s n d -> s b (n d)"),
        ("bnsd", "sbh"): lambda x: rearrange(x, "b n s d -> s b (n d)"),
        ("tnd", "sbh"): lambda x: rearrange(x, "(s b) n d -> s b (n d)", b=batch_size),
        # output layout change to `bsh`
        ("sbh", "bsh"): lambda x: rearrange(x, "s b h -> b s h"),
        ("bsnd", "bsh"): lambda x: rearrange(x, "b s n d -> b s (n d)"),
        ("bnsd", "bsh"): lambda x: rearrange(x, "b n s d -> b s (n d)"),
        ("tnd", "bsh"): lambda x: rearrange(x, "(s b) n d -> b s (n d)", b=batch_size),
    }

    if key in layout_mappings:
        if isinstance(tensor, torch.Tensor):
            return layout_mappings[key](tensor)
        elif isinstance(tensor, (list, tuple)):
            return [layout_mappings[key](t) for t in tensor]
        else:
            raise ValueError(f"Unsupported input type {type(tensor)}")
    else:
        raise ValueError(f"Unsupported layout conversion from {src_layout} to {dst_layout}!")


def reorder_output(attn_output, cp_rank, cp_size, cp_group, dim=0):
    index_this_rank = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], dtype=torch.int8, device=attn_output.device)
    index_list = [torch.zeros_like(index_this_rank, device=attn_output.device) for _ in range(cp_size)]
    torch.distributed.all_gather(index_list, index_this_rank, group=cp_group)

    index_list = [int(item) for item in list(torch.concat(index_list))]
    index_map = {element: idx for idx, element in enumerate(index_list)}
    target = [i for i in range(len(index_list))]
    target_list = [index_map[element] for element in target]
    
    chunks = torch.chunk(attn_output, chunks=len(target_list), dim=dim)
    reordered_chunks = [chunks[idx] for idx in target_list]
    attn_output = torch.concat(reordered_chunks, dim=dim)
    return attn_output


def _gather(
    input_: torch.Tensor,
    pg: torch.distributed.ProcessGroup,
    dim: int = -1,
    gather_size: List = None
):
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(group=pg)

    if input_.device.type not in ["cpu", "npu"]:
        raise AssertionError(f"Only support cpu and npu device, got {input_.device}")

    if world_size == 1:
        return input_
    
    if gather_size is not None:
        tensor_list = []
        tensor_shape_base = input_.size()
        for i in range(world_size):
            tensor_shape = list(tensor_shape_base)
            tensor_shape[dim] = gather_size[i]
            tensor_list.append(torch.empty(tensor_shape, dtype=input_.dtype, device=input_.device))
    else:
        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

    torch.distributed.all_gather(tensor_list, input_, group=pg)

    output = torch.cat(tensor_list, dim=dim).contiguous()
    return output


class _SplitForwardGatherBackWardWithMegatronCP(torch.autograd.Function):
    '''
    Split the input tensor in the forward pass and gather the gradients in the backward pass. 
    It will be implemented in Mindspeed in the future.
    '''
    @staticmethod
    def forward(ctx, val, cp_rank, cp_size, seq_dim, cp_group=None):
        val = val.view(
            *val.shape[0:seq_dim],
            2 * cp_size,
            val.shape[seq_dim] // (2 * cp_size),
            *val.shape[(seq_dim + 1):],
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
        val = val.index_select(seq_dim, index)
        val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2):])

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim

        return val
        
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = {}
        grad_input = _gather(grad_output, ctx.cp_group, dim=ctx.seq_dim) / ctx.cp_size
        grad_input = reorder_output(grad_input, ctx.cp_rank, ctx.cp_size, ctx.cp_group, dim=ctx.seq_dim)
        return grad_input, None, None, None, None


def split_forward_gather_backward_with_megatron_cp(
        input_: torch.Tensor,
        process_group: torch.distributed.ProcessGroup,
        dim: int = 0
) -> torch.Tensor:
    cp_size = torch.distributed.get_world_size(group=process_group)
    cp_rank = torch.distributed.get_rank(group=process_group)

    return _SplitForwardGatherBackWardWithMegatronCP.apply(input_, cp_rank, cp_size, dim, process_group)


class _GatherForwardSplitBackWardWithMegatronCP(torch.autograd.Function):
    '''
    Split the input tensor in the forward pass and gather the gradients in the backward pass with megatron cp(Ring Attention)
    It will be implemented in Mindspeed in the future.
    '''
    @staticmethod
    def forward(ctx, val, cp_rank, cp_size, seq_dim, cp_group=None):
        # Step 1: All-gather shards from all CP ranks along the sequence dimension
        val = _gather(val, cp_group, dim=seq_dim)
        # Step 2: Reorder the gathered tensor
        val = reorder_output(val, cp_rank, cp_size, cp_group, dim=seq_dim)

        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.seq_dim = seq_dim

        return val
        
    @staticmethod
    def backward(ctx, grad_output):
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        seq_dim = ctx.seq_dim

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1):],
        ) * cp_size  # Scale gradients up by cp_size
        # Select the two chunks that belong to the current rank:
        # - One from the forward direction (index = cp_rank)
        # - One from the backward direction (index = 2*cp_size - cp_rank - 1)
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=grad_output.device)
        grad_output = grad_output.index_select(seq_dim, index)

        # Collapse the two selected chunks back into a single contiguous local sequence
        grad_input = grad_output.view(*grad_output.shape[0:seq_dim], -1, *grad_output.shape[(seq_dim + 2):])
        
        return grad_input, None, None, None, None


def gather_forward_split_backward_with_megatron_cp(
    input_: torch.Tensor,
    process_group: torch.distributed.ProcessGroup,
    dim: int = 0
) -> torch.Tensor:
    cp_size = torch.distributed.get_world_size(group=process_group)
    cp_rank = torch.distributed.get_rank(group=process_group)

    return _GatherForwardSplitBackWardWithMegatronCP.apply(input_, cp_rank, cp_size, dim, process_group)


def compute_token_level_loss(loss_dict):
    """Token level loss function"""
    args = get_args()

    if args.context_parallel_size > 1:
        loss = loss_dict['loss']
        total_tokens = loss_dict["token_nums"]
        loss = torch.cat([loss.sum().view(1), total_tokens.sum().view(1)])
    else:
        loss = loss_dict['loss']
        loss_mask = loss_dict['loss_mask']
        loss_mask = loss_mask.view(-1).float()
        total_tokens = loss_mask.sum()
        if loss.view(-1).shape == loss_mask.shape:
            loss = torch.cat([torch.sum(loss.view(-1) * loss_mask).view(1), total_tokens.view(1)])
        else:
            loss = torch.cat([loss.view(1), total_tokens.view(1)])

    # Reduce loss for logging.
    reporting_loss = loss.clone().detach()
    loss[0] = loss[0] / mpu.get_context_parallel_world_size()
    torch.distributed.all_reduce(reporting_loss, group=mpu.get_data_parallel_group())
    # loss[0] is a view of loss, so it has ._base not None, which triggers assert error
    # in core/pipeline_parallel/schedule.py::deallocate_output_tensor, calling .clone()
    # on loss[0] fixes this
    local_num_tokens = loss[1].clone().detach().to(torch.int)

    return loss, local_num_tokens, reporting_loss