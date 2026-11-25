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

"""UDMA fused AllToAllV and grouped-matmul operators for MoE.

The public helpers follow the ``gemm_mc2.py`` interface. In particular,
``send_counts`` and ``recv_counts`` are one-dimensional expert-count tensors.
The UDMA extension calls them ``local_tokens`` and ``global_tokens`` for the
dispatch operator, and consumes them in the opposite order for the combine
operator.

Call ``initialize_udma_group(ep_group)`` during startup, after setting the NPU
device and creating the distributed process groups, before the first forward.
All ranks must participate because initialization contains a global barrier.
Forward and backward operators never initialize the UDMA group implicitly.
"""

import atexit
import importlib
import ipaddress
import os
import threading
from types import ModuleType
from typing import Optional
import socket

import torch
import torch.distributed as dist

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

from torch.library import custom_op


if IS_NPU_AVAILABLE:
    import torch_npu


_UDMA_MODULE_NAME = "ops_transformer_catccos_udma"
_udma_module: Optional[ModuleType] = None
_udma_initialized = False
_udma_lock = threading.Lock()


def _load_udma_module() -> ModuleType:
    global _udma_module
    if _udma_module is None:
        try:
            _udma_module = importlib.import_module(_UDMA_MODULE_NAME)
        except ImportError as exc:
            raise RuntimeError(
                f"{_UDMA_MODULE_NAME} is required when "
                "moe_token_dispatcher_type=alltoall_mc2_udma."
            ) from exc
    return _udma_module


def _get_local_rank() -> int:
    local_rank = os.getenv("LOCAL_RANK")
    if local_rank is not None:
        return int(local_rank)
    if hasattr(torch, "npu"):
        return torch.npu.current_device()
    raise RuntimeError("LOCAL_RANK must be set before initializing the UDMA communication group.")


def initialize_udma_group(group) -> ModuleType:
    """Explicitly initialize UDMA during startup and return its Python module.

    Call on all ranks after device selection and EP process-group creation,
    before any forward/backward execution or graph capture.
    """
    global _udma_initialized
    if _udma_initialized:
        return _load_udma_module()
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before the UDMA communication group.")

    ep_rank = dist.get_rank(group)
    ep_world_size = dist.get_world_size(group)
    local_rank = _get_local_rank()
    global_rank = dist.get_rank()

    print(
        f"[UDMA INIT] "
        f"global_rank={global_rank}, "
        f"ep_rank={ep_rank}, "
        f"ep_world_size={ep_world_size}, "
        f"local_rank={local_rank}",
        flush=True,
    )

    with _udma_lock:
        udma = _load_udma_module()
        root_rank = dist.get_global_rank(group, 0)

        # SHMEM bootstrap requires one reachable numeric IPv4 address shared by all EP ranks.
        bootstrap_host = os.environ.get("UDMA_MASTER_ADDR") or os.environ.get("MASTER_ADDR")
        if not bootstrap_host:
            raise RuntimeError("Set UDMA_MASTER_ADDR or MASTER_ADDR for the UDMA bootstrap.")
        try:
            master_ip = socket.gethostbyname(bootstrap_host)
        except OSError as exc:
            raise RuntimeError(f"Cannot resolve UDMA bootstrap address {bootstrap_host!r}.") from exc
        try:
            ipaddress.IPv4Address(master_ip)
        except ipaddress.AddressValueError as exc:
            raise RuntimeError(f"UDMA bootstrap address must resolve to IPv4, got {master_ip!r}.") from exc

        configured_port = os.environ.get("UDMA_MASTER_PORT")
        if configured_port is None:
            master_port = os.environ.get("MASTER_PORT")
            if master_port is None:
                raise RuntimeError("Set UDMA_MASTER_PORT or MASTER_PORT for the UDMA bootstrap.")
            port = int(master_port) + 1
        else:
            port = int(configured_port)
        if not 1 <= port <= 65535:
            raise ValueError(f"UDMA bootstrap port must be in [1, 65535], got {port}.")

        if not _udma_initialized:
            torch.npu.synchronize()
            torch.distributed.barrier()
            print(
                f"[UDMA BEFORE INIT] ",
                f"host={socket.gethostname()} pid={os.getpid()} ",
                f"global_rank={global_rank} ep_rank={ep_rank} ",
                f"ep_size={ep_world_size} local_rank={local_rank} ",
                f"root_rank={root_rank} ",
                f"members={dist.get_process_group_ranks(group)} ",
                f"init_method=tcp://{master_ip}:{port}",
                flush=True,
            )
            udma.init_udma_group(
                rank=ep_rank,
                world_size=ep_world_size,
                local_rank=local_rank,
                init_method=f"tcp://{master_ip}:{port}",
            )
            torch.npu.synchronize()
            torch.distributed.barrier()
            _udma_initialized = True
            print(f"udma initialized")
            atexit.register(destroy_udma_group)
    return udma


def _get_initialized_udma_module() -> ModuleType:
    """Return the ready module without importing or initializing it."""
    if not _udma_initialized or _udma_module is None:
        raise RuntimeError(
            "UDMA communication group is not initialized. Call "
            "initialize_udma_group(ep_group) on all ranks during startup, "
            "after setting the NPU device and creating the process groups, "
            "before running forward or backward."
        )
    return _udma_module


def destroy_udma_group() -> None:
    """Destroy the process-wide UDMA group if it was initialized."""
    global _udma_initialized
    with _udma_lock:
        if _udma_initialized:
            _load_udma_module().destroy_udma_group()
            _udma_initialized = False


def _validate_group(group: dist.ProcessGroup) -> tuple[int, int]:
    if group is None:
        raise ValueError("UDMA MC2 requires an expert-parallel process group.")

    ep_size = dist.get_world_size(group)
    ep_rank = dist.get_rank(group)
    if hasattr(dist, "get_process_group_ranks"):
        group_ranks = list(dist.get_process_group_ranks(group))
        global_rank = dist.get_rank()
        first_rank = global_rank // ep_size * ep_size
        expected_ranks = list(range(first_rank, first_rank + ep_size))
        if group_ranks != expected_ranks:
            raise ValueError(
                "UDMA MC2 currently requires each expert-parallel group to use "
                "contiguous, EP-size-aligned global ranks."
            )
    return ep_size, ep_rank


def _check_counts(
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    ep_size: int,
) -> int:
    expert_num = send_counts.numel()
    if recv_counts.numel() != expert_num:
        raise ValueError("send_counts and recv_counts must contain the same number of experts.")
    if expert_num % ep_size != 0:
        raise ValueError(
            f"Number of experts ({expert_num}) must be divisible by EP size ({ep_size})."
        )
    return expert_num


def _validate_udma_route_tensors(
    local_tokens: torch.Tensor,
    global_tokens: torch.Tensor,
    expert_num: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the one-dimensional route tensors used by UDMA wrappers."""
    if local_tokens.ndim != 1 or local_tokens.numel() != expert_num:
        raise ValueError(
            f"local_tokens must have shape ({expert_num},), got {tuple(local_tokens.shape)}."
        )
    if global_tokens.ndim != 1 or global_tokens.numel() != expert_num:
        raise ValueError(
            f"global_tokens must have shape ({expert_num},), "
            f"got {tuple(global_tokens.shape)}."
        )

    if local_tokens.dtype != global_tokens.dtype:
        raise ValueError("local_tokens and global_tokens must have the same dtype.")
    if local_tokens.dtype not in (torch.int32, torch.int64):
        raise ValueError("local_tokens and global_tokens must use torch.int32 or torch.int64.")
    if local_tokens.device != global_tokens.device:
        raise ValueError("local_tokens and global_tokens must be on the same device.")
    return local_tokens.contiguous(), global_tokens.contiguous()


def _normalize_permute_output(
    permute_output: torch.Tensor,
    hidden_size: int,
    valid_tokens: int,
) -> torch.Tensor:
    """Validate permute_output's [valid_tokens, hidden_size] layout."""
    if permute_output.ndim != 2:
        raise RuntimeError(
            "UDMA permute_output must be two-dimensional, "
            f"got shape {tuple(permute_output.shape)}."
        )

    # UDMA may return a converted internal storage format, but its logical
    # PyTorch shape is [valid_tokens, hidden_size].
    if permute_output.shape[1] != hidden_size:
        raise RuntimeError(
            "UDMA permute_output must have logical shape "
            f"[{valid_tokens}, {hidden_size}], got {tuple(permute_output.shape)}."
        )
    if permute_output.shape[0] != valid_tokens:
        raise RuntimeError(
            f"UDMA permute_output has {permute_output.shape[0]} rows, "
            f"but routing metadata requires exactly {valid_tokens}."
        )
    return permute_output


@custom_op("mindspeed_mm::udma_all2allv_gmm", mutates_args=())
def udma_all2allv_gmm_op(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        ep_size: int,
        expert_num: int,
        received_rows: int,
        output_all_to_allv: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    udma = _get_initialized_udma_module()
    return udma.all_to_allv_grouped_mat_mul(
        inputs,
        weights,
        local_tokens,
        global_tokens,
        ep_size=ep_size,
        expert_num=expert_num,
        received_rows=received_rows,
        output_all_to_allv=output_all_to_allv,
    )


def _all2all_grouped_matmul(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    local_tokens: torch.Tensor,
    global_tokens: torch.Tensor,
    ep_size: int,
    expert_num: int,
    valid_rows: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    local_tokens, global_tokens = _validate_udma_route_tensors(
        local_tokens,
        global_tokens,
        expert_num,
    )
    # Transposed weights used by the inverse direction in backward are views.
    # The UDMA extension requires a physically contiguous weight tensor.
    weights = weights.contiguous()
    if valid_rows is None:
        valid_rows = int(global_tokens.sum().item())
    output, permute_output = udma_all2allv_gmm_op(
        inputs,
        weights,
        local_tokens,
        global_tokens,
        ep_size=ep_size,
        expert_num=expert_num,
        received_rows=valid_rows,
        output_all_to_allv=True,
    )

    if output.shape[0] != valid_rows:
        raise RuntimeError(
            f"UDMA AllToAllV+GMM returned {output.shape[0]} rows, but routing metadata "
            f"requires exactly {valid_rows}."
        )
    permute_output = _normalize_permute_output(
        permute_output,
        hidden_size=inputs.shape[-1],
        valid_tokens=valid_rows,
    )
    return output, permute_output


@custom_op("mindspeed_mm::udma_gmm_all2allv", mutates_args=())
def udma_gmm_all2allv_op(
        inputs: torch.Tensor,
        weights: torch.Tensor,
        local_tokens: torch.Tensor,
        global_tokens: torch.Tensor,
        ep_size: int,
        expert_num: int,
        output_rows: int,
) -> torch.Tensor:
    udma = _get_initialized_udma_module()
    out = inputs.new_empty((output_rows, weights.shape[-1]))
    return udma.grouped_mat_mul_all_to_allv(
        inputs,
        weights,
        local_tokens,
        global_tokens,
        ep_size=ep_size,
        expert_num=expert_num,
        out=out,
    )


def _grouped_matmul_all2all(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    local_tokens: torch.Tensor,
    global_tokens: torch.Tensor,
    ep_size: int,
    expert_num: int,
    output_rows: Optional[int] = None,
) -> torch.Tensor:
    local_tokens, global_tokens = _validate_udma_route_tensors(
        local_tokens,
        global_tokens,
        expert_num,
    )
    # Enforce the extension contract at the boundary for both public forward
    # calls and inverse-direction calls from autograd backward.
    weights = weights.contiguous()
    if output_rows is None:
        output_rows = int(local_tokens.sum().item())
    output = udma_gmm_all2allv_op(
        inputs,
        weights,
        local_tokens,
        global_tokens,
        ep_size=ep_size,
        expert_num=expert_num,
        output_rows=output_rows,
    )
    return output


def _grouped_weight_grad(
    inputs: torch.Tensor,
    output_grad: torch.Tensor,
    group_list: torch.Tensor,
) -> torch.Tensor:
    return torch_npu.npu_grouped_matmul(
        [inputs.T],
        [output_grad],
        bias=None,
        group_list=group_list,
        split_item=3,
        group_type=2,
        group_list_type=1,
    )[0]


class AllToAllGroupedMatmul(torch.autograd.Function):
    """Autograd wrapper for UDMA AllToAllV + grouped matmul."""

    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight, received_rows=None):
        if shared_inputs is not None or shared_weight is not None:
            raise NotImplementedError("UDMA MC2 does not support fused shared experts.")

        ep_size, _ = _validate_group(group)
        expert_num = _check_counts(send_counts, recv_counts, ep_size)
        output, permute_output = _all2all_grouped_matmul(
            inputs,
            weights,
            send_counts,
            recv_counts,
            ep_size,
            expert_num,
            valid_rows=received_rows,
        )

        ctx.save_for_backward(weights, send_counts, recv_counts, permute_output)
        ctx.ep_size = ep_size
        ctx.expert_num = expert_num
        # The inverse communication returns one gradient row per input row.
        ctx.input_rows = inputs.shape[0]
        return output, None

    @staticmethod
    def backward(ctx, output_grad, shared_output_grad):
        weights, send_counts, recv_counts, permute_output = ctx.saved_tensors
        inputs_grad = _grouped_matmul_all2all(
            output_grad,
            weights.transpose(-1, -2),
            send_counts,
            recv_counts,
            ctx.ep_size,
            ctx.expert_num,
            output_rows=ctx.input_rows,
        )

        group_list = recv_counts.reshape(ctx.ep_size, -1).sum(dim=0)
        weights_grad = _grouped_weight_grad(permute_output, output_grad, group_list)
        grads = (inputs_grad, weights_grad, None, None, None, None, None)
        return grads + (None,) if len(ctx.needs_input_grad) == 8 else grads


class GroupedMatmulAllToAll(torch.autograd.Function):
    """Autograd wrapper for UDMA grouped matmul + AllToAllV."""

    @staticmethod
    def forward(ctx, inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight, output_rows=None):
        if shared_inputs is not None or shared_weight is not None:
            raise NotImplementedError("UDMA MC2 does not support fused shared experts.")

        ep_size, _ = _validate_group(group)
        expert_num = _check_counts(send_counts, recv_counts, ep_size)
        output = _grouped_matmul_all2all(
            inputs,
            weights,
            recv_counts,
            send_counts,
            ep_size,
            expert_num,
            output_rows=output_rows,
        )

        ctx.save_for_backward(inputs, weights, send_counts, recv_counts)
        ctx.ep_size = ep_size
        ctx.expert_num = expert_num
        # The inverse communication returns one gradient row per input row.
        ctx.input_rows = inputs.shape[0]
        return output, None

    @staticmethod
    def backward(ctx, output_grad, shared_output_grad):
        inputs, weights, send_counts, recv_counts = ctx.saved_tensors
        inputs_grad, permute_grad = _all2all_grouped_matmul(
            output_grad,
            weights.transpose(-1, -2),
            recv_counts,
            send_counts,
            ctx.ep_size,
            ctx.expert_num,
            valid_rows=ctx.input_rows,
        )

        group_list = send_counts.reshape(ctx.ep_size, -1).sum(dim=0)
        weights_grad = _grouped_weight_grad(inputs, permute_grad, group_list)
        grads = (inputs_grad, weights_grad, None, None, None, None, None)
        return grads + (None,) if len(ctx.needs_input_grad) == 8 else grads


def all2all_grouped_matmul(
    inputs,
    weights,
    group,
    send_counts,
    recv_counts,
    shared_inputs=None,
    shared_weight=None,
    received_rows=None,
):
    output = AllToAllGroupedMatmul.apply(
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        received_rows,
    )
    if shared_inputs is not None:
        return output[0], output[1]
    return output[0]


def grouped_matmul_all2all(
    inputs,
    weights,
    group,
    send_counts,
    recv_counts,
    shared_inputs=None,
    shared_weight=None,
    output_rows=None,
):
    output = GroupedMatmulAllToAll.apply(
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        output_rows,
    )
    if shared_inputs is not None:
        return output[0], output[1]
    return output[0]
