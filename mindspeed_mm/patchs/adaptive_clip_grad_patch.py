# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import List, Optional, Union
import math
from dataclasses import dataclass
from functools import wraps
import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.module import param_is_not_shared
from megatron.training import get_args


@dataclass
class AdaptiveGradClipInfo:
    weight_norm = 0.0
    moving_avg_max_grad_norm = -1e6
    moving_avg_max_grad_norm_var = 0.0
    max_grad_norm = 0.0
    max_grad_norm_after_clip = 0.0
    max_norm = 0.0
    max_grad_norm_var = 0.0
    num_zero_grad = 0.0
    clip_coef = 1.0
    zero_grad_flag = 0
    zero_grad_flag_list = None
    nan_norm_flag = 0
    extreme_error_flag = 0


def _import_multi_tensor_applier():
    try:
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        from megatron.core.utils import local_multi_tensor_applier
        multi_tensor_applier = local_multi_tensor_applier
    try:
        import amp_C
        l2_norm_impl = amp_C.multi_tensor_l2norm
        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        from megatron.core.utils import local_multi_tensor_l2_norm, local_multi_tensor_scale
        l2_norm_impl = local_multi_tensor_l2_norm
        multi_tensor_scale_impl = local_multi_tensor_scale
    return multi_tensor_applier, l2_norm_impl, multi_tensor_scale_impl


def get_unlocked_params_weight_norm_fp32(params_for_norm, norm_type=2.0, model_parallel_group=None):
    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(p.data.abs().max() for p in params_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if math.isclose(norm_type, 2.0):
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if params_for_norm:
                multi_tensor_applier, l2_norm_impl, _ = _import_multi_tensor_applier()
                weight_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    [params_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                weight_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = weight_norm ** norm_type

        else:
            total_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            for p in params_for_norm:
                weight_norm = torch.norm(p.data, norm_type)
                total_norm += weight_norm ** norm_type

        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm ** (1.0 / norm_type)

        total_norm = total_norm.item()

    return total_norm


def zero_and_clip_grad_(grads, clip_coef=1.0, zero_grad_flag=True):
    multi_tensor_applier, _, multi_tensor_scale_impl = _import_multi_tensor_applier()
    if zero_grad_flag:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], 0
        )
    elif math.isclose(clip_coef, 1.0):
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        multi_tensor_applier(
            multi_tensor_scale_impl, dummy_overflow_buf, [grads, grads], 1 / (clip_coef + 1.0e-6)
        )

    
def get_grad_norm(grads_for_norm, norm_type=2.0, model_parallel_group=None):
    # Calculate norm.
    if norm_type == torch.inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        # Take max across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=model_parallel_group
        )
        total_norm = total_norm_cuda[0].item()

    else:
        if math.isclose(norm_type, 2.0):
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            # Use apex's multi-tensor applier for efficiency reasons.
            # Multi-tensor applier takes a function and a list of list
            # and performs the operation on that list all in one kernel.
            if grads_for_norm:
                multi_tensor_applier, l2_norm_impl, _ = _import_multi_tensor_applier()
                grad_norm, _ = multi_tensor_applier(
                    l2_norm_impl,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,  # no per-parameter norm
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            # Since we will be summing across data parallel groups,
            # we need the pow(norm-type).
            total_norm = grad_norm ** norm_type

        else:
            total_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type
        # Sum across all model-parallel GPUs.
        torch.distributed.all_reduce(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=model_parallel_group
        )
        total_norm = total_norm ** (1.0 / norm_type)

    return total_norm


def adaptive_clip_grad_norm_fp32_with_distributed_optimizer(
    parameters: Union[List[torch.Tensor], torch.Tensor],
    grads_for_norm: Union[List[torch.Tensor], torch.Tensor],
    params_for_norm: Union[List[torch.Tensor], torch.Tensor] = None,
    norm_type: Union[int, float] = 2,
    clip_grad_ema_decay: float = 0.99,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> float:
    """Clips gradient norm of an iterable of parameters whose gradients
       are in fp32.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized.
        grads_for_norm (Iterable[Tensor]): an iterable of Tensors or a single
            Tensor that will be used for calculating the grad norm.
        max_norm (float or int): max norm of the gradients.
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        model_parallel_group (torch.distributed.ProcessGroup, optional): model-parallel
            group over which grad norm needs to be aggregated.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    # Grads.
    grads = []
    for param in parameters:
        if param.grad is not None:
            if param.grad.type() != 'torch.cuda.FloatTensor':
                raise ValueError(f"param.grad.type() must be torch.cuda.FloatTensor")
            grads.append(param.grad.detach())

    if model_parallel_group is not None:
        raise ValueError("When using distributed optimizer, model_parallel_group should not be None (all ranks).")

    # Norm parameters.
    norm_type = float(norm_type)
    AdaptiveGradClipInfo.weight_norm = weight_norm = get_unlocked_params_weight_norm_fp32(params_for_norm, norm_type, model_parallel_group=None)

    grad_norm_before_clip = get_grad_norm(grads_for_norm, norm_type, model_parallel_group=None)

    nan_norm_flag = 0
    if torch.isnan(grad_norm_before_clip) or torch.isinf(grad_norm_before_clip):
        nan_norm_flag = 1
        
    moving_avg_max_grad_norm = AdaptiveGradClipInfo.moving_avg_max_grad_norm
    moving_avg_max_grad_norm_var = AdaptiveGradClipInfo.moving_avg_max_grad_norm_var
    ema_decay = clip_grad_ema_decay
    is_first_step = True if moving_avg_max_grad_norm < 0.0 else False # the value of init is -1e6, before first step

    # initailize
    grad_norm_after_clip = grad_norm_before_clip

    if is_first_step:  
        moving_avg_max_grad_norm = min(3 * grad_norm_before_clip, 1.0)
        moving_avg_max_grad_norm_var = 0.0
        max_grad_norm_var = moving_avg_max_grad_norm_var
        max_norm = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
        clip_coef = 1.0
        max_grad_norm_after_clip = grad_norm_after_clip = grad_norm_before_clip

        AdaptiveGradClipInfo.moving_avg_max_grad_norm = moving_avg_max_grad_norm
        AdaptiveGradClipInfo.moving_avg_max_grad_norm_var = moving_avg_max_grad_norm_var
        AdaptiveGradClipInfo.max_grad_norm_var = max_grad_norm_var
        AdaptiveGradClipInfo.max_norm = max_norm
        AdaptiveGradClipInfo.clip_coef = clip_coef
        AdaptiveGradClipInfo.max_grad_norm = grad_norm_before_clip
        AdaptiveGradClipInfo.max_grad_norm_after_clip = max_grad_norm_after_clip
        
    else:
        clip_threshold = moving_avg_max_grad_norm + 3.0 * (moving_avg_max_grad_norm_var ** 0.5)
        # For grads that are too large, we believe that the data at this point is extremely abnormal and not suitable for further training, so it is forced to terminate
        extreme_error_threshold = max(moving_avg_max_grad_norm + 5.0 * (moving_avg_max_grad_norm_var ** 0.5), 5.0)

        AdaptiveGradClipInfo.max_norm = clip_threshold
        AdaptiveGradClipInfo.max_grad_norm = grad_norm_before_clip

        if grad_norm_before_clip <= clip_threshold:
            moving_avg_max_grad_norm = ema_decay * moving_avg_max_grad_norm + (1 - ema_decay) * grad_norm_before_clip
            max_grad_norm_var = (moving_avg_max_grad_norm - grad_norm_before_clip) ** 2
            moving_avg_max_grad_norm_var = ema_decay * moving_avg_max_grad_norm_var + (1 - ema_decay) * max_grad_norm_var
            max_grad_norm_after_clip = grad_norm_after_clip = grad_norm_before_clip
            AdaptiveGradClipInfo.moving_avg_max_grad_norm = moving_avg_max_grad_norm
            AdaptiveGradClipInfo.max_grad_norm_var = max_grad_norm_var
            AdaptiveGradClipInfo.moving_avg_max_grad_norm_var = moving_avg_max_grad_norm_var
            AdaptiveGradClipInfo.max_grad_norm_after_clip = max_grad_norm_after_clip

            AdaptiveGradClipInfo.clip_coef = 1.0 # clip_coef = 1.0 means no clipping
        # out of 3 sigma mean abnormal step.
        elif grad_norm_before_clip <= extreme_error_threshold:
            clip_coef = grad_norm_before_clip / clip_threshold
            zero_and_clip_grad_(grads, clip_coef, zero_grad_flag=False)
            grad_norm_after_clip = get_grad_norm(grads_for_norm, norm_type, model_parallel_group=None)
            max_grad_norm_after_clip = grad_norm_after_clip
            # only communication bug can cause this situation
            if torch.isnan(grad_norm_after_clip) or torch.isinf(grad_norm_after_clip):
                nan_norm_flag = 1

            AdaptiveGradClipInfo.max_grad_norm_after_clip = max_grad_norm_after_clip
            AdaptiveGradClipInfo.clip_coef = clip_coef

        if nan_norm_flag or grad_norm_before_clip > extreme_error_threshold:
            print('Extreme error, the training process will be interrupted!')
            AdaptiveGradClipInfo.extreme_error_flag = 1

    AdaptiveGradClipInfo.nan_norm_flag = nan_norm_flag

    if isinstance(grad_norm_after_clip, torch.Tensor):
        grad_norm_after_clip = grad_norm_after_clip.item()

    return grad_norm_after_clip


def get_unlocked_main_params_for_norm(params):
    """
    Get main parameters that should be taken into account to compute the norm.
    Filter parameters based on:
        - parameter should not be shared
        - should not be a replica due to tensor model parallelism.
    """
    params_for_norm = []
    for param in params:
        grad = param.grad
        grad_not_none = grad is not None
        is_not_shared = param_is_not_shared(param)
        is_not_tp_duplicate = tensor_parallel.param_is_not_tensor_parallel_duplicate(param)
        if grad_not_none and is_not_shared and is_not_tp_duplicate:
            params_for_norm.append(param)

    return params_for_norm


# replace megatron DistribtedOptimizer.__init__
# Notice: mindspeed wrapped this function at
# MindSpeed/mindspeed/optimizer.distrib_optimizer.reuse_fp32_param_distrib_optimizer_init_wrapper
def adaptive_clip_grad_norm_optimizer_init_wrapper(init_func):
    @wraps(init_func)
    def adaptive_clip_grad_norm_optimizer_init(*args, **kwargs):
        init_func(*args, **kwargs)
        self = args[0]
        adaptive_clip_grad_norm_args = get_args().mm.model.patch.adaptive_clip_grad_norm
        clip_grad_ema_decay = getattr(adaptive_clip_grad_norm_args, "clip_grad_ema_decay", 0.99)
        setattr(self.config, "clip_grad_ema_decay", clip_grad_ema_decay)
    return adaptive_clip_grad_norm_optimizer_init


# replace megatron DistribtedOptimizer.clip_grad_norm 
def adaptive_clip_grad_norm_wrapper(fn):
    @wraps(fn)
    def adaptive_clip_grad_norm(*args, **kwargs):
        # """Compute grad norm."""
        self = args[0]
        params = self.get_parameters()
        grads_for_norm = self.get_main_grads_for_grad_norm()
        if self.config.clip_grad_ema_decay > 0.0:
            params_for_norm = get_unlocked_main_params_for_norm(self.get_parameters())
            return adaptive_clip_grad_norm_fp32_with_distributed_optimizer(
                params, grads_for_norm, params_for_norm, model_parallel_group=self.get_grad_stats_parallel_group(),
                clip_grad_ema_decay=self.config.clip_grad_ema_decay
            )
        return fn(*args, **kwargs)
    return adaptive_clip_grad_norm