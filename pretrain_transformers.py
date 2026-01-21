# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain VLM (ViT+MLP+LLM) MODEL."""
from copy import deepcopy
from functools import partial
from typing import Dict, Any
import importlib.util
from dataclasses import dataclass

from datasets import Dataset
import torch

spec = importlib.util.spec_from_file_location("config_loader", "mindspeed_mm/configs/read_yaml_config.py")
spec.loader.exec_module(importlib.util.module_from_spec(spec))
import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training import get_args, print_rank_0
from megatron.training.utils import average_losses_across_data_parallel_group, unwrap_model
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations, cal_gradient_accumulation_size
from mindspeed_mm.data.data_utils.constants import AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM
from mindspeed_mm.data.dataloader.dataloader import PrefetchGradAccDataLoader
from mindspeed_mm.data.dataloader.dynamic_batching_dataloader import DynamicBatchingDataLoader
from mindspeed_mm.training import pretrain
from mindspeed_mm.models.transformers_model import TransformersModel
mindspeed_args = get_mindspeed_args()
if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def model_provider(*args, **kwargs):
    """Builds the model."""
    args = get_args()
    print_rank_0("building VLMModel ...")
    vlm_config = deepcopy(args.mm.model)
    model = TransformersModel(vlm_config)

    return model


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    """Move batch tensors to current device with given float dtype."""
    new_batch = dict()
    for k, v in batch.items():
        if k in [AVG_PER_STEP_TOKEN_NUM, GLOBAL_STEP_TOKEN_NUM]:
            new_batch[k] = v.to(device=torch.cuda.current_device())
        elif isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            new_batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            new_batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]
    return new_batch


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    return batch


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dir = {}

    loss = output_tensor['loss']    
    if output_tensor.get('token_nums', None) is not None:
        total_tokens = output_tensor['token_nums']
    else:
        loss_mask = output_tensor['loss_mask'].view(-1).float()
        total_tokens = loss_mask.sum()
    
    if args.log_tps:
        dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
        tokens_per_sample = torch.tensor(total_tokens / args.micro_batch_size, device=output_tensor['loss'].device) / dp_size
        torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group(with_context_parallel=True))
        loss_dir["tokens per sample"] = tokens_per_sample

    averaged_loss = loss.clone().detach().view(1)
    torch.distributed.all_reduce(
        averaged_loss, 
        group=mpu.get_data_parallel_group(with_context_parallel=True), 
        op=torch.distributed.ReduceOp.AVG
    )
    averaged_loss *= mpu.get_context_parallel_world_size()
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss, loss_dir


def forward_step(data_iterator, model):
    """Forward step."""
    batch_data = get_batch(data_iterator)
    if get_args().use_torch_fsdp2:
        dtype = unwrap_model(model).model._get_fsdp_state()._mp_policy.param_dtype
        dtype = dtype if dtype is not None else torch.bfloat16
        batch_data = move_to_device(batch_data, dtype)
    else:
        batch_data = move_to_device(batch_data, get_args().params_dtype)

    output_tensor = model(**batch_data)
    return output_tensor, loss_func


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    data_config = args.mm.data

    datasets = build_mm_dataset(data_config.dataset_param)
    build_dataloader = partial(
        build_mm_dataloader,
        dataloader_param=data_config.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        dataset_param=data_config.dataset_param,
        consumed_samples=args.consumed_train_samples
    )
    if isinstance(datasets, tuple) and len(datasets) == 2:
        train_dataset, valid_dataset = datasets
        train_dataloader = build_dataloader(train_dataset)
        valid_dataloader = build_dataloader(valid_dataset)
        train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
    else:
        train_dataset = datasets
        val_rate = getattr(data_config.dataset_param.basic_parameters, 'val_rate', 0.0)
        if not (0.0 <= val_rate <= 1.0):
            raise ValueError(f'val_rate must be between 0.0 and 1.0, got {val_rate}')
        if isinstance(train_dataset, Dataset) and val_rate > 0:
            dataset = train_dataset.train_test_split(test_size=val_rate, seed=args.seed)
            train_dataset, valid_dataset = dataset['train'], dataset['test']
            train_dataloader = build_dataloader(train_dataset)
            valid_dataloader = build_dataloader(valid_dataset)
            if args.use_txt_dynamic_batching:
                train_dataloader = DynamicBatchingDataLoader(
                    train_dataloader,
                    max_seq_len=args.max_seq_len,
                    dynamic_batch_buffer_size=args.dynamic_batch_buffer_size,
                    vision_layout=args.mm.model.image_encoder.vision_encoder.attn_layout,
                    consumed_train_samples=args.consumed_train_samples,
                )
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader, valid_dataloader)
        else:
            train_dataloader = build_dataloader(train_dataset)
            if args.use_txt_dynamic_batching:
                train_dataloader = DynamicBatchingDataLoader(
                    train_dataloader,
                    max_seq_len=args.max_seq_len,
                    dynamic_batch_buffer_size=args.dynamic_batch_buffer_size,
                    vision_layout=args.mm.model.image_encoder.vision_encoder.attn_layout,
                    consumed_train_samples=args.consumed_train_samples,
                )
            train_dataloader, valid_dataloader, test_dataloader = build_iterations(train_dataloader)
    
    loss_config = getattr(args.mm.model, "loss_cfg", None)
    use_prefetch_gradacc_dataloader = False
    if loss_config:
        use_prefetch_gradacc_dataloader = (getattr(loss_config, "loss_type", "default") == "per_token_loss")
    if use_prefetch_gradacc_dataloader:
        train_dataloader = PrefetchGradAccDataLoader(train_dataloader, grad_acc_step=cal_gradient_accumulation_size())
    
    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    from mindspeed_mm.patchs import torch_dcp_patch
    train_valid_test_datasets_provider.is_distributed = True
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=mm_extra_args_provider,
        args_defaults={"dataloader_type": "external"},
    )
