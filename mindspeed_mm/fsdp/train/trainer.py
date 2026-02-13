from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints
import logging
import os
from functools import partial

import torch

from mindspeed.fsdp.utils.log import print_rank, set_log_level
from mindspeed.fsdp.utils.random import set_seed

from mindspeed_mm.fsdp.utils.device import (
    get_dist_comm_backend,
    get_torch_device,
    get_device_type,
    set_accelerator_compatible,
    set_allow_hf32
)
from mindspeed_mm.fsdp.distributed.parallel_state import init_parallel_state, get_parallel_state
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.features.apply_features import FeaturesApplier
from mindspeed_mm.fsdp.utils.utils import to_empty_if_needed, init_model_weights
from mindspeed_mm.fsdp.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.fsdp.data.dataloader.dataloader import PrefetchGradAccDataLoader
from mindspeed_mm.fsdp.optimizer.optimizer import build_optimizer
from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler
from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import DistributedCheckpointer
from mindspeed_mm.fsdp.utils.register import import_plugin
from mindspeed_mm.fsdp.params.argument import Arguments, parse_args
from mindspeed_mm.fsdp.train.train_engine import TrainEngine


logger = logging.getLogger(__name__)



class Trainer():
    def __init__(self, args: Arguments, model_provider: Optional[Callable] = None, dataloader_provider: Optional[Callable] = None):
        """
        Initialize the trainer with configuration and optional custom providers.
        
        Args:
            args: Training configuration arguments
            model_provider: Optional custom function to provide the model
            dataloader_provider: Optional custom function to provide the dataloader
        """
        self.args = args

        self.initialize()

        # Initialize model parallelization and feature application
        self.model_parallel_applier = ParallelApplier(args.parallel)
        self.model_features_applier = FeaturesApplier(args.model)

        # Build core training components
        self.model = self.get_model(model_provider)
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()
        self.train_dataloader = self.get_dataloader() if dataloader_provider is None else dataloader_provider()
        self.checkpointer = self.get_checkpointer()

        # Create the training engine
        self.trainer = TrainEngine(
            args, self.train_dataloader, self.model, self.optimizer, self.lr_scheduler, self.checkpointer
        )


    def initialize(self):
        """Initialize training environment: logging, random seeds, distributed groups."""
        args: Arguments = self.args
        print_rank(logger.info, f"Start initializing training environment!!!")

        # Set allow_hf32
        set_allow_hf32(args.training.allow_hf32)

        # Set accelerator compatibility and logging level
        set_accelerator_compatible(get_torch_device())
        set_log_level()
        # Set device index for current process
        torch.accelerator.set_device_index(int(os.environ['LOCAL_RANK']))
        # Set random seeds for reproducibility
        set_seed(args.training.seed, set_deterministic=args.training.use_deter_comp)

        # import plugin and trigger register
        import_plugin(getattr(args.training, "plugin", []))

        # Initialize process group for distributed training
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_dist_comm_backend())

        # Initialize parallel communication groups and mesh
        init_parallel_state(**asdict(args.parallel))
    
    def get_foundation_model(self):
        """Load the foundation model from the model hub."""
        args: Arguments = self.args
        model = ModelHub.build(args.model, args.training)
        return model

    def get_model(self, model_provider: Optional[Callable] = None):
        """
        Build and prepare the model for training.
        
        Args:
            model_provider: Optional custom function to provide the model
            
        Returns:
            Prepared model with parallelization and features applied
        """
        args = self.args
        model = self.get_foundation_model() if model_provider is None else model_provider()

        # Apply parallelization strategy and model features
        self.model_parallel_applier(model)
        self.model_features_applier(model)

        # Initialize weights on meta device if specified (for memory efficiency)
        if args.training.init_model_with_meta_device:
            to_empty_if_needed(model, device=get_device_type())
            init_model_weights(model)
        
        return model

    def get_optimizer(self):
        args = self.args
        """Build optimizer for the model."""
        optimizer = build_optimizer(
            model=self.model,
            lr=args.training.lr,
            betas=(args.training.adam_beta1, args.training.adam_beta2),
            eps=args.training.adam_eps,
            weight_decay=args.training.weight_decay,
            fused=args.training.adam_fused,
            optimizer_type=args.training.optimizer,
        )
        return optimizer

    def get_scheduler(self):
        """Build learning rate scheduler."""
        args = self.args
        lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=args.training.train_iters,
            lr=args.training.lr,
            lr_min=args.training.lr_min,
            lr_decay_style=args.training.lr_decay_style,
            lr_decay_ratio=args.training.lr_decay_ratio,
            lr_warmup_ratio=args.training.lr_warmup_ratio,
            lr_start=args.training.lr_start,
        )
        return lr_scheduler

    def get_dataloader(self):
        """Build training dataloader with proper parallel partitioning."""
        args = self.args
        print_rank(logger.info, "Prepare data")
        data_config = args.data
        ps = get_parallel_state()

        datasets = build_mm_dataset(data_config.dataset_param)
        dataloader_param = data_config.dataloader_param.to_dict()
        dataloader_param.update(
            {
                "batch_size": args.training.micro_batch_size,
                "seed": args.training.seed,
            }
        )
        build_dataloader = partial(
            build_mm_dataloader,
            dataloader_param=dataloader_param,
            process_group=ps.get_dp_group(),
            dataset_param=data_config.dataset_param
        )
        train_dataloader = build_dataloader(datasets)

        if args.model.loss_cfg.loss_type == "per_token_loss":
            train_dataloader = PrefetchGradAccDataLoader(train_dataloader,
                                                         grad_acc_step=args.training.gradient_accumulation_steps)

        return train_dataloader

    def get_checkpointer(self):
        """Return checkpointing class (can be overridden for different checkpoint formats)."""
        return DistributedCheckpointer
    
    def train(self):
        """Start the training process."""
        self.trainer.train()


if __name__ == "__main__":
    # Entry point for training script
    args = parse_args(Arguments)
    trainer = Trainer(args=args)
    trainer.train()