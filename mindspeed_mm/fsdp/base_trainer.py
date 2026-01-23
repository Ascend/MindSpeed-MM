from dataclasses import MISSING, asdict, dataclass, field, fields
from typing import Any, Callable, Dict, List, Literal, Optional, TypeVar, Union, get_type_hints
import logging
import os
from functools import partial
from datetime import datetime
import sys

import torch
import yaml

from mindspeed.fsdp.utils.log import print_rank, set_log_level
from mindspeed.fsdp.utils.device import set_accelerator_compatible
from mindspeed.fsdp.utils.random import set_seed

from mindspeed_mm.fsdp.params.data_args import DataArguments
from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments
from mindspeed_mm.fsdp.params.parallel_args import ParallelArguments
from mindspeed_mm.fsdp.utils.device import (
    get_dist_comm_backend,
    get_torch_device,
    get_device_type
)
from mindspeed_mm.fsdp.distributed.parallel_state import init_parallel_state, get_parallel_state
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.distributed.torch_parallelize import ParallelApplier
from mindspeed_mm.fsdp.utils.utils import to_empty_if_needed, move_to_device, get_time
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.optimizer.optimizer import build_optimizer
from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler
from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import DistributedCheckpointer
from mindspeed_mm.fsdp.tools.profiler import Profiler
from mindspeed_mm.fsdp.params.utils import allow_extra_fields, instantiate_dataclass


logger = logging.getLogger(__name__)


@allow_extra_fields
@dataclass
class Arguments:
    """Root argument class: model/data/parallel/training four types of parameters"""
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)


class BaseTrainer:
    def __init__(self):
        # 1. Parse arguments
        self.parallel_args, self.model_args, self.data_args, self.training_args = self.parse_args()
        
        # 2. Initialize
        self.initialize()

        # 3. Build each module
        self.checkpointer = self.get_checkpointer()
        self.model_parallel_applier = ParallelApplier(self.parallel_args)
        self.model = self.get_model()
        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_scheduler()
        # Load checkpoint if specified
        if self.training_args.load:
            self.iteration, self.consumed_train_samples = self.load()

        self.train_dataloder = self.build_dataloader()

        self.profiler = None
        if self.training_args.profile.profile_this_rank:
            self.profiler = Profiler(self.training_args.profile)
            self.profiler.start()

    def validate_args(self, args):
        """
        Validate arguments and compute dependent parameters across configuration sections.
        """
        # Compute distributed training parameters based on parallel configuration
        args.training.compute_distributed_training(args.parallel)

    def parse_args(self):
        """Parse YAML arguments into structured dataclasses."""

        # Parse command line arguments
        cmd_args = sys.argv[1:]

        # Validate that a configuration file was provided
        if not cmd_args:
            raise ValueError(
                "❌ No configuration file provided.\n"
            )

        # Handle config file input
        input_data = {}
        # Validate file extension to ensure it's a YAML configuration file
        if not (cmd_args[0].endswith(".yaml") or cmd_args[0].endswith(".yml")):
            raise ValueError(
                f"❌ Invalid configuration file: '{cmd_args[0]}'\n"
                f"Expected a YAML file with extension .yaml or .yml\n"
            )
        with open(os.path.abspath(cmd_args[0]), encoding="utf-8") as f:
            input_data: Dict[str, Dict[str, Any]] = yaml.safe_load(f)
        
        # Instantiate the Arguments dataclass from YAML data
        args = instantiate_dataclass(Arguments, input_data)
        
        # Critical: Resolve dependencies between different configuration sections
        # and validate parameter consistency across the entire configuration
        self.validate_args(args)

        return (
            args.parallel,
            args.model,
            args.data,
            args.training
        )
    
    def initialize(self):
        """Initialize training environment: logging, random seeds, distributed groups."""
        print_rank(logger.info, f"Start initializing training environment!!!")

        # Set accelerator compatibility and logging level
        set_accelerator_compatible(get_torch_device())
        set_log_level()
        # Set device index for current process
        torch.accelerator.set_device_index(int(os.environ['LOCAL_RANK']))
        # Set random seeds for reproducibility
        set_seed(self.training_args.seed, set_deterministic=self.training_args.use_deter_comp)

        # Initialize process group for distributed training
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=get_dist_comm_backend())

        # Initialize parallel communication groups and mesh
        init_parallel_state(**asdict(self.parallel_args))

        # Initialize training state variables
        self.iteration = 0
        self.consumed_train_samples = 0

    def get_model(self):
        """Build and initialize the model with parallel strategies applied."""
        model = ModelHub.build(self.model_args, self.training_args)

        # Apply parallel strategies (FSDP2, tensor parallelism, etc.)
        self.model_parallel_applier(model)
        print_rank(logger.info, f"model: \n{model}")

        # Initialize weights on meta device if specified (for memory efficiency)
        if self.training_args.init_model_with_meta_device:
            to_empty_if_needed(model, device=get_device_type())
            model.init_weights()
        
        return model

    def build_dataloader(self):
        """Build training dataloader with proper parallel partitioning."""
        print_rank(logger.info, "Prepare data")
        data_config = self.data_args
        ps = get_parallel_state()

        datasets = build_mm_dataset(data_config.dataset_param)
        dataloader_param = data_config.dataloader_param.to_dict()
        dataloader_param.update(
            {
                "batch_size": self.training_args.micro_batch_size,
                "seed": self.training_args.seed,
            }
        )
        build_dataloader = partial(
            build_mm_dataloader,
            dataloader_param=dataloader_param,
            process_group=ps.get_dp_group(),
            dataset_param=data_config.dataset_param,
            consumed_samples=self.consumed_train_samples
        )
        dataloader = build_dataloader(datasets)
        train_dataloader, _, _ = build_iterations(dataloader)

        return train_dataloader

    def get_optimizer(self):
        """Build optimizer for the model."""
        optimizer = build_optimizer(
            model=self.model,
            lr=self.training_args.lr,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            eps=self.training_args.adam_eps,
            weight_decay=self.training_args.weight_decay,
            fused=self.training_args.adam_fused,
            optimizer_type=self.training_args.optimizer,
        )
        return optimizer
    
    def get_scheduler(self):
        """Build learning rate scheduler."""
        lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=self.training_args.train_iters,
            lr=self.training_args.lr,
            lr_min=self.training_args.lr_min,
            lr_decay_style=self.training_args.lr_decay_style,
            lr_decay_ratio=self.training_args.lr_decay_ratio,
            lr_warmup_ratio=self.training_args.lr_warmup_ratio,
            lr_start=self.training_args.lr_start,
        )
        return lr_scheduler
    
    def get_checkpointer(self):
        """Return checkpointing class (can be overridden for different checkpoint formats)."""
        return DistributedCheckpointer

    def average_losses_across_data_parallel_group(self, losses):
        """Reduce a tensor of losses across all GPUs."""
        ps = get_parallel_state()
        averaged_losses = torch.cat(
            [loss.clone().detach().view(1) for loss in losses])
        torch.distributed.all_reduce(averaged_losses,
                                    group=ps.get_dp_group())
        averaged_losses = averaged_losses / \
            torch.distributed.get_world_size(group=ps.get_dp_group())

        return averaged_losses

    def get_batch(self, data_iterator):
        """Generate a batch."""
        if data_iterator is not None:
            batch = next(data_iterator)
        else:
            raise ValueError("Data iterator is None. Unable to retrieve batch.")
        return batch

    def train_step(self):
        """Perform a single training step with gradient accumulation."""
        total_loss = 0
        # Gradient accumulation
        for _ in range(self.training_args.gradient_accumulation_steps):
            # Get current batch data
            batch_data = self.get_batch(self.train_dataloder)

            # Move input to device and cast precision
            batch_data = move_to_device(batch_data, torch.bfloat16 if self.parallel_args.fsdp_plan.enable_mixed_precision else None)

            # forward step
            output = self.model(**batch_data)
            loss = output.loss

            # Backward
            loss.backward()

            total_loss += loss
        
        # Average loss across data parallel group
        total_loss = self.average_losses_across_data_parallel_group([total_loss])

        return total_loss

    def train(self):
        """Main training loop."""
        self.model.train()

        # --- Train Loop ---
        while self.iteration < self.training_args.train_iters:
            # Record memory usage if profiling
            if self.profiler is not None:
                self.profiler.memory_record()
            start_time = get_time(barrier=True)

            loss = self.train_step()

            # Clip gradients when clip_grad>0 and get total grad_norm
            grad_norm = clip_grad_norm(self.model, max_norm=self.training_args.clip_grad, foreach=self.training_args.clip_grad_foreach)

            # Update parameters
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Stop profiling if enabled
            if self.profiler is not None:
                self.profiler.step()
                self.profiler.stop()

            # Update training state
            self.consumed_train_samples += self.training_args.global_batch_size
            self.iteration += 1
            
            # Calculate iteration time
            elapsed_time_per_iteration = get_time(barrier=True) - start_time

            # Logging
            if self.iteration % self.training_args.log_interval == 0:
                self.training_log(
                    self.iteration,
                    elapsed_time_per_iteration,
                    self.consumed_train_samples,
                    loss,
                    grad_norm
                )
            
            # Save checkpoint at specified intervals
            if self.training_args.save and self.training_args.save_interval > 0 and self.iteration % self.training_args.save_interval == 0:
                self.save(self.iteration, self.consumed_train_samples)
        
        # Final save after training completes
        if self.training_args.save:
            self.save(self.iteration, self.consumed_train_samples)

    def training_log(
        self,
        iteration,
        elapsed_time_per_iteration,
        consumed_train_samples,
        loss,
        grad_norm
    ):
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, self.training_args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.6E} |'.format(self.lr_scheduler.get_last_lr()[0])
        log_string += ' global batch size: {:5d} |'.format(self.training_args.global_batch_size)
        log_string += ' loss: {:.6E} |'.format(loss.item())
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)

        print_rank(logger.info, log_string)

    def load(self):
        """Load checkpoint and restore training state."""
        iteration, consumed_train_samples = 0, 0
        state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}  # cannot be None
        release = self.checkpointer.load(path=self.training_args.load, state=state)

        if not release:
            iteration = state["extra_state"]["iteration"]
            consumed_train_samples = state["extra_state"]["consumed_train_samples"]

            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        # Synchronize all processes after loading
        torch.distributed.barrier()

        return iteration, consumed_train_samples

    def save(self, iteration, consumed_train_samples):
        """Save checkpoint with model, optimizer, and training state."""

        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {
                "iteration": iteration,
                "consumed_train_samples": consumed_train_samples,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        self.checkpointer.save(self.training_args.save, state=state, iteration=iteration)

        # Synchronize all processes after saving
        torch.distributed.barrier()

if __name__ == "__main__":
    trainer = BaseTrainer()
    trainer.train()
