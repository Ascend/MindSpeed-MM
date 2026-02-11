import logging
from datetime import datetime

import torch

from mindspeed.fsdp.utils.log import print_rank

from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.utils.utils import move_to_device, get_time
from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.tools.profiler import Profiler
from mindspeed_mm.fsdp.loss.loss_func import build_loss_func
from mindspeed_mm.fsdp.params.argument import Arguments


logger = logging.getLogger(__name__)


class TrainEngine:
    """Training engine that manages the main training loop and operations."""
    def __init__(self, args: Arguments, train_dataloader, model, optimizer, scheduler, checkpointer, **kwargs):
        self.args = args

        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = scheduler
        self.checkpointer = checkpointer

        # Training state tracking
        self.iteration, self.consumed_train_samples = 0, 0
        
        # Load checkpoint if specified
        if args.training.load:
            self.iteration, self.consumed_train_samples = self.load()

        self.profiler = None
        if args.training.profile.profile_this_rank:
            self.profiler = Profiler(args.training.profile)
            self.profiler.start()

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

    def get_loss_func(self, batch_data):
        args = self.args
        if args.model.loss_cfg.loss_type == "custom":
            return batch_data

        if args.model.loss_cfg.enable_chunk_loss:
            loss_func, loss_mask = build_loss_func(args.model.loss_cfg.loss_type,
                                                 chunk_size=args.model.loss_cfg.chunk_size, **batch_data)
        else:
            loss_func, loss_mask = build_loss_func(args.model.loss_cfg.loss_type, chunk_size=None, **batch_data)
        batch_data.update(
            enable_chunk_loss=args.model.loss_cfg.enable_chunk_loss,
            router_aux_loss_coef=args.model.loss_cfg.router_aux_loss_coef,
            output_router_logits=args.model.loss_cfg.router_aux_loss_coef > 0.0,
            loss_func=loss_func,
        )
        return batch_data

    def train_step(self, train_dataloader_iter):
        """Perform a single training step with gradient accumulation."""
        args = self.args
        total_loss = 0
        # Gradient accumulation
        for _ in range(args.training.gradient_accumulation_steps):
            # Get current batch data
            batch_data = self.get_batch(train_dataloader_iter)

            # Move input to device and cast precision
            batch_data = move_to_device(batch_data, get_dtype(args.parallel.fsdp_plan.param_dtype) if args.parallel.fsdp_plan.param_dtype else None)

            # setup loss ctx
            batch_data = self.get_loss_func(batch_data)

            # forward step
            output = self.model(**batch_data)
            loss = output.loss / args.training.gradient_accumulation_steps

            # Backward
            loss.backward()

            total_loss += loss

        # Average loss across data parallel group
        total_loss = self.average_losses_across_data_parallel_group([total_loss])

        return total_loss

    def train(self):
        """Main training loop."""
        args = self.args

        # Get data iterator
        train_dataloader_iter, _, _ = build_iterations(self.train_dataloader)
        self.model.train()

        # --- Train Loop ---
        curr_step_lr = self.lr_scheduler.get_last_lr()[0]
        while self.iteration < args.training.train_iters:
            # Record memory usage if profiling
            if self.profiler is not None:
                self.profiler.memory_record()
            start_time = get_time(barrier=True)

            loss = self.train_step(train_dataloader_iter)

            # Clip gradients when clip_grad>0 and get total grad_norm
            grad_norm = clip_grad_norm(self.model, max_norm=args.training.clip_grad, foreach=args.training.clip_grad_foreach)

            # Update parameters
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            # Stop profiling if enabled
            if self.profiler is not None:
                self.profiler.step()
                self.profiler.stop()

            # Update training state
            self.consumed_train_samples += args.training.global_batch_size
            self.iteration += 1

            # Calculate iteration time
            elapsed_time_per_iteration = get_time(barrier=True) - start_time

            # Logging
            if self.iteration % args.training.log_interval == 0:
                self.training_log(
                    self.iteration,
                    elapsed_time_per_iteration,
                    curr_step_lr,
                    self.consumed_train_samples,
                    loss,
                    grad_norm
                )

            curr_step_lr = self.lr_scheduler.get_last_lr()[0]

            # Save checkpoint at specified intervals
            if args.training.save and args.training.save_interval > 0 and self.iteration % args.training.save_interval == 0:
                self.save(self.iteration, self.consumed_train_samples)

        # Final save after training completes
        if args.training.save:
            self.save(self.iteration, self.consumed_train_samples)

    def training_log(
        self,
        iteration,
        elapsed_time_per_iteration,
        curr_step_lr,
        consumed_train_samples,
        loss,
        grad_norm
    ):
        args = self.args
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.training.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            consumed_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        log_string += ' learning rate: {:.6E} |'.format(curr_step_lr)
        log_string += ' global batch size: {:5d} |'.format(args.training.global_batch_size)
        log_string += ' loss: {:.6E} |'.format(loss.item())
        if grad_norm is not None:
            log_string += ' grad norm: {:.3f} |'.format(grad_norm)

        print_rank(logger.info, log_string)

    def load(self):
        """Load checkpoint and restore training state."""
        args = self.args
        iteration, consumed_train_samples = 0, 0
        state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}  # cannot be None
        release = self.checkpointer.load(path=args.training.load, state=state)

        if not release:
            iteration = state["extra_state"]["iteration"]
            consumed_train_samples = state["extra_state"]["consumed_train_samples"]

            self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
            self.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        # Synchronize all processes after loading
        torch.distributed.barrier()

        return iteration, consumed_train_samples

    def save(self, iteration, consumed_train_samples):
        """Save checkpoint with model, optimizer, and training state."""
        args = self.args
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "extra_state": {
                "iteration": iteration,
                "consumed_train_samples": consumed_train_samples,
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "train_dataloader": self.train_dataloader.state_dict()
            },
        }
        self.checkpointer.save(args.training.save, state=state, iteration=iteration)

        # Synchronize all processes after saving
        torch.distributed.barrier()