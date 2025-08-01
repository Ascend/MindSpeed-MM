import os
import argparse
import random
import time
from itertools import cycle

import numpy as np
import torch
from torch import nn
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
from diffusers.models.attention import JointTransformerBlock

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from configuration_bailingmm import BailingMMConfig
from modeling_bailing_moe import BailingMoeDecoderLayer
from dataset.t2i_dataset import T2IDataset, collate_fn
from diffusion.sd3_transformer import SD3SingleTransformerBlock


class BailingMMT2IModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path,
        device,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        load_image_gen=True
    ):
        super().__init__()
        self.current_device = device

        config = BailingMMConfig.from_pretrained(pretrained_model_name_or_path)
        config.audio_config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = attn_implementation
        config.llm_config._attn_implementation = attn_implementation
        config.talker_config._attn_implementation = attn_implementation
        BailingMMNativeForConditionalGeneration._supports_flash_attn_2 = False
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            load_image_gen=load_image_gen
        ).to(device=self.current_device, dtype=torch_dtype)

    def forward(self, images, batched_input_ids, batched_attn_mask, args):
        vae = self.model.diffusion_loss.vae
        train_model = self.model.diffusion_loss.train_model
        noise_scheduler = self.model.diffusion_loss.noise_scheduler
        prompt_embeds = []
        bs = batched_input_ids.shape[0]
        for index in range(bs):
            prompt_ids = batched_input_ids[index]
            attn_mask = batched_attn_mask[index]
            prompt_embed = self.model.get_condition_embeds_for_image_gen(
                input_ids=prompt_ids, 
                attention_mask=attn_mask,
                image_embeds=None,
                position_ids=None,
                image_grid_thw=None,
                use_cache=False
            )
            prompt_embeds.append(prompt_embed)
        prompt_embeds = torch.cat(prompt_embeds)

        latents = vae.encode(images).latent_dist.mode()
        model_input = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        loss = self._compute_diffusion_loss(train_model, model_input, prompt_embeds, noise_scheduler, args)
        return loss

    def _compute_diffusion_loss(self, train_model, model_input, prompt_embeds, noise_scheduler, args):
        noise = torch.randn_like(model_input, device=self.current_device)
        bsz = model_input.shape[0]
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=self.current_device)

        # Add noise according to flow matching.
        # zt = (1 - texp) * x + texp * z1
        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = noise_scheduler.sigmas.to(device=self.current_device, dtype=dtype)
            schedule_timesteps = noise_scheduler.timesteps.to(self.current_device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma
        sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # predict output
        model_output = train_model(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss
        target = noise - model_input

        loss = torch.mean(
            (weighting.float() * (model_output.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        loss = loss.mean()
        return loss


class MingT2ITrainer:

    def __init__(self, args, world_size, device, rank) -> None:
        self.args = args
        self.world_size = world_size
        self.device = device
        self.rank = rank
        self.build_dataloader()
        self.build_model_and_optimizer()

    def build_dataloader(self):
        # load dataset
        train_dataset = T2IDataset(self.args)
        sampler = DistributedSampler(train_dataset, rank=self.rank, num_replicas=self.world_size, shuffle=True, seed=self.args.sampler_seed)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            batch_size=self.args.micro_batch_size,
            num_workers=self.args.dataloader_num_workers,
            drop_last=True,
        )
        self.data_iter = cycle(train_dataloader)

    def set_seed(self, seed: int, deterministic: bool = False):
        """
        Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

        Args:
            seed (`int`):
                The seed to set.
            deterministic (`bool`, *optional*, defaults to `False`):
                Whether to use deterministic algorithms where available. Can slow down training.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.npu.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True)

    def build_model_and_optimizer(self):
        # build model and apply fsdp2 strategy
        self.model = BailingMMT2IModel(self.args.pretrained_model_name_or_path, self.device)
        self.model.requires_grad_(False)
        self.train_model = self.model.model.diffusion_loss.train_model
        self.train_model.requires_grad_(True)
        self._apply_fsdp2(self.model)

        # build opt
        params_to_optimize = self.model.parameters()
        params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
        self.optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.learning_rate,
            betas=(self.args.beta1, self.args.beta2),
            weight_decay=self.args.weight_decay,
            eps=self.args.eps,
        )
        if dist.get_rank() <= 0:
            print(f"Optimizer: {self.optimizer}, Trainable params: {sum(p.numel() for p in params_to_optimize)}")

        # build scheduler
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps,
            num_training_steps=self.args.max_train_steps
        )

    def train_step(self):
        for _ in range(self.args.gradient_accumulation_steps):
            total_loss = 0
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_data = next(self.data_iter)
                inputs = {k: v.to(self.device) for k, v in batch_data.items()}
                loss = self.model(**inputs, args=self.args)
                loss = loss / self.args.gradient_accumulation_steps
                loss.backward()
                total_loss += loss
        return total_loss.detach().item()

    def train(self):
        self.train_model.train()
        iteration = 0 
        while iteration < self.args.max_train_steps:
            start_time = time.time()
            if iteration % self.args.checkpointing_steps == 0:
                self.save_checkpoint(self.model, self.rank, self.args.output_dir, iteration)
                dist.barrier()

            loss = self.train_step()
            if self.args.clip_grad > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(self.train_model.parameters(), max_norm=self.args.clip_grad)
            else:
                gnorm = None
            self.optimizer.step()
            self.lr_scheduler.step()

            step_time = time.time() - start_time
            iteration += 1

            log_string = f"iteration {iteration:8d}/{self.args.max_train_steps:8d}"
            log_string += f" | learning rate: {self.lr_scheduler.get_last_lr()[0]:.6E}"
            gbs = self.args.gradient_accumulation_steps * self.args.micro_batch_size
            log_string += f" | global batch size: {gbs:5d}"
            log_string += f" | loss: {loss:.6E}"
            log_string += f" | step time: {step_time: .2E}"
            if gnorm:
                log_string += f" | grad norm: {gnorm.item():.6E}"
            if dist.get_rank() <= 0:
                print(log_string)

    def save_checkpoint(self, model, rank, output_dir, iteration):
        # Not Support Now
        pass

    def _apply_fsdp2(
        self,
        model,
        fsdp2_wrap_modules=(BailingMoeDecoderLayer, JointTransformerBlock, SD3SingleTransformerBlock),
        mesh=None,
        reshard_after_forward=True,
        shard_placement_fn=None,
        mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        offload_policy=OffloadPolicy(),
        ignored_params=None
    ):
        fsdp2_kwargs = {
            "mesh": mesh,
            "reshard_after_forward": reshard_after_forward,
            "shard_placement_fn": shard_placement_fn,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
            "ignored_params": ignored_params
        }
        for module in model.modules():
            if any(
                isinstance(module, fsdp2_wrap_module)
                for fsdp2_wrap_module in fsdp2_wrap_modules
            ):
                fully_shard(module, **fsdp2_kwargs)
        fully_shard(model, **fsdp2_kwargs)

    @staticmethod
    def setup_distributed():
        """init parallel state"""
        dist.init_process_group(backend="hccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        device = torch.device(f"cuda:{rank}")
        return world_size, device, rank

    @staticmethod
    def cleanup_distributed():
        dist.destroy_process_group()


def get_parser():
    parser = argparse.ArgumentParser(description='MingT2I Trainer Arguments', allow_abbrev=False)
    # model
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/data/weights/inclusionAI/Ming-Lite-Omni-1.5/")
    # dataset, dataloader
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        help=(
            "The resolution for input images, all the images in the dataset will be resized to this"
        ),
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="/data/datasets/t2i_dataset/data_new.jsonl",
        help=(
            "json path for dataset"
        ),
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="/data/datasets/t2i_dataset/images/",
        help=(
            "image forder for dataset"
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader.",
    )
    # validation & logs
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=1234,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # optimizer & scheduler & Training
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--eps", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--clip_grad", default=1.0, type=float, help="clip grad."
    )
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    # diffusion setting
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    return parser


def main():
    args = get_parser().parse_args()
    world_size, device, rank = MingT2ITrainer.setup_distributed()
    trainer = MingT2ITrainer(
        args=args, world_size=world_size, device=device, rank=rank
    )
    trainer.train()
    MingT2ITrainer.cleanup_distributed()


if __name__ == "__main__":
    torch.npu.config.allow_internal_format = False
    main()
