import argparse
import math
import itertools

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.utils.data import DataLoader, DistributedSampler
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ocr_dataset import OCRDataset, DataCollatorForDeepSeekOCR
from transformers import AutoModel


class OCRTrainer:
    def __init__(self, config):
        self.config = config
        self.validate_args()
        self.build_dataloader()
        self.build_model_and_optimizer()

    def validate_args(self):
        dp_size = torch.distributed.get_world_size()
        gbs = self.config.global_batch_size
        mbs = self.config.micro_batch_size
        if gbs % mbs != 0 or gbs % (mbs * dp_size) != 0:
            raise ValueError(f"Gobal batch size {gbs} must be multiple of "
                             f"micro batch size {mbs} times data parallel size {dp_size}")
        gradient_accumulation_steps = gbs // mbs // dp_size
        setattr(self.config, "gradient_accumulation_steps", gradient_accumulation_steps)

    def build_dataloader(self):
        train_dataset = OCRDataset(
            self.config.data_path, self.config.load,
            cutoff_len=self.config.seq_length,
            trust_remote_code=self.config.trust_remote_code
        )
        collate_fn = DataCollatorForDeepSeekOCR()
        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        sampler = DistributedSampler(
            train_dataset, rank=rank, num_replicas=num_replicas,
            shuffle=not self.config.no_shuffle, seed=self.config.seed, drop_last=True
        )
        dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            batch_size=self.config.micro_batch_size,
            num_workers=self.config.num_workers
        )
        self.data_iter = itertools.cycle(dataloader)

    def build_model_and_optimizer(self, attn_implementation="eager"):
        self.model = AutoModel.from_pretrained(
            self.config.load,
            _attn_implementation=attn_implementation,
            trust_remote_code=True,
            use_safetensors=True
        ).to("cuda", dtype=torch.bfloat16)
        # freeze vis part
        self.model.model.vision_model.requires_grad_(False)

        fsdp_kwargs = {}
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32
        )

        fully_shard(self.model.model.embed_tokens)
        for layer in self.model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        for layer in self.model.model.vision_model.transformer.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(self.model.lm_head, **fsdp_kwargs)
        fully_shard(self.model, **fsdp_kwargs)

        if torch.distributed.get_rank == 0:
            print(self.model)

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        num_warmup_steps = int(self.config.warmup_ratio * self.config.train_iters)
        self.scheduler = OCRTrainer.get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.config.train_iters
        )

    def train_step(self):
        for _ in range(self.config.gradient_accumulation_steps):
            total_loss = 0
            self.optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_data = next(self.data_iter)
                inputs = {k: v.to(torch.cuda.current_device()) for k, v in batch_data.items()}
                outputs = self.model(**inputs)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()
                total_loss += loss
        return total_loss.detach().item()

    def train(self):
        self.model.train()
        iteration = 0

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False, data_simplification=True
        )

        prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=1, repeat=1, skip_first=10),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profiling_data")
        )
        if self.config.profiling:
            prof.start()

        while iteration < self.config.train_iters:
            loss = self.train_step()
            if self.config.clip_grad > 0:
                gnorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_grad)
            self.optimizer.step()
            self.scheduler.step()

            iteration += 1
            if self.config.profiling:
                prof.step()
            log_string = "iteration {:8d}/{:8d}".format(iteration, self.config.train_iters)
            log_string += f" | learning rate: {self.scheduler.get_last_lr()[0]:.6E}"
            log_string += f" | global batch size: {self.config.global_batch_size:5d}"
            log_string += f" | loss: {loss:.6E}"
            if self.config.clip_grad > 0:
                log_string += f" | grad norm: {gnorm.item():.6E}"
            if torch.distributed.get_rank() == 0:
                print(log_string)
        if self.config.profiling:
            prof.stop()

    @staticmethod
    def get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps,
            num_training_steps,
            min_lr_ratio=0.0,
            num_cycles=0.5,
            last_epoch=-1,
    ):
        if min_lr_ratio < 0 or min_lr_ratio > 1.0:
            raise ValueError("`min_lr_ratio` must be in [0, 1]")
        coef = (1 - min_lr_ratio) * 0.5
        intercept = (1 + min_lr_ratio) * 0.5

        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            x = math.cos(math.pi * float(num_cycles) * 2.0 * progress)
            return max(0.0, x * coef + intercept)

        return LambdaLR(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def cleanup_distributed():
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @staticmethod
    def setup_distributed():
        if torch.distributed.is_initialized():
            return
        torch.distributed.init_process_group(backend="hccl")
        rank = torch.distributed.get_rank()
        torch.cuda.set_device(rank)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Ming Model Training Configuration',
        allow_abbrev=False
    )

    # 数据加载相关参数
    parser.add_argument(
        '--num-workers',
        type=int,
        default=8,
        help='Dataloader number of workers'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Disable shuffling of training data (use for deterministic results)'
    )

    # 训练过程控制参数
    parser.add_argument(
        '--seed',
        type=int,
        default=1234,
        help='Random seed for reproducibility (affects Python, NumPy, PyTorch and CUDA)'
    )
    parser.add_argument(
        '--micro-batch-size',
        type=int,
        default=None,
        help='Batch size per GPU (before gradient accumulation)'
    )
    parser.add_argument(
        '--global-batch-size',
        type=int,
        default=None,
        help='Training batch size'
    )
    parser.add_argument(
        '--train-iters',
        type=int,
        default=None,
        help='Total number of training iterations (alternative to --train-samples)'
    )

    # 模型结构参数
    parser.add_argument(
        '--seq-length',
        type=int,
        default=None,
        help='Maximum sequence length to process'
    )

    # 优化器相关参数
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Initial learning rate (before warmup and decay)'
    )
    parser.add_argument(
        '--clip-grad',
        type=float,
        default=1.0,
        help='Maximum gradient norm for clipping (set to 0 to disable)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.01,
        help='L2 regularization coefficient'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Proportion of training steps used for linear learning rate warmup'
    )

    # 路径相关参数
    parser.add_argument(
        '--data-path',
        type=str,
        default=None,
        help='Path to training data file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Directory containing auxiliary data files'
    )
    parser.add_argument(
        '--processor-path',
        type=str,
        default=None,
        help='Path to pretrained processor/tokenizer directory'
    )
    parser.add_argument(
        '--load',
        type=str,
        default=None,
        help='Directory containing a model checkpoint.'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Output directory to save checkpoints to.'
    )

    # 安全相关参数
    parser.add_argument(
        '--trust-remote-code',
        action='store_true',
        default=False,
        help='Whether or not to allow for custom models defined on the Hub in their own modeling files.')
    parser.add_argument(
        '--profiling',
        action='store_true',
        default=False,
        help="Whether or not to start profiling"
    )
    return parser


def main():
    args = get_parser().parse_args()

    OCRTrainer.setup_distributed()
    ocr_trainer = OCRTrainer(args)
    ocr_trainer.train()
    OCRTrainer.cleanup_distributed()


if __name__ == "__main__":
    torch.npu.config.allow_internal_format = False
    main()