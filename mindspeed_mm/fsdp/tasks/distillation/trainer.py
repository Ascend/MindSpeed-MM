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

"""H3 DMD2 training task using MindSpeed-MM FSDP2 infrastructure."""

from dataclasses import field
import hashlib
import logging
from pathlib import Path
from typing import Any
import torch
from mindspeed_mm.fsdp.config.config_manager import ConfigManager
from mindspeed_mm.fsdp.params.argument import Arguments
from mindspeed_mm.fsdp.train.trainer import Trainer
from mindspeed_mm.fsdp.models.modelhub import ModelHub
from mindspeed_mm.fsdp.models.minimax_h3.distillation_adapter import H3DistillationRoles
from mindspeed_mm.fsdp.checkpoint.hf_checkpointer import HuggingFaceCheckpointer
from mindspeed_mm.fsdp.checkpoint.dcp_checkpointer import DistributedCheckpointer
from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.optimizer.clip_grad_norm import clip_grad_norm
from mindspeed_mm.fsdp.optimizer.lr_scheduler import build_lr_scheduler
from mindspeed_mm.fsdp.utils.dtype import get_dtype
from mindspeed_mm.fsdp.utils.device import get_device_type, synchronize
from mindspeed_mm.fsdp.utils.random import set_seed
from mindspeed_mm.fsdp.utils.utils import configure_hsdp_gradient_sync, move_to_device, to_empty_if_needed
from mindspeed_mm.fsdp.tools.profiler import profiler
from mindspeed_mm.fsdp.tools.memory_profiler import memory_profiler
from mindspeed_mm.fsdp.log import print_rank
from mindspeed_mm.fsdp.log.metrics import metrics
from .checkpoint import load_checkpoint, save_checkpoint
from .config import DMD2Config
from .lora_import import (
    LoraImportConfig,
    attach_prepared_lora,
    fresh_lora_plan,
    load_prepared_lora_weights,
    prepare_lora,
)
from .train_engine import DistillationEngine

logger = logging.getLogger(__name__)


class DistillationArguments(Arguments):
    distillation: dict[str, Any] = field(default_factory=dict)


class DistillationConfigManager(ConfigManager):
    """Keep the recipe dictionary opaque to dynamic field registration.

    DMD2Config validates recipe fields after normal YAML/CLI merging.
    """

    def _process_yaml_extensions(self, yaml_config):
        super()._process_yaml_extensions({key: value for (key, value) in yaml_config.items() if key != "distillation"})


class DistillationTrainer(Trainer):

    def __init__(self, args, **kwargs):
        self.recipe = DMD2Config(**args.distillation)
        parallel = args.parallel
        if any(
            (
                getattr(parallel, name, 1) != 1
                for name in (
                    "tensor_parallel_size",
                    "ulysses_parallel_size",
                    "ring_attention_size",
                    "expert_parallel_size",
                    "kvallgather_parallel_size",
                )
            )
        ):
            raise ValueError("This DMD2 stage supports FSDP2 data parallelism only; CP/TP/EP need separate validation")
        if args.training.micro_batch_size != 1 or args.training.train_epochs is not None:
            raise ValueError("DMD2 uses micro_batch_size=1 and explicit train_iters counted as role updates")
        if args.training.save_async or args.training.quantization_plan.quant_recipe:
            raise ValueError("Async checkpoint and quantized training are not enabled for the DMD2 baseline")
        if args.training.optimizer != "adamw" or args.training.lr_scaling_plan:
            raise ValueError("This DMD2 baseline uses AdamW without a global layerwise LR override")
        if args.data.dataloader_param.enable_preload:
            raise ValueError("Disable enable_preload for the initial exact-resume DMD2 baseline")
        if args.data.dataloader_param.num_workers != 0:
            raise ValueError("Use num_workers=0 for this initial exact-resume baseline")
        if args.training.lora.pretrained_lora_path:
            raise ValueError("Configure Student/Critic import_lora separately, not the global pretrained_lora_path")
        if args.parallel.fsdp_plan.cpu_offload:
            raise ValueError("CPU parameter offload is not enabled in this DMD2 stage")
        if args.training.val_interval > 0 or args.features.enable_grad_norm_overlap:
            raise ValueError("Generic validation and gradient-norm overlap are not enabled for DMD2")
        if args.training.no_save_optim or args.training.no_save_rng:
            raise ValueError("DMD2 checkpoints require both optimizer and random state")
        if args.training.load:
            raise ValueError("Use distillation.resume_from for full DMD2 resume, not training.load")
        if args.training.save and args.training.save_format != "dcp":
            raise ValueError("DMD2 training-state checkpoints require training.save_format=dcp")
        args.training.lora.enable = True
        args.training.init_model_with_meta_device = True
        self.base_assets = {}
        super().__init__(args, **kwargs)

    def get_model(self, model_provider=None):
        if model_provider is not None:
            raise ValueError("Use the registered H3 adapter for this training task")
        args = self.args
        roles, plans, paths = ({}, {}, {})
        base_dtype = (
            get_dtype(args.parallel.fsdp_plan.param_dtype) if args.parallel.fsdp_plan.param_dtype else torch.float32
        )
        for index, role in enumerate(("student", "teacher", "critic")):
            settings = getattr(self.recipe, role, None)
            path = (
                self.recipe.teacher_path if role == "teacher" else settings.base_model_path
            ) or args.model.model_name_or_path
            if not path:
                raise ValueError(f"A local base-model directory is required for {role}")
            paths[role] = str(Path(path).expanduser().resolve())
            source_config = Path(paths[role]) / "config.json"
            self.base_assets[role] = {
                "path": paths[role],
                "config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
            }
            model_args = args.model.model_copy(deep=True)
            model_args.model_id = "minimax_h3_distillation"
            model_args.model_name_or_path = paths[role]
            model = ModelHub.build(model_args, args.features, args.training, meta_init=True).to(dtype=base_dtype)
            if role == "teacher":
                model.requires_grad_(False)
            else:
                if settings.import_lora:
                    options = dict(settings.import_lora)
                    options.setdefault("head_dim", model.blocks[0].attn.head_dim)
                    options = LoraImportConfig(**options)
                    if options.dropout != 0:
                        raise ValueError("Inference-consistent DMD2 rollout requires adapter dropout=0")
                    plan = prepare_lora(options)
                else:
                    plan = fresh_lora_plan(
                        model,
                        rank=settings.rank,
                        alpha=settings.alpha,
                        targets=settings.targets,
                        seed=args.training.seed + index,
                    )
                attach_prepared_lora(model, plan, allow_meta=True)
                plans[role] = plan
            roles[role] = model
        models = H3DistillationRoles(**roles)
        self.model_features_applier.pre_fully_shard_apply(models)
        models = self.model_parallel_applier(models)
        self.model_features_applier.post_fully_shard_apply(models)
        for role, model in models.get_sub_models().items():
            if any((not callable(getattr(block, "unshard", None)) for block in model.blocks)):
                raise ValueError(f"FSDP plan must wrap individual {role}.blocks; refusing a full-model gather")
            to_empty_if_needed(model, device=get_device_type(), buffer_device=get_device_type())
            HuggingFaceCheckpointer.load(
                paths[role],
                {"model": model},
                load_rank0_and_broadcast=args.training.load_rank0_and_broadcast,
                load_strict=True,
                enable_lora=role != "teacher",
                model_id="minimax_h3_distillation",
            )
            if role in plans:
                load_prepared_lora_weights(model, plans.pop(role))
                settings = getattr(self.recipe, role)
                initialization = "imported_lora" if settings.import_lora else "fresh_lora"
                print_rank(
                    logger.info,
                    f"[DMD2] role={role} init={initialization} lr={settings.lr:g} clip_grad={settings.clip_grad:g}",
                )
        if plans:
            raise RuntimeError("Some role adapters were not loaded")
        return models

    def get_optimizer(self):
        result = {}
        for role in ("student", "critic"):
            model = getattr(self.model, role)
            args = self.args.training
            from mindspeed_mm.fsdp.optimizer.optimizer import build_optimizer

            result[role] = build_optimizer(
                model=model,
                lr=getattr(self.recipe, role).lr,
                betas=(args.adam_beta1, args.adam_beta2),
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
                fused=args.adam_fused,
                optimizer_type=args.optimizer,
            )
        return result

    def get_dataloader(self):
        if self.recipe.teacher_video_cfg != 1 or self.recipe.teacher_audio_cfg != 1:
            basic = self.args.data.dataset_param.basic_parameters
            if not getattr(basic, "load_negative", False):
                raise ValueError("Teacher CFG requires data basic_parameters.load_negative=true")
        return super().get_dataloader()

    def get_scheduler(self):
        counts = self.recipe.role_counts(self.args.training.train_iters)
        return {
            role: build_lr_scheduler(
                optimizer,
                train_steps=max(1, counts[role]),
                lr=getattr(self.recipe, role).lr,
                lr_min=self.args.training.lr_min,
                lr_decay_style=self.args.training.lr_decay_style,
                lr_decay_ratio=self.args.training.lr_decay_ratio,
                lr_warmup_ratio=self.args.training.lr_warmup_ratio,
                lr_start=self.args.training.lr_start,
            )
            for (role, optimizer) in self.optimizer.items()
        }

    def get_checkpointers(self):
        return (HuggingFaceCheckpointer, DistributedCheckpointer)

    def get_train_engine(self):
        args = self.args
        state = get_parallel_state()
        set_seed(args.training.seed + state.get_dp_rank(), set_deterministic=args.training.use_deter_comp)

        def on_update(values):
            self.model_features_applier.on_step_end()
            profiler.step()
            role = values["role"]
            task_label = "DMD2"
            if args.training.log_interval > 0 and values["update"] % args.training.log_interval == 0:
                print_rank(
                    logger.info,
                    f"[{task_label}] update={values['update']}/{args.training.train_iters} role={role} objective={values['objective']} teacher_forwards={values['teacher_forwards']} role_update={values['role_update']} video_loss={values['video_loss']:.6g} audio_loss={values['audio_loss']:.6g} grad={values['grad_norm']:.6g} seconds={values['seconds']:.2f} consumed_samples={values['consumed_samples']}",
                )
                metrics.record(
                    values["update"],
                    {
                        key: values[key]
                        for key in ("video_loss", "audio_loss", "grad_norm", "lr", "seconds", "teacher_forwards")
                    },
                    kind="scalar",
                    prefix=f"{'dmd2'}/{role}",
                )

        engine = DistillationEngine(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.recipe,
            gradient_accumulation=args.training.gradient_accumulation_steps,
            global_batch_size=args.training.global_batch_size,
            data_parallel_rank=state.get_dp_rank(),
            move_batch=lambda batch: move_to_device(batch, get_dtype(args.parallel.fsdp_plan.param_dtype)),
            clip_grad=lambda model, maximum: clip_grad_norm(model, maximum, foreach=args.training.clip_grad_foreach),
            prepare_backward=configure_hsdp_gradient_sync,
            synchronize=synchronize,
            on_update=on_update,
            asset_identity={
                "base_models": self.base_assets,
                "data": args.data.dataset_param.to_dict(),
                "training": {
                    key: getattr(args.training, key)
                    for key in (
                        "train_iters",
                        "lr_decay_style",
                        "lr_decay_ratio",
                        "lr_warmup_ratio",
                        "lr_min",
                        "lr_start",
                    )
                },
            },
        )
        return DistillationRunner(engine, args)


class DistillationRunner:

    def __init__(self, engine, args):
        self.engine, self.args = (engine, args)

    def train(self):
        engine, args = (self.engine, self.args)
        if engine.config.resume_from:
            load_checkpoint(engine, engine.config.resume_from)
        last_saved = engine.update
        profiler.start()
        try:
            while engine.update < args.training.train_iters:
                memory_profiler.step()
                print_rank(
                    logger.info,
                    f"[{'DMD2'}] update={engine.update + 1} role={engine.config.role_at(engine.update)} status=start",
                )
                engine.train_update()
                if (
                    args.training.save
                    and args.training.save_interval > 0
                    and (engine.update % args.training.save_interval == 0)
                ):
                    save_checkpoint(engine, Path(args.training.save) / f"update_{engine.update:08d}")
                    last_saved = engine.update
            if args.training.save and engine.update != last_saved:
                save_checkpoint(engine, Path(args.training.save) / f"update_{engine.update:08d}")
        finally:
            profiler.stop()
            memory_profiler.stop()
            metrics.close()


if __name__ == "__main__":
    arguments = DistillationConfigManager(config_class=DistillationArguments).load_and_parse()
    DistillationTrainer(arguments).train()
