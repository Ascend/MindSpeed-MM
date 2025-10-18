from typing import Optional, Dict, Union

import re
import yaml
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.modelzoo import ModelZoo


class TransformersModel(MultiModalModule):

    def __init__(self, config) -> None:
        super().__init__(config=config)
        args = get_args()

        hf_path = args.mm.model.init_from_hf_path
        self.config = core_transformer_config_from_args(args)
        self.transformer_config = AutoConfig.from_pretrained(hf_path)

        model_cls = ModelZoo.build(config, self.transformer_config)
        self.model = model_cls.from_pretrained(
            hf_path,
            config=self.transformer_config,
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="cpu"
        )
        print_rank_0("> load model successfully")

        with open(args.fsdp2_config_path, "r", encoding="utf-8") as fr:
            fsdp2_config = yaml.safe_load(fr)
        if fsdp2_config.get("recompute_modules", None):
            self.model.gradient_checkpointing_enable()


    def compute_language_model_loss(self, logits: Tensor, labels: Tensor, ignore_index: int = -100, **kwargs) -> Tensor:
        args = get_args()
        loss = None
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = shift_labels > 1

        # The three loss calculation modes are mutually exclusive:
        # 1. Default behavior (calculate_per_sample_loss=False and calculate_per_token_loss=False):
        #   Calculate the average loss for the micro batch and dividing by micro batch num
        # 2. Token level (calculate_per_token_loss=True):
        #    Keep per-token losses without any aggregation, used for scenarios requiring token-level loss
        # 3. Sample level (calculate_per_sample_loss=True):
        #    Calculate per-sample average loss by first computing the average loss of valid tokens within each sample, then averaging across all samples
        if args.calculate_per_sample_loss:
            logits = logits.permute(0, 2, 1).contiguous()
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)
            batch_mean_loss = loss.sum(dim=1) / (shift_labels > -1).sum(dim=1)
            loss = batch_mean_loss.mean()
        elif args.calculate_per_token_loss:
            shift_labels = shift_labels.view(-1)
            # Flatten the tokens
            logits = logits.view(-1, logits.shape[-1])
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)
        else:
            shift_labels = shift_labels.view(-1)
            # Flatten the tokens
            logits = logits.view(-1, logits.shape[-1])
            loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

        return loss, loss_mask


    def forward(
            self,
            input_ids: torch.Tensor,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            cache_position: Optional[torch.LongTensor] = None,
            *args, **kwargs
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cache_position=cache_position,
            use_cache=False,
            **kwargs
        )
        logits = outputs.logits.contiguous().float()
        loss_dict = {}

        loss, loss_mask = self.compute_language_model_loss(logits, labels, **kwargs)
        loss_dict["loss"] = loss
        loss_dict["loss_mask"] = loss_mask
        return loss_dict
