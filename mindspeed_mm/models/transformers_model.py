from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoConfig

from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.modelzoo import ModelZoo
from mindspeed_mm.models.common.chunkloss import chunk_loss, calculate_lm_loss


class TransformersModel(MultiModalModule):

    def __init__(self, config) -> None:
        super().__init__(config=config)
        args = get_args()

        hf_path = args.mm.model.init_from_hf_path
        trust_remote_code = args.trust_remote_code
        self.config = core_transformer_config_from_args(args)
        self.transformer_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=trust_remote_code)

        model_cls = ModelZoo.build(config, self.transformer_config)

        if args.init_model_with_meta_device:
            self.model = model_cls._from_config(self.transformer_config).float()
            for m in self.model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            self.model = model_cls.from_pretrained(
                hf_path,
                config=self.transformer_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=trust_remote_code
            )
        print_rank_0("> load model successfully")

        self.model.train()
        if hasattr(self.model, 'freeze') and callable(getattr(self.model, 'freeze')):
            self.model.freeze(config)
        
        # Retrieve loss configuration from args if available; otherwise, set to None.
        loss_cfg = getattr(args.mm.model, "loss_cfg", None)

        # Default loss mode is "default". It can be overridden by loss_cfg.mode if provided.
        self.loss_compute_mode = "default"
        self.loss_chunk_size = None  # Initialize chunk size as None; only used in "chunk" mode.

        if loss_cfg is not None:
            # Override loss mode based on configuration (supports "default" or "chunk").
            self.loss_compute_mode = getattr(loss_cfg, "compute_mode", "default")
            
            # If using "chunk" mode, retrieve the chunk size (default: 1024).
            if self.loss_compute_mode == "chunk":
                self.loss_chunk_size = getattr(loss_cfg, "chunk_size", 1024)
            
        self.model.use_cache = False

    def compute_language_model_loss(self, logits: Tensor, labels: Tensor, ignore_index: int = -100, **kwargs) -> Tensor:
        args = get_args()
        loss = None
        labels = F.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = shift_labels > -1

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
            loss = torch.sum(loss.view(-1) * loss_mask)
        elif args.calculate_token_loss:
            shift_labels = shift_labels.view(-1)
            # Flatten the tokens
            logits = logits.view(-1, logits.shape[-1])
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)

            loss_weight = (labels != -100).float()

            shift_weights = loss_weight[..., 1:].contiguous()
            shift_weights = shift_weights.view(-1)
            shift_weights = shift_weights.to(logits.device)
            shift_weights_sum = shift_weights.sum()

            torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum

        elif args.calculate_square_loss:
            shift_labels = shift_labels.view(-1)
            # Flatten the tokens
            logits = logits.view(-1, logits.shape[-1])
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)

            loss_weight = (labels != -100).sum(dim=-1).float()
            loss_weight = 1 / loss_weight.sqrt()
            loss_weight = torch.where(labels != -100, loss_weight.unsqueeze(1), 0.0)

            shift_weights = loss_weight[..., 1:].contiguous()
            shift_weights = shift_weights.view(-1)
            shift_weights = shift_weights.to(logits.device)
            shift_weights_sum = shift_weights.sum()

            torch.distributed.all_reduce(shift_weights_sum, op=torch.distributed.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
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
        loss_dict = {}

        if self.loss_compute_mode == "chunk":
            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=self.loss_chunk_size)
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                loss_ctx=loss_ctx,
                **kwargs
            )
            loss_dict["loss"] = outputs.loss
            loss_dict["loss_mask"] = loss_mask
        else:
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

            loss, loss_mask = self.compute_language_model_loss(logits, labels, **kwargs)
            loss_dict["loss"] = loss
            loss_dict["loss_mask"] = loss_mask
        return loss_dict

    def fully_shard(
        self,
        process_group,
        fsdp2_config_path,
        **kwargs
    ):
        # If the model has its own 'fully_shard' method, use it directly
        if hasattr(self.model, 'fully_shard') and callable(getattr(self.model, 'fully_shard')):
            return self.model.fully_shard(
                process_group=process_group,
                fsdp2_config_path=fsdp2_config_path,
                **kwargs
            )
        return False
    
    def build_loss_ctx(
        self,
        labels,
        ignore_index=-100,
        chunk_size=1024,
    ):
        labels = F.pad(labels, (0, 1), value=ignore_index)
        # Shift labels to match the input sequence for next-token prediction.
        shift_labels = labels[..., 1:].contiguous()
        
        # Create a mask to identify valid tokens (typically > 1 means non-special tokens)
        loss_mask = shift_labels > 1
        
        # Retrieve global arguments to determine loss reduction behavior.
        args = get_args()
        if args.calculate_per_sample_loss:
            # Compute per-sample loss: alpha scales each sample by total valid tokens in the batch.
            alpha = loss_mask.sum(1) * loss_mask.shape[0]  # shape: [batch_size]
            reduction = "none"  # Keep per-token losses for sample-wise aggregation.
        elif args.calculate_per_token_loss:
            # Use raw sum loss without normalization here; 
            # token-level loss equivalence will be achieved later by scaling the gradient norm.
            alpha = torch.tensor(1)
            reduction = "sum"
        elif args.calculate_token_loss:
            raise NotImplementedError(f"Chunk loss not support token_loss now")
        elif args.calculate_square_loss:
            raise NotImplementedError(f"Chunk loss not support square_loss now")
        else:
            # Default: normalize loss by total number of valid tokens in the batch.
            alpha = loss_mask.sum()  # scalar
            reduction = "sum"

        # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
        chunk_labels = torch.split(shift_labels, chunk_size, dim=1)
        
        # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
        loss_ctx_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": ignore_index,
                "reduction": reduction,
                "alpha": alpha,
            }
            for i in range(len(chunk_labels))
        ]
        
        # Return a closure that computes the chunked language modeling loss using the prepared config.
        def loss_ctx(hidden_states, head_weight, head_bias):
            return chunk_loss(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward=calculate_lm_loss,
                loss_kwargs_chunks=loss_ctx_kwargs,
                chunk_size=chunk_size
            )
        
        return loss_ctx, loss_mask