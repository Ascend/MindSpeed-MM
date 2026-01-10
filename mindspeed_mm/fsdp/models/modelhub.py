import os
import importlib
import logging

import torch
import torch.distributed as dist
from transformers import AutoConfig
from accelerate import init_empty_weights

from mindspeed.lite.utils.str_match import module_name_match
from mindspeed.lite.utils.log import print_rank

from mindspeed_mm.fsdp.params.model_args import ModelArguments
from mindspeed_mm.fsdp.params.training_args import TrainingArguments

logger = logging.getLogger(__name__)


class ModelHub:
    """
    Responsible for building HuggingFace native models.
    """
    MODEL_MAPPINGS = {}# Mapping from model IDs to custom model classes

    @staticmethod
    def build(model_args: ModelArguments, training_args: TrainingArguments):
        """
        Build a model instance from HuggingFace based on model arguments and training configuration.
        
        Args:
            model_args: Contains model_name_or_path, trust_remote_code, etc.
            training_args: Contains training configuration like init_model_with_meta_device, etc.
        
        Returns:
            Configured model instance ready for training.
        """

        # Load HuggingFace Config
        print_rank(logger.info, f"> Loading AutoConfig from {model_args.model_name_or_path}...")
        transformer_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code
        )

        # Get model architecture from config
        architectures = getattr(transformer_config, "architectures", [])
        model_cls = None

        # First try to get model class from custom MODEL_MAPPINGS using model_id
        model_id = getattr(model_args, "model_id", None)
        if model_id:
            model_cls = ModelHub.MODEL_MAPPINGS.get(model_id, None)

        # If not found in mappings, try to get from transformers module using architecture name
        if architectures:
            transformers_module = importlib.import_module("transformers")
            model_cls = getattr(transformers_module, architectures[0], None)

        if model_cls is None:
            raise ValueError("load model from config failed")

        # Initialize model with meta device for memory efficiency if specified
        if training_args.init_model_with_meta_device:
            with init_empty_weights():
                model = model_cls._from_config(transformer_config).float()
            for m in model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            # Load model from pretrained weights
            model = model_cls.from_pretrained(
                model_args.model_name_or_path,
                config=transformer_config,
                dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                trust_remote_code=model_args.trust_remote_code
            )

        # Apply parameter freezing if specified
        freezed_named_modules = []
        if len(model_args.freeze) > 0:
            for name, module in model.named_modules():
                for pattern in model_args.freeze:
                    if module_name_match(pattern, name):
                        freezed_named_modules.append((name, module))
            for name, module in freezed_named_modules:
                print_rank(logger.info, f"freezing module {name}...")
                for param in module.parameters():
                    param.requires_grad_(False)

        return model