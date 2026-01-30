# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import torch
from megatron.training.global_vars import get_args


def load_checkpoint_bridge(ddp_model, optimizer, opt_param_scheduler, load_arg='load', strict=True,
                           checkpointing_context=None, skip_load_to_model_and_opt=False):
    """
    Load a model checkpoint and return the iteration.
    "return 0, 0" means the iteration returned is 0.
    """
    args = get_args()
    load_dir = getattr(args, load_arg)

    from bridge.models.conversion.auto_bridge import AutoBridge
    bridge = AutoBridge.from_hf_pretrained(load_dir)
    bridge.load_hf_weights(ddp_model)

    if (args.fp16 or args.bf16) and optimizer is not None:
        optimizer.reload_model_params()
    # Some utilities want to load a checkpoint without distributed being initialized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    torch.cuda.empty_cache()
    return 0, 0