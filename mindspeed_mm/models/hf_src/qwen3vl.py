import yaml
import torch
import transformers
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import (
    fully_shard,
    CPUOffloadPolicy,
    OffloadPolicy,
    MixedPrecisionPolicy
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper, CheckpointImpl

from megatron.training import get_args
from mindspeed.utils import _get_dtype


class Qwen3VLFSDP2Mixin:
    """
    Mixin class for FSDP2 of the Qwen3VL-series
    """
    def fully_shard(
        self,
        process_group,
        fsdp2_config_path,
        **kwargs
    ):
        """
        Applies Fully Sharded Data Parallel v2 (FSDP2) wrapping to the model for distributed training.

        Args:
            process_group (torch.distributed.ProcessGroup):
                The process group used for communication within a shard group.

            fsdp2_config_path (str):
                Path to the YAML configuration file for FSDP2 settings. Must include:
                - sharding_size (int or "auto" or null): Number of devices in a shard group.
                  If "auto" or null, defaults to world size.
                - param_dtype (str, optional): Data type for parameters (e.g., "bf16", "fp16", "fp32").
                - reduce_dtype (str, optional): Data type for gradient reduction.
                - cast_forward_inputs (bool, optional): Whether to cast forward inputs to param_dtype.
                - offload_to_cpu (bool, optional): Whether to enable CPU offloading.
                - pin_memory (bool, optional): Whether to use pinned memory for offloaded tensors.

            **kwargs:
                Additional keyword arguments for potential future FSDP options (currently unused).

        Returns:
            bool: True if sharding and setup completed successfully.

        Example:
            >>> mixin = Qwen3VLFSDP2Mixin()
            >>> mixin.fully_shard(
            ...     process_group=pg,
            ...     fsdp2_config_path="configs/fsdp2.yaml"
            ... )
            True

        Note:
            - This method modifies the model in-place by applying checkpoint wrappers and FSDP wrappers.
            - `post_meta_init` is only called if `args.init_model_with_meta_device` is True,
              to handle models initialized on meta device.
        """
        with open(fsdp2_config_path, "r", encoding="utf-8") as fr:
            fsdp2_config = yaml.safe_load(fr)
        
        shard_size = fsdp2_config["sharding_size"] if fsdp2_config["sharding_size"] not in ["auto", None] else torch.distributed.get_world_size()
        device_mesh = init_device_mesh("cuda", (shard_size, ))
        
        fsdp2_kwargs = {
            "mesh": device_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=_get_dtype(fsdp2_config.get("param_dtype")),
                reduce_dtype=_get_dtype(fsdp2_config.get("reduce_dtype")),
                output_dtype=None,
                cast_forward_inputs=fsdp2_config.get("cast_forward_inputs", False)
            ),
            "offload_policy": CPUOffloadPolicy(pin_memory=fsdp2_config.get("pin_memory", True)) if fsdp2_config.get("offload_to_cpu", False) else OffloadPolicy()
        }
        
        # recompute
        for i, block in enumerate(self.model.visual.blocks):
            self.model.visual.blocks[i] = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)
        
        for i, layer in enumerate(self.model.language_model.layers):
            self.model.language_model.layers[i] = checkpoint_wrapper(layer, CheckpointImpl.REENTRANT)        
        
        # fully_shard
        for block in self.model.visual.blocks:
            fully_shard(block, **fsdp2_kwargs)
        fully_shard(self.model.visual.merger, **fsdp2_kwargs)
        for merger in self.model.visual.deepstack_merger_list:
            fully_shard(merger, **fsdp2_kwargs)
        fully_shard(self.model.visual, **fsdp2_kwargs)
        
        fully_shard(self.model.language_model.embed_tokens, **fsdp2_kwargs)
        for layer in self.model.language_model.layers:
            fully_shard(layer, **fsdp2_kwargs)
        fully_shard(self.lm_head, **fsdp2_kwargs)
        fully_shard(self, **fsdp2_kwargs)
        
        # post meta init
        args = get_args()
        if args.init_model_with_meta_device:
            self.post_meta_init()
            
        return True
    
    
    def post_meta_init(self):
        """
        Moves the model from meta device to NPU and reinitializes buffers 
        (e.g., rotary position embeddings) after FSDP sharding.
        """
        self.to_empty(device="cuda")
        # reset buffer
        self.visual.rotary_pos_emb(dim=self.config.vision_config.hidden_size // self.config.vision_config.num_heads // 2)
        self.language_model.rotary_emb(self.config.text_config)
        
        self.visual.rotary_pos_emb.to(device="cuda")
        self.language_model.rotary_emb.to(device="cuda")
        
        
class Qwen3VLForConditionalGeneration(transformers.Qwen3VLForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass


class Qwen3VLMoeForConditionalGeneration(transformers.Qwen3VLMoeForConditionalGeneration, Qwen3VLFSDP2Mixin):
    pass