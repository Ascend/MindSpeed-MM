import functools
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)
from megatron.training import get_args
from mindspeed.core.distributed.torch_fully_sharded_data_parallel.torch_fully_sharded_data_parallel_adaptor import (
    Fsdp2Config,
    _create_device_mesh,
    _get_device_from_mesh
)


class FSDP2Mixin:
    """
    Mixin class for FSDP2
    """

    def post_meta_init(self):
        pass

    def _pre_fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        fsdp2_config = Fsdp2Config.load_from_yaml(fsdp2_config_path)
        fsdp2_kwargs = fsdp2_config.to_dict()
        device_mesh = _create_device_mesh(fsdp2_config.sharding_size, process_group)
        fsdp2_kwargs["mesh"] = device_mesh
        return fsdp2_kwargs, fsdp2_config

    def _post_fully_shard(self):
        args = get_args()
        if args.init_model_with_meta_device:
            self.post_meta_init()

    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        sub_modules_to_wrap = sub_modules_to_wrap if fsdp2_config.sub_modules_to_wrap is None else fsdp2_config.sub_modules_to_wrap
        if fsdp2_config.sub_modules_to_wrap is None:
            sub_modules_to_wrap = set(sub_modules_to_wrap)
            for sub_module in self.modules():
                fsdp_modules = getattr(sub_module, "_fsdp_modules", [])
                for f in fsdp_modules:
                    sub_modules_to_wrap.add(f)

        device_mesh = fsdp2_kwargs.get("mesh")
        # collect ignored params
        args = get_args()
        ignored_params = set()
        if fsdp2_config.ignored_modules:
            for sub_module in self.modules():
                if isinstance(sub_module, tuple(fsdp2_config.ignored_modules)):
                    if args.use_cpu_initialization:
                        sub_module.to(_get_device_from_mesh(device_mesh))

                    ignored_params.update(sub_module.parameters())
        if ignored_params:
            fsdp2_kwargs["ignored_params"] = ignored_params

        prev_module = None
        wrapped_modules_in_order: list[torch.nn.Module] = []
        for sub_module in self.modules():
            # Wrap individual submodules to fetch parameters just-in-time rather than
            # conservatively fetching all parameters at the start of each iteration.
            if any(
                    isinstance(sub_module, sub_module_to_wrap)
                    for sub_module_to_wrap in sub_modules_to_wrap
            ):
                fully_shard(sub_module, **fsdp2_kwargs)

                # Explicitly set the FSDP backward prefetch schedule to prevent activation
                # recomputation from disrupting the automatically generated default schedule.
                if fsdp2_config.recompute_modules is not None:
                    sub_module.set_modules_to_backward_prefetch(
                        [prev_module] if prev_module else []
                    )
                prev_module = sub_module
                wrapped_modules_in_order.append(sub_module)

        num_to_forward_prefetch = getattr(fsdp2_config, "num_to_forward_prefetch", 0)
        if num_to_forward_prefetch > 0:
            raise NotImplementedError("Do not support prefetch function in default fully shard method, please rewrite `_fully_shard` method")

        # Wrap the root module as required by the FSDP API.
        fully_shard(self, **fsdp2_kwargs)

        # recompute modules to wrap
        if fsdp2_config.recompute_modules:
            apply_activation_checkpointing(
                self,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper, checkpoint_impl=CheckpointImpl.REENTRANT
                ),
                check_fn=lambda module: isinstance(module, tuple(fsdp2_config.recompute_modules))
            )

    def fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        """
        Applies Fully Sharded Data Parallel v2 (FSDP2) wrapping to the model for distributed training.

        Args:
            process_group (torch.distributed.ProcessGroup):
                The process group used for communication within a shard group.

            fsdp2_config_path (str):
                Path to the YAML configuration file for FSDP2 settings. Must include:
                - sharding_size (int or null): Number of devices in a shard group.
                  If null, defaults to world size.
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
        fsdp2_kwargs, fsdp2_config = self._pre_fully_shard(process_group, fsdp2_config_path, **kwargs)
        self._fully_shard(fsdp2_kwargs, fsdp2_config)
        self._post_fully_shard()
        return True