import torch

from mindspeed.fsdp.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from ..features.memory.recompute import recompute_modules

from .expert_parallel.expert_parallel import expert_parallelize_modules
from .expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from .fully_shard_parallel import fully_shard_parallel_modules
from .parallel_state import get_parallel_state
from ..params.parallel_args import ParallelArguments


class ParallelApplier:
    def __init__(self, parallel_config: ParallelArguments):
        self.config = parallel_config
        self.parallel_state = get_parallel_state()

    def apply_fsdp_modules(self, model):
        model = fully_shard_parallel_modules(model, self.parallel_state.get_fsdp_device_mesh(), self.config.fsdp_plan)

    def apply_tp_modules(self, model):
        if self.config.tensor_parallel_size == 1:
            return
        model = tensor_parallel_modules(model, self.parallel_state.get_tp_device_mesh(), self.config.tp_plan)

    def apply_ep_modules(self, model):
        if not self.config.ep_plan.apply_efsdp_modules:
            self.config.ep_plan.apply_efsdp_modules = self.config.ep_plan.apply_modules
        if self.config.ep_plan._gradient_divide_factor is None:
            self.config.ep_plan._gradient_divide_factor = torch.distributed.get_world_size()
        if self.config.expert_parallel_size > 1 and self.config.ep_plan.apply_modules:
            model = expert_parallelize_modules(model, self.parallel_state.get_ep_device_mesh(), self.config.ep_plan)
            model = expert_fully_shard_modules(model, self.parallel_state.get_efsdp_device_mesh(), self.config.ep_plan, self.config.fsdp_plan)

    def apply_recompute_modules(self, model):
        if not self.config.recompute:
            return
        model = recompute_modules(model, self.config.recompute_plan)

    def __call__(self, model):
        # Apply configuration-based parallel strategies
        # Order matters: TP -> EP -> Recompute -> FSDP
        self.apply_tp_modules(model=model)
        self.apply_ep_modules(model=model)
        self.apply_recompute_modules(model=model)
        self.apply_fsdp_modules(model=model)

