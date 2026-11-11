import logging

import torch

from .expert_parallel.expert_parallel import expert_parallelize_modules
from .expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from .vocab_parallel.vocab_parallel import get_vp_modules, set_host_resident, vocab_parallelize_modules
from .fully_shard_parallel import fully_shard_parallel_modules, set_modules_to_prefetch
from .parallel_state import get_parallel_state
from ..log import print_rank
from ..params.parallel_args import ParallelArguments
from ..params.training_args import TrainingArguments
from ..params.feature_args import FeatureArguments

logger = logging.getLogger(__name__)


class ParallelApplier:
    def __init__(self, parallel_config: ParallelArguments, training_config: TrainingArguments, feature_config=None):
        self.config = parallel_config
        self.training_config = training_config
        self.feature_config = feature_config
        self.parallel_state = get_parallel_state()

    def apply_fsdp_modules(self, model, training_config):
        model = fully_shard_parallel_modules(model, self.parallel_state.get_hsdp_device_mesh(), self.config.fsdp_plan,
                                             training_config)
        return model

    def apply_tp_modules(self, model):
        if self.config.tensor_parallel_size == 1:
            return model
        raise NotImplementedError("Tensor parallelization is not implemented yet.")

    def apply_ep_modules(self, model):
        if not self.config.ep_plan.apply_efsdp_modules:
            self.config.ep_plan.apply_efsdp_modules = self.config.ep_plan.apply_modules
        if self.config.ep_plan._gradient_divide_factor is None:
            # The HSDP all-reduce over the edp replicate dim averages (divides by
            # edp) on top of the reduce-scatter's 1/factor, so the effective
            # divisor is factor * edp. For the expert gradient to land on the mean
            # over the data-parallel group -- the scale the non-EP parameters get
            # -- the factor must therefore be world_size / edp. With world_size it
            # is silently 2x too small for every edp > 1, halving the experts'
            # effective learning rate.
            self.config.ep_plan._gradient_divide_factor = (
                torch.distributed.get_world_size() / self.config.expert_data_parallel_size
            )
        if self.config.expert_parallel_size > 1 and self.config.ep_plan.apply_modules:
            model = expert_parallelize_modules(model, self.parallel_state.get_ep_device_mesh(), self.config.ep_plan)
            model = expert_fully_shard_modules(model, self.parallel_state.get_ehsdp_device_mesh(), self.config.ep_plan, self.config.fsdp_plan)
            # Remove modules already handled by efsdp from the standard FSDP plan to prevent duplicate fully_shard errors
            self.config.fsdp_plan.apply_modules = [x for x in self.config.fsdp_plan.apply_modules if x not in self.config.ep_plan.apply_efsdp_modules]
        return model

    def apply_vocab_modules(self, model):
        if not (self.config.vocab_parallel_size > 1 and self.config.vocab_plan.apply_modules):
            # Vocabulary parallelism offloads the n-gram table to CPU memory. Drop that declaration when
            # it is disabled, so the table is an ordinary accelerator parameter again.
            set_host_resident(model, host_resident=False)
            return model
        if self.config.fully_shard_parallel_size == 1 and not self.training_config.init_model_with_meta_device:
            # DDP replicates the module and all-reduces every parameter on one device, which cannot
            # coexist with a table deliberately kept in host memory: reject it where the table is created
            # instead of after the model has been moved to the accelerator.
            raise NotImplementedError(
                "a host-resident n-gram table (vocab_plan.host_by_cpu=true) needs FSDP2: DDP "
                "(fully_shard_parallel_size=1 without init_model_with_meta_device) has no way to keep one "
                "parameter off the accelerator and out of the gradient all-reduce. Set "
                "fully_shard_parallel_size > 1, enable init_model_with_meta_device, or set "
                "vocab_plan.host_by_cpu=false."
            )
        return vocab_parallelize_modules(
            model,
            self.parallel_state.get_vp_device_mesh(),
            self.config.vocab_plan,
            fsdp_plan=self.config.fsdp_plan,
        )

    def set_modules_to_prefetch(self, model):
        if self.config.fsdp_plan.num_to_forward_prefetch > 0 or self.config.fsdp_plan.num_to_backward_prefetch > 0:
            ep_plan = self.config.ep_plan if self.config.expert_parallel_size > 1 and self.config.ep_plan.apply_modules else None
            set_modules_to_prefetch(model, fsdp_plan=self.config.fsdp_plan, ep_plan=ep_plan)
        return model

    def apply_quantization_modules(self, model):
        """Apply quantization based on quantization_format + quantization_recipe"""
        if not self.training_config.quantization_plan.quant_recipe:
            return
        try:
            quantization_plan = self.training_config.quantization_plan
            quantization_plan.fsdp_world_size = self.config.fully_shard_parallel_size
            quantization_plan._recompute_aware = (
                bool(self.feature_config)
                and self.feature_config.recompute
                and quantization_plan.enable_fsdp_low_precision_all_gather
                and quantization_plan.fsdp_low_precision_all_gather_mode == "on-demand"
            )

            from fsdp_turbo.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.training_config.quantization_plan)
            model_converters.convert(model)
        except Exception as e:
            raise RuntimeError("Failed to convert quantization plan") from e

    def __call__(self, model):
        # Apply configuration-based parallel strategies
        # Order matters: TP -> EP -> VP -> Recompute -> FSDP

        # must set before fully_shard, inplace operation
        self.apply_quantization_modules(model=model)

        model = self.apply_tp_modules(model=model)
        model = self.apply_ep_modules(model=model)
        model = self.apply_vocab_modules(model=model)
        model = self.apply_fsdp_modules(model=model, training_config=self.training_config)
        model = self.set_modules_to_prefetch(model=model)
        return model
