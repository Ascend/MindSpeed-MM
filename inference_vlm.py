import torch

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.training import get_args
from mindspeed_mm.tasks.inference.pipeline import vlm_pipeline_dict
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.patchs.patch_manager import PatchesManager
mindspeed_args = get_mindspeed_args()

if hasattr(mindspeed_args, "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor 


def main():
    from megatron.training.initialize import initialize_megatron
    from mindspeed_mm.configs.config import merge_mm_args

    # just inference
    torch.set_grad_enabled(False)

    initialize_megatron(
        extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider), args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    args = get_args()
    merge_mm_args(args)
    # apply patches
    PatchesManager.apply_patches_from_config()
    if not hasattr(args, "dist_train"):
        args.dist_train = False
    inference_config = args.mm.model
    vlm_pipeline_dict[inference_config.pipeline_class](inference_config)()


if __name__ == '__main__':
    main()