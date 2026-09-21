import os
os.environ["USE_TF"] = "FALSE"
import mindspeed.megatron_adaptor
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.patchs.patch_manager import PatchesManager
from mindspeed_mm.tasks.evaluation.eval_datasets import eval_dataset_dict
from mindspeed_mm.tasks.evaluation.eval_impl import eval_impl_dict, eval_pipeline_dict
from mindspeed_mm.tasks.evaluation.eval_prompt import eval_model_prompt_dict


def main():
    initialize_megatron(extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider))
    args = get_args()
    merge_mm_args(args)
    args = args.mm.model
    # apply patches
    PatchesManager.apply_patches_from_config()

    inference_pipeline_class = eval_pipeline_dict[args.evaluation_model]
    eval_dataset_class = eval_dataset_dict[args.evaluation_dataset]
    eval_impl_class = eval_impl_dict[args.evaluation_dataset]
    if args.evaluation_model in eval_model_prompt_dict:
        model_prompt_build = eval_model_prompt_dict[args.evaluation_model](getattr(args, 'use_custom_prompt', True))
    else:
        model_prompt_build = None
    inference_pipeline = inference_pipeline_class(args)

    eval_dataset = eval_dataset_class(args.dataset_path, args.evaluation_dataset)
    eval_impl = eval_impl_class(dataset=eval_dataset, inference_pipeline=inference_pipeline,
                                model_prompt_template=model_prompt_build, args=args)

    eval_impl()


if __name__ == "__main__":
    main()
