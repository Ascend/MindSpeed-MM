import mindspeed.megatron_adaptor
import torch
from megatron.training import get_args
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron

from mindspeed_mm.patchs.patch_manager import PatchesManager
from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tasks.evaluation.eval_datasets import eval_dataset_dict
from mindspeed_mm.tasks.evaluation.gen_impl import eval_impl_dict, eval_pipeline_dict
from mindspeed_mm.utils.utils import get_device, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def init_eval_pipline(args):
    ori_args = get_args()
    args.pipeline_config.seed = ori_args.seed
    device = get_device(args.device)
    vae = AEModel(args.ae).get_model().to(device, args.ae.dtype).eval()
    text_encoder = TextEncoder(args.text_encoder).get_model().to(device).eval()
    predict_model = PredictModel(args.predictor).get_model()
    if ori_args.load is not None:
        load_checkpoint([predict_model], None, None, strict=False)
    predict_model = predict_model.to(device, args.predictor.dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    if not hasattr(vae, 'dtype'):
        vae.dtype = args.ae.dtype
    inference_pipeline_class = eval_pipeline_dict[args.eval_config.evaluation_model]
    eval_pipeline = inference_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args.pipeline_config)
    return eval_pipeline


def main():
    initialize_megatron(extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider), args_defaults={})
    PatchesManager.apply_patches_from_config()
    args = get_args()
    merge_mm_args(args)
    args.mm.model.micro_batch_size = args.micro_batch_size
    args = args.mm.model

    # prepare arguments
    torch.set_grad_enabled(False)

    eval_dataset_class = eval_dataset_dict[args.eval_config.dataset.type]
    if args.eval_config.evaluation_impl in eval_impl_dict:
        eval_impl_class = eval_impl_dict[args.eval_config.evaluation_impl]
    else:
        raise NotImplementedError(f"eval impl {args.eval_config.evaluation_impl} not found")

    inference_pipeline = init_eval_pipline(args)

    eval_dataset = eval_dataset_class(args.eval_config.dataset.basic_param.to_dict(),
                                      args.eval_config.dataset.extra_param.to_dict(),
                                      args.eval_config.dimensions)
    eval_impl = eval_impl_class(dataset=eval_dataset, inference_pipeline=inference_pipeline, args=args)

    eval_impl()

if __name__ == '__main__':
    main()