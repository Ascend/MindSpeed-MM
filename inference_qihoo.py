import os
import torch

import mindspeed.megatron_adaptor
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import save_videos, load_prompts, save_image_or_videos
from mindspeed_mm.models.predictor import PredictModel
from mindspeed_mm.models.diffusion import DiffusionModel
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder, Tokenizer
from mindspeed_mm.utils.utils import get_dtype, get_device, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_pipeline(args, device):
    vae = AEModel(args.ae).get_model().to(device, args.ae.dtype).eval()
    text_encoder = TextEncoder(args.text_encoder).get_model().to(device).eval()
    predict_model = PredictModel(args.predictor).get_model().to(device, args.predictor.dtype).eval()
    scheduler = DiffusionModel(args.diffusion).get_model()
    tokenizer = Tokenizer(args.tokenizer).get_tokenizer()
    if not hasattr(vae, 'dtype'):
        vae.dtype = args.ae.dtype
    tokenizer.model_max_length = args.model_max_length
    sora_pipeline_class = sora_pipeline_dict[args.pipeline_class]
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args)
    return sora_pipeline


def main():
    initialize_megatron(extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider), args_defaults={})
    args = get_args()
    merge_mm_args(args)
    args = args.mm.model
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.dtype)
    device = get_device(args.device)

    prompts = load_prompts(args.prompt)
    negative_prompt = ""
    start_idx = 0
    save_fps = args.fps // args.frame_interval
    os.makedirs(args.save_path, exist_ok=True)

    # prepare pipeline
    sora_pipeline = prepare_pipeline(args, device)

    # Iter over all samples
    video_grids = []
    save_iter = 1
    for i in range(0, len(prompts), args.micro_batch_size):
        # prepare batch prompts
        batch_prompts = prompts[i: i + args.micro_batch_size]
        batch_negative_prompts = [negative_prompt] * args.micro_batch_size
        videos = sora_pipeline(prompt=batch_prompts, negative_prompt=batch_negative_prompts, fps=save_fps, device=device, dtype=dtype)
        video_grids.append(videos)
        start_idx += len(batch_prompts)
        if start_idx % (save_iter * args.micro_batch_size) == 0:
            video_grids = torch.cat(video_grids, dim=0)
            save_image_or_videos(video_grids, args.save_path, start_idx - (save_iter * args.micro_batch_size), save_fps, value_range=(-1, 1), normalize=True)
            video_grids = []
            print("Saved %s samples to %s" % (start_idx, args.save_path))
    print("Inference finished.")

if __name__ == "__main__":
    main()