import os

import torch
import mindspeed.megatron_adaptor
import torch.distributed as dist
from megatron.core import mpu
from megatron.training.checkpointing import load_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.tasks.inference.pipeline import sora_pipeline_dict
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import (
    save_videos,
    save_video_grid,
    load_prompts,
    load_images,
    load_conditional_pixel_values,
    load_videos,
)
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
    ori_args = get_args()
    args.pipeline_config.seed = ori_args.seed
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
    sora_pipeline_class = sora_pipeline_dict[args.pipeline_class]
    sora_pipeline = sora_pipeline_class(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                                        predict_model=predict_model, config=args.pipeline_config)
    return sora_pipeline


def main():
    initialize_megatron(extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider), args_defaults={})
    args = get_args()
    merge_mm_args(args)
    if not hasattr(args, "dist_train"):
        args.dist_train = False
    args.mm.model.micro_batch_size = args.micro_batch_size
    args = args.mm.model
    # prepare arguments
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.dtype)
    device = get_device(args.device)

    prompts = load_prompts(args.prompt)

    if not hasattr(args, "dual_image"):
        args.dual_image = False
    images = load_images(args.image, args.dual_image) if hasattr(args, "image") else None

    # Generate args.num_inference_videos_per_sample inference videos for the same prompt.
    if hasattr(args, "num_inference_videos_per_sample") and args.num_inference_videos_per_sample > 1:
        prompts = [
            item
            for item in prompts
            for _ in range(args.num_inference_videos_per_sample)
        ]

        if images is not None:
            images = [
                item
                for item in images
                for _ in range(args.num_inference_videos_per_sample)
            ]

    if hasattr(args, "video"):
        if args.start_frame is None or args.num_frames is None:
            raise ValueError("Please select both starting frame index and total number of frames")
        input_videos = load_videos(args.video, args.start_frame, args.num_frames) if hasattr(args, "video") else None
    else:
        input_videos = None
    conditional_pixel_values_path = load_conditional_pixel_values(args.conditional_pixel_values_path) if hasattr(args, "conditional_pixel_values_path") else None
    mask_type = args.mask_type if hasattr(args, "mask_type") else None
    crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
    max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None
    strength = args.strength if hasattr(args, "strength") else None

    if images is not None and len(prompts) != len(images):
        raise AssertionError(f'The number of images {len(images)} and the numbers of prompts {len(prompts)} do not match')

    if input_videos is not None and len(prompts) != len(input_videos):
        raise AssertionError(f'The number of videos {len(input_videos)} and the numbers of prompts {len(prompts)} do not match')

    if len(prompts) % args.micro_batch_size != 0:
        raise AssertionError(f'The number of  prompts {len(prompts)} is not divisible by the batch size {args.micro_batch_size}')

    save_fps = args.fps // args.frame_interval

    # prepare pipeline
    sora_pipeline = prepare_pipeline(args, device)

    # == Iter over all samples ==
    video_grids = []
    start_idx = 0
    rank = mpu.get_data_parallel_rank()
    world_size = mpu.get_data_parallel_world_size()
    for i in range(rank * args.micro_batch_size, len(prompts), args.micro_batch_size * world_size):
        # == prepare batch prompts ==
        batch_prompts = prompts[i: i + args.micro_batch_size]
        kwargs = {}
        if conditional_pixel_values_path:
            batch_pixel_values_path = conditional_pixel_values_path[i: i + args.micro_batch_size]
            kwargs.update({"conditional_pixel_values_path": batch_pixel_values_path,
                           "mask_type": mask_type,
                           "crop_for_hw": crop_for_hw,
                           "max_hxw": max_hxw})

        if images is not None:
            batch_images = images[i: i + args.micro_batch_size]
        else:
            batch_images = None

        if input_videos is not None:
            batch_videos = input_videos[i: i + args.micro_batch_size]
        else:
            batch_videos = None

        videos = sora_pipeline(prompt=batch_prompts,
                               image=batch_images,
                               video=batch_videos,
                               fps=save_fps,
                               use_prompt_preprocess=args.use_prompt_preprocess,
                               device=device,
                               dtype=dtype,
                               strength=strength,
                               **kwargs
                               )
        if mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            save_videos(videos, i, args.save_path, save_fps)
            start_idx += len(batch_prompts) * world_size
            video_grids.append(videos)
    if len(video_grids) > 0:
        video_grids = torch.cat(video_grids, dim=0).to(device)

    if len(prompts) < args.micro_batch_size * world_size:
        active_ranks = range(len(prompts) // args.micro_batch_size)
    else:
        active_ranks = range(world_size)
    active_ranks = [x * mpu.get_tensor_model_parallel_world_size() * mpu.get_context_parallel_world_size() for x in active_ranks]

    dist.barrier()
    gathered_videos = []
    rank = dist.get_rank()
    if rank == 0:
        for r in active_ranks:
            if r != 0:  # main process does not need to receive from itself
                # receive tensor shape
                shape_tensor = torch.empty(5, dtype=torch.int, device=device)
                dist.recv(shape_tensor, src=r)
                shape_videos = shape_tensor.tolist()

                # create receiving buffer based on received shape
                received_videos = torch.empty(shape_videos, dtype=video_grids.dtype, device=device)
                dist.recv(received_videos, src=r)
                gathered_videos.append(received_videos.cpu())
            else:
                gathered_videos.append(video_grids.cpu())
    elif rank in active_ranks:
        # send tensor shape first
        shape_tensor = torch.tensor(video_grids.shape, dtype=torch.int, device=device)
        dist.send(shape_tensor, dst=0)

        # send the tensor
        dist.send(video_grids, dst=0)
    dist.barrier()
    if rank == 0:
        video_grids = torch.cat(gathered_videos, dim=0)
        save_video_grid(video_grids, args.save_path, save_fps)
        print("Inference finished.")
        print("Saved %s samples to %s" % (video_grids.shape[0], args.save_path))


if __name__ == "__main__":
    main()
