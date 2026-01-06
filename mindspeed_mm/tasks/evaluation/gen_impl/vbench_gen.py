import json
import os

import imageio
import pandas as pd
import torch
from megatron.core import mpu
from megatron.training.utils import print_rank_0
from peft.config import PeftConfigMixin

from mindspeed_mm.data import build_mm_dataloader
from mindspeed_mm.data.data_utils.utils import build_iterations
from mindspeed_mm.tasks.evaluation.gen_impl.base_gen import BaseGenEvalImpl
from mindspeed_mm.tasks.inference.pipeline.utils.sora_utils import safe_load_image
from mindspeed_mm.utils.utils import get_dtype, get_device
from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.compute_score import compute_score
from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_t2v_patch import patch_t2v
from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_i2v_patch import PatchPeftConfigMixin
from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_i2v_patch import evaluate_i2v
from mindspeed_mm.tasks.evaluation.gen_impl.vbench_utils.vbench_long_patch import (patch_static_filter_load_model,
                                                                                    evaluate_long)

RATIO = ["1-1", "8-5", "7-4", "16-9"]


class VbenchGenEvalImpl(BaseGenEvalImpl):
    def __init__(self, dataset, inference_pipeline, args):
        super().__init__(dataset, inference_pipeline, args)
        self.eval_type = args.eval_config.eval_type
        self.result_output_path = args.eval_config.eval_result_path
        self.mode = "vbench_standard" if args.eval_config.eval_type in ["i2v", "t2v"] else "long_vbench_standard"
        self.dimensions = args.eval_config.dimensions
        self.videos_path = args.save_path
        self.load_ckpt_from_local = args.eval_config.load_ckpt_from_local
        self.full_json_dir = args.eval_config.dataset.basic_param.data_path
        self.prompt = getattr(args.eval_config, "prompt", [])
        self.ratio = getattr(args.eval_config.dataset.extra_param, "ratio", "16-9")
        self.vbench = None
        self.image_path = getattr(args.eval_config, "image_path", None)
        self.long_eval_config = getattr(args.eval_config, "long_eval_config", "")
        self.need_inference = getattr(args.eval_config, "need_inference", True)
        self.pipeline = inference_pipeline
        self.eval_args = args
        self.dataset = dataset
        self.full_dimension_list = []
        self.res_score = {}

    def __call__(self):
        if self.need_inference:
            self.inference_video()
        self.analyze_result()
        if self.eval_type == "t2v" or self.eval_type == "long":
            self.compute_t2v_long_score()

    def check_dimension_list(self):
        self.full_dimension_list = self.vbench.build_full_dimension_list()

        if not self.dimensions:
            self.dimensions = self.full_dimension_list

        if not set(self.dimensions).issubset(set(self.full_dimension_list)):
            raise NotImplementedError("Support dimensions contains:{}".format(self.full_dimension_list))

    def save_result_to_excel(self, data):
        excel_res_path = os.path.join(self.result_output_path, "eval_result.xlsx")
        with pd.ExcelWriter(excel_res_path) as writer:
            for key, value in data.items():
                score = value[0]
                rows = []
                result_list = value[1] if key != "camera_motion" else value[2]
                for item in result_list:
                    # split prompt from file path
                    prompt = os.path.splitext(os.path.basename(item["video_path"]))[0][:-2]
                    row = {
                        "total score": score,
                        "prompt": prompt,
                        "video_results": item["video_results"]
                    }
                    rows.append(row)
                    # save res to compute total score
                    self.res_score[key] = score
                df = pd.DataFrame(rows)
                df.to_excel(writer, sheet_name=key, index=False)
        print_rank_0(f"Save excel to {excel_res_path}.")

    def analyze_result(self):
        import vbench2_beta_i2v
        from vbench import VBench
        from vbench2_beta_i2v import VBenchI2V
        from vbench2_beta_long import VBenchLong
        from vbench2_beta_long.static_filter import StaticFilter
        patch_t2v()

        device = torch.device("npu")
        result_file_name = f'{self.eval_type}'

        if "i2v_background" in self.dimensions or (self.mode == "long_vbench_standard" and "background_consistency"
                                                   in self.dimensions):
            PeftConfigMixin.from_pretrained = PatchPeftConfigMixin.from_pretrained

        if self.eval_type == "t2v":
            self.vbench = VBench(device, self.full_json_dir, self.result_output_path)
            self.check_dimension_list()

            self.vbench.evaluate(
                videos_path=self.videos_path,
                name=result_file_name,
                prompt_list=self.prompt,
                dimension_list=self.dimensions,
                local=self.load_ckpt_from_local,
                read_frame=False,
                mode=self.mode
            )
        elif self.eval_type == "i2v":
            vbench2_beta_i2v.utils.load_i2v_dimension_info = self.load_i2v_dimension_info
            self.vbench = VBenchI2V(device, self.full_json_dir, self.result_output_path)
            self.check_dimension_list()
            if self.ratio not in RATIO:
                raise ValueError(f"Not support this ratio {self.ratio}")

            evaluate_i2v(
                self.vbench,
                videos_path=self.videos_path,
                name=result_file_name,
                dimension_list=self.dimensions,
                resolution=self.ratio,
                mode=self.mode
            )
        elif self.eval_type == "long":
            self.vbench = VBenchLong(device, self.full_json_dir, self.result_output_path)
            StaticFilter.load_model = patch_static_filter_load_model
            self.check_dimension_list()

            # Extra params
            kwargs = {"sb_clip2clip_feat_extractor": 'dinov2', "bg_clip2clip_feat_extractor": "dreamsim",
                      "clip_length_config": "clip_length_mix.yaml", "w_inclip": 1.0, "w_clip2clip": 0.0,
                      "use_semantic_splitting": False,
                      "slow_fast_eval_config": os.path.join(self.long_eval_config, "configs/slow_fast_params.yaml"),
                      "sb_mapping_file_path": os.path.join(self.long_eval_config,
                                                            "configs/subject_mapping_table.yaml"),
                      "bg_mapping_file_path": os.path.join(self.long_eval_config,
                                                            "configs/background_mapping_table.yaml"),
                      "dev_flag": True, "num_of_samples_per_prompt": 5, "static_filter_flag": True}
            evaluate_long(
                self.vbench,
                videos_path=self.videos_path,
                name=result_file_name,
                prompt_list=self.prompt,
                dimension_list=self.dimensions,
                local=self.load_ckpt_from_local,
                read_frame=False,
                mode="long_vbench_standard",
                **kwargs
            )
        else:
            raise NotImplementedError("Not support evaluate type.")

        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            result_path = os.path.join(self.result_output_path, result_file_name + '_eval_results.json')
            with open(result_path, 'r', encoding='utf-8') as f:
                self.save_result_to_excel(json.load(f))

        print_rank_0("Evaluation Done.")

    def compute_t2v_long_score(self):
        if torch.distributed.get_rank() == 0:
            res_score_replace_key = {}
            res_dimension_key = []
            for key, value in self.res_score.items():
                res_dimension_key.append(key)
                res_score_replace_key[key.replace("_", " ")] = value
            if sorted(self.full_dimension_list) != sorted(res_dimension_key):
                print_rank_0('Not contain full dimension, can not compute total score.')
            else:
                compute_score(res_score_replace_key)

    def load_i2v_dimension_info(self, json_dir, dimension, lang, resolution):
        video_pair_list = []
        prompt_dict_ls = []
        with open(json_dir, 'r', encoding='utf-8') as f:
            full_prompt_list = json.load(f)

        if not self.image_path:
            raise ValueError("please set image_path in config.")
        image_root = os.path.join(self.image_path, resolution)
        for prompt_dict in full_prompt_list:
            if dimension in prompt_dict['dimension'] and 'video_list' in prompt_dict:
                prompt = prompt_dict[f'prompt_{lang}']
                cur_video_list = prompt_dict['video_list'] if isinstance(prompt_dict['video_list'], list) else [
                    prompt_dict['video_list']]
                # create image-video pair
                if "image_name" in prompt_dict:
                    image_path = os.path.join(image_root, prompt_dict["image_name"])
                elif "custom_image_path" in prompt_dict:
                    image_path = prompt_dict["custom_image_path"]
                else:
                    raise Exception("prompt_dict doesn't contain 'image_name' or 'custom_image_path' key")

                cur_video_pair = [(image_path, video) for video in cur_video_list]
                video_pair_list += cur_video_pair
                if 'auxiliary_info' in prompt_dict and dimension in prompt_dict['auxiliary_info']:
                    prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list,
                                        'auxiliary_info': prompt_dict['auxiliary_info'][dimension]}]
                else:
                    prompt_dict_ls += [{'prompt': prompt, 'video_list': cur_video_list}]
        return video_pair_list, prompt_dict_ls

    def inference_video(self):
        args = self.eval_args
        # prepare arguments
        torch.set_grad_enabled(False)
        dtype = get_dtype(args.dtype)
        device = get_device(args.device)

        # prepare datasets
        eval_dataloader = build_mm_dataloader(
            self.dataset,
            args.eval_config.dataloader_param,
            process_group=mpu.get_data_parallel_group(),
            dataset_param=args.eval_config.dataset,
        )
        data_iterator, _, _ = build_iterations(train_dl=eval_dataloader, iterator_type="single")

        # prepare pipeline
        save_fps = args.fps // args.frame_interval
        mask_type = args.mask_type if hasattr(args, "mask_type") else None
        crop_for_hw = args.crop_for_hw if hasattr(args, "crop_for_hw") else None
        max_hxw = args.max_hxw if hasattr(args, "max_hxw") else None

        image = None
        image_path = None
        for item in data_iterator:
            caption = item["caption"]
            prefix = item["prefix"]
            if self.eval_type == "i2v":
                image_path = item["image"]
                image = safe_load_image(image_path[0].strip())

            kwargs = {}
            if args.pipeline_class == "OpenSoraPlanPipeline" and image_path:
                kwargs.update({"conditional_pixel_values_path": [[path] for path in image_path],
                            "mask_type": mask_type,
                            "crop_for_hw": crop_for_hw,
                            "max_hxw": max_hxw})

            print(f"*** generator video now, eval_type: {self.eval_type}, prompt: {caption}, prefix: {prefix}, image_path: {image_path}")
            videos = self.pipeline(prompt=caption,
                                image=image,
                                fps=save_fps,
                                use_prompt_preprocess=args.use_prompt_preprocess,
                                device=device,
                                dtype=dtype,
                                **kwargs
                                )
            self.save_eval_videos(videos, args.save_path, save_fps, prefix)

    def save_eval_videos(self, videos, save_path, fps, save_names):
        os.makedirs(save_path, exist_ok=True)
        if isinstance(videos, (list, tuple)) or videos.ndim == 5:  # [b, t, h, w, c]
            for i, video in enumerate(videos):
                save_path_i = os.path.join(save_path, f"{save_names[i]}.mp4")
                imageio.mimwrite(save_path_i, video, fps=fps, quality=6)
        elif videos.ndim == 4:
            save_path = os.path.join(save_path, f"{save_names[0]}.mp4")
            imageio.mimwrite(save_path, videos, fps=fps, quality=6)
        else:
            raise ValueError("The video must be in either [b, t, h, w, c] or [t, h, w, c] format.")