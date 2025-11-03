import os
import importlib
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as ddp


def evaluate_long(self, videos_path, name, prompt_list=None, dimension_list=None, local=False, read_frame=False,
             mode='vbench_standard', **kwargs):
    from vbench.utils import init_submodules, save_json
    from vbench.distributed import get_rank, print0
    from vbench2_beta_long import subject_consistency, background_consistency

    _dimensions = self.build_full_dimension_list()
    is_dimensional_structure = any(os.path.isdir(os.path.join(videos_path, dim)) for dim in _dimensions)
    kwargs['preprocess_dimension_flag'] = dimension_list
    if is_dimensional_structure:
        # 1. Under dimensions folders
        for dimension in _dimensions:
            dimension_path = os.path.join(videos_path, dimension)
            self.preprocess(dimension_path, mode, **kwargs)
    else:
        self.preprocess(videos_path, mode, **kwargs)

    # Now, long videos have been split into clips
    results_dict = {}
    if dimension_list is None:
        dimension_list = self.build_full_dimension_list()
    submodules_dict = init_submodules(dimension_list, local=local, read_frame=read_frame)
    # loop for build_full_info_json for clips
    cur_full_info_path = self.build_full_info_json(videos_path, name, dimension_list, prompt_list, mode=mode, **kwargs)

    subject_consistency.compute_long_subject_consistency = patch_compute_long_subject_consistency
    background_consistency.compute_long_background_consistency = patch_compute_long_background_consistency

    for dimension in dimension_list:
        try:
            dimension_module = importlib.import_module(f'vbench2_beta_long.{dimension}')
            evaluate_func = getattr(dimension_module, f'compute_long_{dimension}')
        except Exception as e:
            raise NotImplementedError(f'UnImplemented dimension {dimension}!, {e}') from e
        submodules_list = submodules_dict[dimension]
        print(f'cur_full_info_path: {cur_full_info_path}')

        results = evaluate_func(cur_full_info_path, self.device, submodules_list, **kwargs)
        results_dict[dimension] = results
    output_name = os.path.join(self.output_path, name + '_eval_results.json')
    if get_rank() == 0:
        save_json(results_dict, output_name)
        print0(f'Evaluation results saved to {output_name}')


def patch_compute_long_subject_consistency(json_dir, device, submodules_list, **kwargs):
    from vbench.subject_consistency import compute_subject_consistency, subject_consistency
    from vbench2_beta_long.utils import reorganize_clips_results, create_video_from_first_frames, fuse_inclip_clip2clip
    from vbench2_beta_long.subject_consistency import subject_consistency_dinov2, subject_consistency_dreamsim
    from dreamsim import dreamsim
    # compute inclip scores
    all_results, detailed_results = compute_subject_consistency(json_dir, device, submodules_list)

    inclip_all_results, inclip_detailed_results, inclip_average_scores = reorganize_clips_results(detailed_results)
    inclip_all_results = all_results

    # compute clip2clip scores
    # sample first frames in each clip, and cat them into a new video
    base_path_video = os.path.dirname(list(detailed_results[0].values())[0]).split("split_clip")[0]
    long_video_path = os.path.join(base_path_video, "split_clip")
    new_cat_video_path = os.path.join(base_path_video, 'subject_consistency_cat_firstframes_videos')
    if not os.path.exists(new_cat_video_path):
        os.makedirs(new_cat_video_path, exist_ok=True)
        create_video_from_first_frames(long_video_path, new_cat_video_path, detailed_results)
    else:
        print(f"{new_cat_video_path} has already been created, please check the path")

    # get the new video_list
    video_list = []
    for video_path in os.listdir(new_cat_video_path):
        video_list.append(os.path.join(new_cat_video_path, video_path))

    def _compute_subject_consistency(video_list, device, submodules_list, **kwargs):
        if kwargs['sb_clip2clip_feat_extractor'] == 'dino':
            dino_model = torch.hub.load(**submodules_list).to(device)
            read_frame = submodules_list['read_frame']
            print("Initialize DINO success")
            all_results, video_results = subject_consistency(dino_model, video_list, device, read_frame)
        elif kwargs['sb_clip2clip_feat_extractor'] == 'dinov2':
            dinov2_dict = {
                'repo_or_dir': f'facebookresearch/dinov2',
                'model': 'dinov2_vitb14',
            }
            dinov2_model = torch.hub.load(**dinov2_dict).to(device)
            read_frame = submodules_list['read_frame']
            print("Initialize DINOv2 success")
            all_results, video_results = subject_consistency_dinov2(dinov2_model, video_list, device, read_frame)

        elif kwargs['sb_clip2clip_feat_extractor'] == 'dreamsim':
            read_frame = submodules_list['read_frame']
            dreamsim_model, _ = dreamsim(pretrained=True, cache_dir="./models")
            all_results, video_results = subject_consistency_dreamsim(dreamsim_model, video_list, device, read_frame)
        return all_results, video_results

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        clip2clip_all_results, clip2clip_detailed_results = _compute_subject_consistency(video_list, device,
                                                                                         submodules_list, **kwargs)
        dimension = 'subject_consistency'
        fused_all_results, fused_detailed_results = fuse_inclip_clip2clip(inclip_all_results, clip2clip_all_results,
                                                                          inclip_average_scores, clip2clip_detailed_results,
                                                                          dimension, **kwargs)
        return fused_all_results, fused_detailed_results


def patch_compute_long_background_consistency(json_dir, device, submodules_list, **kwargs):
    from vbench.background_consistency import compute_background_consistency, background_consistency
    from vbench2_beta_long.utils import reorganize_clips_results, create_video_from_first_frames, fuse_inclip_clip2clip
    from dreamsim import dreamsim
    from vbench2_beta_long.background_consistency import background_consistency_dreamsim
    import clip

    # compute inclip scores
    all_results, detailed_results = compute_background_consistency(json_dir, device, submodules_list)

    inclip_all_results, inclip_detailed_results, inclip_average_scores = reorganize_clips_results(detailed_results)

    # compute clip2clip scores
    # sample first frames in each clip, and cat them into a new video
    base_path_video = os.path.dirname(list(detailed_results[0].values())[0]).split("split_clip")[0]
    long_video_path = os.path.join(base_path_video, "split_clip")
    new_cat_video_path = os.path.join(base_path_video, 'background_consistency_cat_firstframes_videos')
    if not os.path.exists(new_cat_video_path):
        os.makedirs(new_cat_video_path, exist_ok=True)
        create_video_from_first_frames(long_video_path, new_cat_video_path, detailed_results)
    else:
        print(f"{new_cat_video_path} has already been created, please check the path")

    # get the new video_list
    video_list = []
    for video_path in os.listdir(new_cat_video_path):
        video_list.append(os.path.join(new_cat_video_path, video_path))

    def _compute_background_consistency(video_list, device, submodules_list, **kwargs):
        if kwargs['bg_clip2clip_feat_extractor'] == 'clip':
            vit_path, read_frame = submodules_list[0], submodules_list[1]
            clip_model, preprocess = clip.load(vit_path, device=device)
            all_results, video_results = background_consistency(clip_model, preprocess, video_list, device, read_frame)
        elif kwargs['bg_clip2clip_feat_extractor'] == 'dreamsim':
            read_frame = submodules_list[1]
            dreamsim_model, preprocess = dreamsim(pretrained=True, cache_dir="./models")
            all_results, video_results = background_consistency_dreamsim(dreamsim_model, preprocess, video_list, device,
                                                                         read_frame)
        return all_results, video_results

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        clip2clip_all_results, clip2clip_detailed_results = _compute_background_consistency(video_list, device,
                                                                                            submodules_list, **kwargs)

        dimension = 'background_consistency'
        fused_all_results, fused_detailed_results = fuse_inclip_clip2clip(inclip_all_results, clip2clip_all_results,
                                                                          inclip_average_scores, clip2clip_detailed_results,
                                                                          dimension, **kwargs)
        return fused_all_results, fused_detailed_results


def patch_static_filter_load_model(self):
    from vbench.third_party.RAFT.core.raft import RAFT

    self.model = ddp(RAFT(self.args).to(self.device))
    self.model.load_state_dict(torch.load(self.args.model))

    self.model = self.model.module
    self.model.eval()

