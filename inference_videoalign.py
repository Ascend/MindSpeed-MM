import os
import json

import torch
import pandas as pd
import numpy as np

import mindspeed.megatron_adaptor
from mindspeed.megatron_adaptor import get_mindspeed_args
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from mindspeed_mm.configs.config import merge_mm_args
from mindspeed_mm.tasks.inference.pipeline.videoalign_pipeline import VideoAlignPipeline
from mindspeed_mm.tasks.evaluation.reward_impl.utils.eval_accuracy import calc_accuracy_with_ties, \
    calc_accuracy_without_ties
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.arguments import extra_args_provider_decorator
from mindspeed_mm.patchs.patch_manager import PatchesManager
from mindspeed_mm.utils.security_utils.input_filter import sanitize_dataframe

mindspeed_args = get_mindspeed_args()

if hasattr(mindspeed_args,
           "ai_framework") and mindspeed_args.ai_framework == "mindspore" and mindspeed_args.optimization_level >= 0:
    import mindspeed_mm.mindspore.mindspore_adaptor


def convert_pair_to_single(df_pair_anno):
    df_A = df_pair_anno[['path_A', 'prompt', 'fps_A', 'num_frames_A']]
    df_A.columns = ['path', 'prompt', 'fps', 'num_frames']

    df_B = df_pair_anno[['path_B', 'prompt', 'fps_B', 'num_frames_B']]
    df_B.columns = ['path', 'prompt', 'fps', 'num_frames']

    df_single = pd.concat([df_A, df_B], axis=0)
    df_single = df_single.drop_duplicates(subset=['path'])
    df_single = df_single.sort_values(by=['path'])

    df_single = df_single.reset_index(drop=True)

    return df_single


def convert_single_to_pair(df_pair_anno, df_single_pred):
    score_dict = {}
    keys_to_store = ["reward_VQ", "reward_MQ", "reward_TA", "reward_Overall"]

    for _, row in df_single_pred.iterrows():
        score_dict[row["path"]] = {k: row[k] for k in keys_to_store}

    for key in keys_to_store:
        df_pair_anno[f"{key}_A"] = 0.0
        df_pair_anno[f"{key}_B"] = 0.0

    for i, row in df_pair_anno.iterrows():
        for key in keys_to_store:
            df_pair_anno.at[i, f"{key}_A"] = score_dict[row["path_A"]][key]
            df_pair_anno.at[i, f"{key}_B"] = score_dict[row["path_B"]][key]

    return df_pair_anno


def main():
    # just inference
    torch.set_grad_enabled(False)

    initialize_megatron(
        extra_args_provider=extra_args_provider_decorator(mm_extra_args_provider),
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    )
    args = get_args()
    merge_mm_args(args)
    # apply patches
    PatchesManager.apply_patches_from_config()
    if not hasattr(args, "dist_train"):
        args.dist_train = False
    model_config = args.mm.model
    data_config = args.mm.data
    data_params = data_config.to_dict()

    preprocess_params = data_params['dataset_param'].get('preprocess_parameters', None)
    if not preprocess_params:
        raise ValueError('No preprocess_parameters found in data_config!')

    inference_param = getattr(data_config, 'inference_param', None)
    if not inference_param:
        raise ValueError('No inference_param found in data_config!')
    data_path = getattr(inference_param, 'data_path', None)
    data_folder = getattr(inference_param, 'data_folder', None)
    save_path = getattr(inference_param, 'save_path', None)
    if not data_path or not data_folder or not save_path:
        raise ValueError('data_path | data_folder | save_path are not found in inference_param of data_config!')
    os.makedirs(save_path, exist_ok=True)
    task = getattr(inference_param, 'task', None)
    if task not in ['inference', 'evaluate']:
        raise ValueError(f'task:{task} is not in [inference | evaluate], please choose right task!')
    use_norm = getattr(inference_param, 'use_norm', False)
    norm_param = getattr(inference_param, 'norm_param', None)
    norm_param = norm_param.to_dict() if norm_param else norm_param

    batch_size = args.micro_batch_size
    if task == 'evaluate':
        df_input_pair = pd.read_csv(data_path)
        df_infer_single = convert_pair_to_single(df_input_pair)
    elif task == 'inference':
        df_infer_single = pd.read_csv(data_path)

    df_infer_single["reward_VQ"] = 0.0
    df_infer_single["reward_MQ"] = 0.0
    df_infer_single["reward_TA"] = 0.0
    df_infer_single["reward_Overall"] = 0.0

    data_num = len(df_infer_single)
    reward_pipeline = VideoAlignPipeline(model_config, preprocess_params, norm_param)
    for idx in range(0, data_num, batch_size):
        end_idx = min(idx + batch_size, data_num)
        batch_indices = np.arange(idx, end_idx)
        batch_data = df_infer_single.loc[idx:end_idx - 1, ['path', 'prompt', 'fps', 'num_frames']].to_numpy()
        rewards = reward_pipeline(data_folder, batch_data, use_norm)

        for i, batch_idx in enumerate(batch_indices):
            df_infer_single.loc[batch_idx, 'reward_VQ'] = rewards[i]['VQ']
            df_infer_single.loc[batch_idx, 'reward_MQ'] = rewards[i]['MQ']
            df_infer_single.loc[batch_idx, 'reward_TA'] = rewards[i]['TA']
            df_infer_single.loc[batch_idx, 'reward_Overall'] = rewards[i]['Overall']
            print(
                f"{df_infer_single.loc[batch_idx, 'path']} reward: VQ:{rewards[i]['VQ']}, MQ:{rewards[i]['MQ']}, TA:{rewards[i]['TA']}, Overall:{rewards[i]['Overall']}")

    df_infer_single_ = sanitize_dataframe(df_infer_single)
    df_infer_single_.to_excel(os.path.join(save_path, 'reward_out_single.xlsx'), index=False, engine='openpyxl')

    if task == 'evaluate':
        df_infer_pair = convert_single_to_pair(df_input_pair, df_infer_single)
        df_infer_pair_ = sanitize_dataframe(df_infer_pair)
        df_infer_pair_.to_excel(os.path.join(save_path, "reward_out_pair.xlsx"), index=False, engine='openpyxl')

        reward_attributes = getattr(inference_param, 'reward_attributes', ["VQ", "MQ", "TA", "Overall"])
        eval_results = {}
        for reward_attr in reward_attributes:
            df_infer_pair[f'reward_{reward_attr}'] = df_infer_pair[f"reward_{reward_attr}_A"] - df_infer_pair[
                f"reward_{reward_attr}_B"]
            df_infer_pair[f"{reward_attr}"] = df_infer_pair[f"{reward_attr}"].map({'A': 1, 'B': -1, 'same': 0})

            eval_results[f"{reward_attr} Accuracy"] = {
                "with_ties": calc_accuracy_with_ties(df_infer_pair[f"{reward_attr}"],
                                                     df_infer_pair[f"reward_{reward_attr}"]),
                "without_ties": calc_accuracy_without_ties(df_infer_pair[f"{reward_attr}"],
                                                           df_infer_pair[f"reward_{reward_attr}"])
            }
            print(f"{reward_attr} Accuracy: ", end="")
            print(f"With ties: {eval_results[f'{reward_attr} Accuracy']['with_ties']}, ", end="")
            print(f"Without ties: {eval_results[f'{reward_attr} Accuracy']['without_ties']}")

        with open(os.path.join(save_path, "eval_accuracy.json"), "w") as f:
            json.dump(eval_results, f, indent=4)

    print('Reward over')


if __name__ == '__main__':
    main()