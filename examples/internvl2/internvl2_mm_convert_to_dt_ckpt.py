import argparse
import ast
import re
from pathlib import Path
from typing import Optional, Sequence, Any, Tuple, Dict
from copy import deepcopy

import torch
import mindspeed.megatron_adaptor


def get_ckpt_path(path: str) -> Tuple[Path, Optional[str]]:
    """ 
    Determine whether there is iteration information in the path. If so, find the corresponding directory; 
    if not, determine whether the current directory meets the requirements.
    """
    path = Path(path)
    files = [item.name for item in path.iterdir() if item.is_file()]
    if 'latest_checkpointed_iteration.txt' in files:
        iteration = path.joinpath('latest_checkpointed_iteration.txt').read_text('utf-8')
        if iteration.isdigit():
            path_ = path.joinpath(f'iter_{int(iteration):07d}')
        else:
            path_ = path.joinpath(iteration)
        if not Path.exists(path_):
            raise FileNotFoundError(f'Path `{path_}` is not exists.')
        return path_, iteration
    else:
        pattern = re.compile(r'mp_?([a-zA-Z0-9]+)?_rank_(\d{2})(?:_(\d{3}))?')
        dirs = [item.name for item in path.iterdir() if item.is_dir()]
        for dir_name in dirs:
            if len(dir_name) > 100 or not pattern.match(dir_name):
                raise ValueError(f'Found unexpected dir `{dir_name}`')
        return path, None


def get_loaded_ckpt_tp_pp(path: Path, extra_model_name: str = None):
    """ Automatically get the tp and pp of the ckpt to be loaded """
    dirs = [item.name for item in path.iterdir() if item.is_dir()]
    pattern = re.compile(r'mp_?([a-zA-Z0-9]+)?_rank_(\d{2})(?:_(\d{3}))?')
    tp_size, pp_size = 0, 0
    for dir_name in dirs:
        match = pattern.match(dir_name)
        if match:
            if match.group(1) != extra_model_name:
                continue
            tp_size = max(int(match.group(2)) + 1, tp_size)
            pp_size = max(int(match.group(3)) + 1 if match.group(3) is not None else 1, pp_size)
        else:
            raise ValueError(f'Unexpected dir: {dir_name}')
    if tp_size <= 0 or pp_size <= 0:
        raise ValueError(f'tp_size ({tp_size}) or pp_size ({pp_size}) must greater than 0.')
    return tp_size, pp_size


def load_ckpt(
    path, tp_size, pp_size, extra_model_name: str = None
) -> Tuple[Dict[Tuple[int, int], Dict[str, torch.Tensor]], Dict[str, Any]]:
    """ Load ckpt, return {(int, int): {'layers': Tensor}}, {all information in the original state_dict except 'model'} """
    ckpts, params = {}, None
    for tp in range(tp_size):
        for pp in range(pp_size):
            model_name = '' if extra_model_name is None else '_' + extra_model_name
            pp_suffix = '' if pp_size == 1 else f'_{pp:03d}'
            dir_name = 'mp' + model_name + f'_rank_{tp:02d}' + pp_suffix
            state_dict = torch.load(path / dir_name / 'model_optim_rng.pt', map_location='cpu')
            ckpts[(tp, pp)] = deepcopy(state_dict['model'])
            if params is None:
                params = state_dict
                params.pop('model')
    return ckpts, params


def calculate_pp_layer_sizes(ckpt_list, startswith):
    """ Calculate pp_layer_sizes based on ckpts, used in merge_by_pp """
    pp_layer_sizes = [0 for _ in range(len(ckpt_list))]
    j = len(startswith.split('.'))
    for i, ckpt in enumerate(ckpt_list):
        for k in ckpt.keys():
            if k.startswith(startswith):
                pp_layer_sizes[i] = max(pp_layer_sizes[i], int(k.split('.')[j]) + 1)
    return pp_layer_sizes


def _cumulative_sum(pp_layer_sizes):
    """ Calculate the starting layer_index of each pp_layer """
    result = [0]
    for num in pp_layer_sizes:
        result.append(result[-1] + num)
    return result


def _get_key_startswith_index_and_str(key, prefix_list):
    """ Return which prefix_list the key corresponds to, and return the index and prefix """
    for i, prefix in enumerate(prefix_list):
        if key.startswith(prefix):
            return i, prefix
    return -1, None


def _merge_by_pp(ori_ckpt, new_ckpt, cur_pp_idx, keys_full_prefix_on_pp_layer, pp_layer_start_index):
    for k, v in ori_ckpt.items():
        startswith_list_index, startswith = _get_key_startswith_index_and_str(k, keys_full_prefix_on_pp_layer)
        if startswith_list_index == -1:
            new_ckpt[k] = v
            continue
        ori_layer_num_index = len(keys_full_prefix_on_pp_layer[startswith_list_index].split('.'))
        ori_layer_num = int(k.split('.')[ori_layer_num_index])
        new_layer_num = ori_layer_num + pp_layer_start_index[startswith][cur_pp_idx]
        k_split = k.split('.')
        k_split[ori_layer_num_index] = str(new_layer_num)
        new_key = '.'.join(k_split)
        new_ckpt[new_key] = v


def merge_by_pp(ckpts, tp_size, pp_size, keys_full_prefix_on_pp_layer: Sequence[str]):
    """ Merge ckpts by pp """
    new_ckpts = {}
    for tp in range(tp_size):
        ckpt = deepcopy(ckpts[(tp, 0)])
        ckpts_tp = [ckpts[(tp, pp)] for pp in range(pp_size)]
        pp_layer_start_index = {}
        for startswith in keys_full_prefix_on_pp_layer:
            pp_layer_sizes = calculate_pp_layer_sizes(ckpts_tp, startswith)
            pp_layer_start_index[startswith] = _cumulative_sum(pp_layer_sizes)

        for pp in range(1, pp_size):
            ckpt_ = deepcopy(ckpts_tp[pp])
            _merge_by_pp(ckpt_, ckpt, pp, keys_full_prefix_on_pp_layer, pp_layer_start_index)

        new_ckpts[(tp, 0)] = ckpt

    del ckpts
    return new_ckpts


def _merge_by_tp(ori_ckpt, new_ckpt, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):
    for key, tensor in ori_ckpt.items():
        v_copy = deepcopy(tensor)
        if any(key.startswith(start) and key.endswith(end)
               for start, end_list in keys_part_on_tp_dim_0.items()
               for end in end_list):
            new_ckpt[key] = torch.cat((new_ckpt[key], v_copy), dim=0)
        elif any(key.startswith(start) and key.endswith(end)
                 for start, end_list in keys_part_on_tp_dim_1.items()
                 for end in end_list):
            new_ckpt[key] = torch.cat((new_ckpt[key], v_copy), dim=1)
        else:
            new_ckpt[key] = v_copy


def merge_by_tp(ckpts, tp_size, pp_size, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):
    """ Merge ckpts by tp """
    new_ckpts = {}
    for pp in range(pp_size):
        ckpt = ckpts[(0, pp)]
        ckpts_pp = [ckpts[(tp, pp)] for tp in range(tp_size)]

        for tp in range(1, tp_size):
            ckpt_ = ckpts_pp[tp]
            _merge_by_tp(ckpt_, ckpt, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)

        new_ckpts[(0, pp)] = ckpt

    new_ckpts = deepcopy(new_ckpts)
    del ckpts
    return new_ckpts


def unmerge_to_dt(ckpts: dict, prefix: str = None, not_prefix: str = None):
    """ Unmerge ckpt """
    if not (isinstance(ckpts, dict) and len(ckpts) == 1 and list(ckpts.keys())[0] == (0, 0)):
        raise ValueError('Except merged `ckpts` with tp size 1 and pp size 1 like `{(0, 0): {\'layers\': Tensor}}`')

    ckpts_ = deepcopy(ckpts)
    keys = deepcopy(list(ckpts[(0, 0)].keys()))
    for k in keys:
        if prefix is not None:
            if k.startswith(prefix):
                ckpts[(0, 0)].pop(k)
            else:
                ckpts_[(0, 0)].pop(k)

        if not_prefix is not None:
            if k.startswith(not_prefix):
                ckpts[(0, 0)].pop(k)

    return ckpts_


def merge_to_mm(ckpts_list, tp_size, pp_size):
    """ Merge ckpts """
    keys = {(tp, pp) for tp in range(tp_size) for pp in range(pp_size)}
    if not all(set(ckpts.keys()).issubset(keys) for ckpts in ckpts_list[1:]):
        raise ValueError('Different keys `(tp, pp)` on ckpts_list\'items, can not merge.')
    keys = deepcopy(list(keys))

    ckpts_ = {}
    for k in keys:
        ckpt = ckpts_list[0].get(k, {})
        for ckpts in ckpts_list[1:]:
            ckpt.update(ckpts.get(k, {}))
        ckpts_[k] = ckpt
    ckpts_ = deepcopy(ckpts_)
    del ckpts_list
    return ckpts_


def _split_by_tp(new_ckpt, ori_ckpt, cur_tp_idx, tp_size, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):
    for key, tensor in ori_ckpt.items():
        if any(key.startswith(start) and key.endswith(end)
               for start, end_list in keys_part_on_tp_dim_0.items()
               for end in end_list):
            new_ckpt[key] = torch.chunk(tensor, tp_size, dim=0)[cur_tp_idx]
        elif any(key.startswith(start) and key.endswith(end)
                 for start, end_list in keys_part_on_tp_dim_1.items()
                 for end in end_list):
            new_ckpt[key] = torch.chunk(tensor, tp_size, dim=1)[cur_tp_idx]
        else:
            new_ckpt[key] = tensor


def split_by_tp(ckpts, tp_size, pp_size, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1):
    """ Split ckpts by tp """
    if not all(0 == tp_pp[0] for tp_pp in ckpts.keys()):
        raise AssertionError('Make sure tp_size of inputted ckpts equals 1.')
    new_ckpts = {}
    for pp in range(pp_size):
        ckpt = ckpts[(0, pp)]

        for tp in range(tp_size):
            ckpt_ = {}
            _split_by_tp(ckpt_, ckpt, tp, tp_size, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)

            new_ckpts[(tp, pp)] = ckpt_

    new_ckpts = deepcopy(new_ckpts)
    del ckpts
    return new_ckpts


def _split_by_pp(new_ckpt, ori_ckpt, cur_pp_idx, start_layers, is_last, is_first,
                 keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess):
    for k, v in ori_ckpt.items():
        mid_index, mid_prefix = _get_key_startswith_index_and_str(k, keys_full_prefix_on_pp_process)
        post_index, post_prefix = _get_key_startswith_index_and_str(k, keys_full_prefix_on_pp_postprocess)

        if mid_index >= 0:
            ori_layer_index_index = len(mid_prefix.split('.'))
            ori_layer_index = int(k.split('.')[ori_layer_index_index])
            if start_layers[cur_pp_idx] <= ori_layer_index < start_layers[cur_pp_idx + 1]:
                new_layer_index = ori_layer_index - start_layers[cur_pp_idx]
                k_split = k.split('.')
                k_split[ori_layer_index_index] = str(new_layer_index)
                new_key = '.'.join(k_split)
                new_ckpt[new_key] = v
        elif post_index >= 0:
            if is_last:
                new_ckpt[k] = v
        else:
            if is_first:
                new_ckpt[k] = v


def split_by_pp(ckpts, tp_size, pp_layers, keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess):
    """ Split ckpts by pp """
    if not all(0 == tp_pp[1] for tp_pp in ckpts.keys()):
        raise AssertionError('Make sure pp_size of inputted ckpts equals 1.')
    if isinstance(pp_layers, Sequence) and not pp_layers:
        pp_layers = [0]
    pp_size = len(pp_layers)
    new_ckpts = {}
    positive_indices = [i for i, x in enumerate(pp_layers) if x > 0]
    # 获取首个和末个大于0的数字的index
    pp_layers_first_index = positive_indices[0] if positive_indices else None
    pp_layers_last_index = positive_indices[-1] if positive_indices else None
    start_layers = _cumulative_sum(pp_layers)
    for tp in range(tp_size):
        ckpt = ckpts[(tp, 0)]

        for pp in range(pp_size):
            if pp not in positive_indices:
                continue

            ckpt_ = {}
            is_first = pp == pp_layers_first_index
            is_last = pp == pp_layers_last_index

            _split_by_pp(ckpt_, ckpt, pp, start_layers, is_last, is_first,
                         keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess)

            new_ckpts[(tp, pp)] = ckpt_

    new_ckpts = deepcopy(new_ckpts)
    del ckpts
    return new_ckpts


def add_extra_params(ckpts, params, tp_size, pp_size, cp_size):
    """ Add extra params to ckpts """
    state_dict_dict = {}
    for k in ckpts.keys():
        new_state_dict = {'model': ckpts[k]}
        new_state_dict.update(deepcopy(params))
        if 'args' in new_state_dict.keys():
            new_state_dict['args'].tensor_model_parallel_size = tp_size
            new_state_dict['args'].pipeline_model_parallel_size = pp_size
            new_state_dict['args'].context_parallel_size = cp_size
        state_dict_dict[k] = new_state_dict
    return state_dict_dict


def save_by_pp_tp(path, state_dict_dict, pp_size, model_name=None, iteration='release', exist_ok=False):
    """ Save ckpt """
    path = Path(path)
    if path.exists():
        if path.is_dir():
            if any(path.iterdir()) and not exist_ok:
                raise FileExistsError(f'Not an empty path: {path}')
        else:
            raise NotADirectoryError(f'Not a path: {path}')
    else:
        path.mkdir()

    iteration_path = path / 'latest_checkpointed_iteration.txt'
    iteration_path.touch()
    with iteration_path.open(mode='w', encoding='utf-8') as f:
        f.write(iteration)

    if iteration == 'release':
        directory = 'release'
    else:
        if not iteration.isdigit():
            raise ValueError(f'Invalid latest_checkpointed_iteration: {iteration}')
        directory = f'iter_{int(iteration):07d}'

    path = path / directory
    path.mkdir(exist_ok=exist_ok)
    for k, state_dict in state_dict_dict.items():
        _model_name = '' if model_name is None else '_' + model_name
        _pp_suffix = f'_{k[1]:03d}' if pp_size > 1 else ''
        folder_name = f'mp{_model_name}_rank_{k[0]:02d}{_pp_suffix}'
        save_path = path / folder_name
        save_path.mkdir(exist_ok=False)
        torch.save(state_dict, save_path / 'model_optim_rng.pt')


def print_keys(ckpts):
    for k, v in ckpts.items():
        print(k)
        for kk, vv in sorted(v.items()):
            info = kk
            if isinstance(vv, torch.Tensor):
                info += f' {vv.shape} {vv.sum()}'
            else:
                info += f' {type(vv)}'
            print(info)
        print('-' * 30)


def main(args):
    # pp keys
    keys_full_prefix_on_pp_layer = ['image_encoder.encoder.encoder.layers', 'text_decoder.decoder.layers']
    keys_full_prefix_on_pp_process = ['text_decoder.decoder.layers', 'image_encoder.encoder.encoder.layers']
    keys_full_prefix_on_pp_postprocess = ['text_decoder.decoder.final_layernorm', 'text_decoder.output_layer',
                                          'image_encoder.projector']
    # tp keys
    keys_part_on_tp_dim_0 = {
        'image': ['fc1.weight', 'linear_qkv.weight', 'output_layer.weight', 'linear_qkv.lora_B',
                  'word_embeddings.weight', 'fc1.bias', 'linear_qkv.bias'],
        'text': ['fc1.weight', 'linear_qkv.weight', 'output_layer.weight', 'linear_qkv.lora_B',
                 'word_embeddings.weight']
    }
    keys_part_on_tp_dim_1 = {
        'image': ['linear_proj.lora_A', 'fc2.weight', 'self_attention.linear_proj.weight'],
        'text': ['linear_proj.lora_A', 'fc2.weight', 'linear_qkv.bias', 'self_attention.linear_proj.weight', 'fc1.bias']
    }

    # mm part
    path, iteration = get_ckpt_path(args.load_dir)
    if iteration is None:
        iteration = 'release'
    tp_size, pp_size = get_loaded_ckpt_tp_pp(path)
    print(f'Get saved ckpts have {tp_size=} {pp_size=} {iteration=}, prepare to loading.')
    ckpts, params = load_ckpt(path, tp_size, pp_size)
    args_gpt = getattr(params, 'args', None)
    if args_gpt is not None:
        if tp_size != args_gpt.tensor_model_parallel_size:
            raise ValueError(f'tp_size ({tp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.tensor_model_parallel_size}).')
        if pp_size != args_gpt.pipeline_model_parallel_size:
            raise ValueError(f'pp_size ({pp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.pipeline_model_parallel_size}).')
    print('Ckpts loaded.')
    ckpts = merge_by_pp(ckpts, tp_size, pp_size, keys_full_prefix_on_pp_layer)
    print('Ckpts merged by pp.')
    ckpts = merge_by_tp(ckpts, tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Ckpts merged by tp.')
    print_keys(ckpts)

    # vit part
    ckpts_vit = unmerge_to_dt(ckpts, 'image_encoder')
    print('Get vit ckpts.')
    ckpts_vit = split_by_tp(ckpts_vit, args.target_vit_tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Vit ckpts split by tp.')
    ckpts_vit = split_by_pp(ckpts_vit, args.target_vit_tp_size, args.target_vit_pp_layers,
                            keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess)
    print('Vit ckpts split by pp.')
    state_dicts_vit = add_extra_params(ckpts_vit, params, args.target_vit_tp_size, args.target_vit_pp_size,
                                       args.target_vit_cp_size)
    save_by_pp_tp(args.save_dir, state_dicts_vit, args.target_vit_pp_size, 'vit', iteration)
    print('Vit ckpts saved.')
    print_keys(ckpts_vit)

    # gpt part
    ckpts_gpt = unmerge_to_dt(ckpts, 'text_decoder')
    print('Get gpt ckpts.')
    ckpts_gpt = split_by_tp(ckpts_gpt, args.target_gpt_tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Gpt ckpts split by tp.')
    ckpts_gpt = split_by_pp(ckpts_gpt, args.target_gpt_tp_size, args.target_gpt_pp_layers,
                            keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess)
    print('Gpt ckpts split by pp.')
    state_dicts_gpt = add_extra_params(ckpts_gpt, params, args.target_gpt_tp_size, args.target_gpt_pp_size,
                                       args.target_gpt_cp_size)
    save_by_pp_tp(args.save_dir, state_dicts_gpt, args.target_gpt_pp_size, 'gpt', iteration, True)
    print('Gpt ckpts saved.')
    print_keys(ckpts_gpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistTrain Checkpoint Utility Arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path for storing the CKPT files to be loaded')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path for storing the converted CKPT files.')
    parser.add_argument('--target-vit-tp-size', type=int, required=True,
                        help='TP size of the vit part to be converted.')
    parser.add_argument('--target-gpt-tp-size', type=int, required=True,
                        help='TP size of the gpt part to be converted.')
    parser.add_argument('--target-vit-pp-size', type=int, required=True,
                        help='PP size of the vit part to be converted.')
    parser.add_argument('--target-gpt-pp-size', type=int, required=True,
                        help='PP size of the gpt part to be converted.')
    parser.add_argument('--target-vit-cp-size', type=int, required=True,
                        help='CP size of the vit part to be converted.')
    parser.add_argument('--target-gpt-cp-size', type=int, required=True,
                        help='CP size of the gpt part to be converted.')
    parser.add_argument('--target-vit-pp-layers', type=str, required=True,
                        help='PP layers of the vit part to be converted.')
    parser.add_argument('--target-gpt-pp-layers', type=str, required=True,
                        help='PP layers of the gpt part to be converted.')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")
    args.target_vit_pp_layers = ast.literal_eval(args.target_vit_pp_layers)
    args.target_gpt_pp_layers = ast.literal_eval(args.target_gpt_pp_layers)
    if len(args.target_vit_pp_layers) != args.target_vit_pp_size:
        raise ValueError(f'len({args.target_vit_pp_layers}) must equals to {args.target_vit_pp_size=}.')
    if len(args.target_gpt_pp_layers) != args.target_gpt_pp_size:
        raise ValueError(f'len({args.target_gpt_pp_layers}) must equals to {args.target_gpt_pp_size=}.')

    main(args)
