import argparse
import ast
import re
from pathlib import Path
from typing import Optional, Sequence, Any, Tuple, Dict
from copy import deepcopy

import torch
import mindspeed.megatron_adaptor


def get_ckpt_path(path: str) -> Tuple[Path, Optional[str]]:
    """ 判断path内是否有iteration信息，如果有，去找对应目录，判断对应目录下是否有符合要求的目录；如果没有，则判断本目录是否已经满足要求 """
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
            if not pattern.match(dir_name):
                raise ValueError(f'Found unexpected dir `{dir_name}`')
        return path, None


def get_loaded_ckpt_tp_pp(path: Path, extra_model_name: str = None):
    """ 自动获取待加载的ckpt的tp和pp """
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
    """ 加载ckpt，返回{(int, int): {'layers': Tensor}}, {原来state_dict中除了'model'以外的所有信息} """
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
    """ 用来根据ckpts计算pp_layer_sizes，在merge_by_pp中使用 """
    pp_layer_sizes = [0 for _ in range(len(ckpt_list))]
    j = len(startswith.split('.'))
    for i, ckpt in enumerate(ckpt_list):
        for k in ckpt.keys():
            if k.startswith(startswith):
                pp_layer_sizes[i] = max(pp_layer_sizes[i], int(k.split('.')[j]) + 1)
    return pp_layer_sizes


def _cumulative_sum(pp_layer_sizes):
    """ 计算每个pp_layer的起始layer_index """
    result = [0]  # 初始化结果列表，第一个元素为0
    for num in pp_layer_sizes:
        result.append(result[-1] + num)  # 将当前元素与结果列表的最后一个元素相加，并添加到结果列表中
    return result


def _get_key_startswith_index_and_str(key, prefix_list):
    """ 返回key对应prefix_list中的哪个，并返回这个index和prefix """
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
    """ 沿着pp合并ckpt """
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
    """ 沿着tp合并ckpt """
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
    """ 沿着tp切分ckpt """
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
    """ 沿着pp分开ckpt """
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
    """ 将最初从state_dict中pop掉的内容给加回来 """
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
    """ 保存ckpt """
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
    keys_full_prefix_on_pp_layer = ['videodit_sparse_blocks']
    keys_full_prefix_on_pp_process = ['videodit_sparse_blocks']
    keys_full_prefix_on_pp_postprocess = ['scale_shift_table', 'proj_out.weight', 'proj_out.bias']
    # tp keys
    keys_part_on_tp_dim_0 = {
        'videodit_sparse_blocks': ["atten.proj_q.weight", "atten.proj_q.bias", "atten.proj_k.weight",
                                   "atten.proj_k.bias", "atten.proj_v.weight", "atten.proj_v.bias",
                                   "ff.net.0.proj.weight", "ff.net.0.proj.bias"]
    }
    keys_part_on_tp_dim_1 = {
        'videodit_sparse_blocks': ["atten.proj_out.weight", "ff.net.2.weight"]
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

    # vae part
    ckpts_vae = {(0, 0): None}
    state_dicts_vae = add_extra_params(ckpts_vae, params, args.target_vae_tp_size, args.target_vae_pp_size,
                                       args.target_vae_cp_size)
    save_by_pp_tp(args.save_dir, state_dicts_vae, max(1, len(args.target_vae_pp_layers)), 'vae', iteration)
    print('Empty vae ckpts saved.')

    # dit part
    ckpts_dit = ckpts
    print('Get dit ckpts.')
    ckpts_dit = split_by_tp(ckpts_dit, args.target_dit_tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Dit ckpts split by tp.')
    ckpts_dit = split_by_pp(ckpts_dit, args.target_dit_tp_size, args.target_dit_pp_layers,
                            keys_full_prefix_on_pp_process, keys_full_prefix_on_pp_postprocess)
    print('Dit ckpts split by pp.')
    state_dicts_dit = add_extra_params(ckpts_dit, params, args.target_dit_tp_size, args.target_dit_pp_size,
                                       args.target_dit_cp_size)
    save_by_pp_tp(args.save_dir, state_dicts_dit, max(1, len(args.target_dit_pp_layers)), 'dit', iteration, True)
    print('Dit ckpts saved.')
    print_keys(ckpts_dit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DistTrain Checkpoint Utility Arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path for storing the CKPT files to be loaded')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path for storing the converted CKPT files.')
    parser.add_argument('--target-vae-tp-size', type=int, required=True,
                        help='TP size of the vae part to be converted.')
    parser.add_argument('--target-dit-tp-size', type=int, required=True,
                        help='TP size of the dit part to be converted.')
    parser.add_argument('--target-vae-pp-size', type=int, required=True,
                        help='PP size of the vae part to be converted.')
    parser.add_argument('--target-dit-pp-size', type=int, required=True,
                        help='PP size of the dit part to be converted.')
    parser.add_argument('--target-vae-cp-size', type=int, required=True,
                        help='CP size of the vae part to be converted.')
    parser.add_argument('--target-dit-cp-size', type=int, required=True,
                        help='CP size of the dit part to be converted.')
    parser.add_argument('--target-vae-pp-layers', type=str, default='[]',
                        help='PP layers of the vae part to be converted.')
    parser.add_argument('--target-dit-pp-layers', type=str, required=True,
                        help='PP layers of the dit part to be converted.')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")
    args.target_vae_pp_layers = ast.literal_eval(args.target_vae_pp_layers)
    args.target_dit_pp_layers = ast.literal_eval(args.target_dit_pp_layers)
    if max(1, len(args.target_vae_pp_layers)) != args.target_vae_pp_size:  # ckpt file has no vae part, [] -> pp_size=1
        raise ValueError(f'len({args.target_vae_pp_layers}) must equals to {args.target_vae_pp_size=}.')
    if len(args.target_dit_pp_layers) != args.target_dit_pp_size:
        raise ValueError(f'len({args.target_dit_pp_layers}) must equals to {args.target_dit_pp_size=}.')

    main(args)
