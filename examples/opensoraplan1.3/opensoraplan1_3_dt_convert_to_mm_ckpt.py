import argparse
import ast

import opensoraplan1_3_mm_convert_to_dt_ckpt as ckpt_utils


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
    path, iteration = ckpt_utils.get_ckpt_path(args.load_dir)
    if iteration is None:
        iteration = 'release'

    # dit part is true ckpt part
    tp_size, pp_size = ckpt_utils.get_loaded_ckpt_tp_pp(path, 'dit')
    print(f'Get saved (dit) ckpts have {tp_size=} {pp_size=} {iteration=}, prepare to loading.')
    ckpts, params_dit = ckpt_utils.load_ckpt(path, tp_size, pp_size, 'dit')
    args_gpt = getattr(params_dit, 'args', None)
    if args_gpt is not None:
        if tp_size != args_gpt.tensor_model_parallel_size:
            raise ValueError(f'tp_size ({tp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.tensor_model_parallel_size}).')
        if pp_size != args_gpt.pipeline_model_parallel_size:
            raise ValueError(f'pp_size ({pp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.pipeline_model_parallel_size}).')
    print('MM ckpts loaded.')
    ckpts = ckpt_utils.merge_by_pp(ckpts, tp_size, pp_size, keys_full_prefix_on_pp_layer)
    print('MM ckpts merged by pp.')
    ckpts = ckpt_utils.merge_by_tp(ckpts, tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('MM ckpts merged by tp.')
    ckpt_utils.print_keys(ckpts)

    # mm
    ckpts_dit_pp = ckpt_utils.split_by_pp(ckpts, 1, args.target_dit_pp_layers, keys_full_prefix_on_pp_process,
                                          keys_full_prefix_on_pp_postprocess)
    print('MM ckpts split by pp.')
    ckpts = ckpt_utils.split_by_tp(ckpts_dit_pp, args.target_tp_size, args.target_pp_size, keys_part_on_tp_dim_0,
                                   keys_part_on_tp_dim_1)
    print('MM ckpts split by tp and pp.')
    state_dicts = ckpt_utils.add_extra_params(ckpts, params_dit, args.target_tp_size, args.target_pp_size,
                                              args.target_cp_size)
    ckpt_utils.save_by_pp_tp(args.save_dir, state_dicts, args.target_pp_size, None, iteration)
    print('MM ckpts saved.')
    ckpt_utils.print_keys(ckpts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DistTrain Checkpoint Utility Arguments',
                                     allow_abbrev=False,
                                     conflict_handler='resolve')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Path for storing the CKPT files to be loaded.')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Path for storing the converted CKPT files.')
    parser.add_argument('--target-tp-size', type=int, required=True,
                        help='TP size of the CKPT to be converted.')
    parser.add_argument('--target-pp-size', type=int, required=True,
                        help='PP size of the CKPT to be converted.')
    parser.add_argument('--target-cp-size', type=int, required=True,
                        help='CP size of the CKPT to be converted.')
    parser.add_argument('--target-dit-pp-layers', type=str, required=True,
                        help='PP layers of the dit part to be converted.')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")
    args.target_dit_pp_layers = ast.literal_eval(args.target_dit_pp_layers)
    if len(args.target_dit_pp_layers) != args.target_pp_size:
        raise ValueError(f'len({args.target_dit_pp_layers}) must equals to {args.target_pp_size=}')

    main(args)
