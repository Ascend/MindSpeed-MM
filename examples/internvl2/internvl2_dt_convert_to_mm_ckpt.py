import argparse
import ast

import internvl2_mm_convert_to_dt_ckpt as ckpt_utils


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

    path, iteration = ckpt_utils.get_ckpt_path(args.load_dir)
    if iteration is None:
        iteration = 'release'

    # vit part
    tp_size, pp_size = ckpt_utils.get_loaded_ckpt_tp_pp(path, 'vit')
    print(f'Get saved vit ckpts have {tp_size=} {pp_size=} {iteration=}.')
    ckpts_vit, params_vit = ckpt_utils.load_ckpt(path, tp_size, pp_size, 'vit')
    args_vit = getattr(params_vit, 'args', None)
    if args_vit is not None:
        if tp_size != args_vit.tensor_model_parallel_size:
            raise ValueError(f'tp_size ({tp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_vit.tensor_model_parallel_size}).')
        if pp_size != args_vit.pipeline_model_parallel_size:
            raise ValueError(f'pp_size ({pp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_vit.pipeline_model_parallel_size}).')
    print('Vit ckpts loaded.')
    ckpts_vit = ckpt_utils.merge_by_pp(ckpts_vit, tp_size, pp_size, keys_full_prefix_on_pp_layer)
    print('Vit ckpts merged by pp.')
    ckpts_vit = ckpt_utils.merge_by_tp(ckpts_vit, tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Vit ckpts merged by tp.')
    ckpt_utils.print_keys(ckpts_vit)

    # gpt part
    tp_size, pp_size = ckpt_utils.get_loaded_ckpt_tp_pp(path, 'gpt')
    print(f'Get saved gpt ckpts have {tp_size=} {pp_size=} {iteration=}, prepare to loading.')
    ckpts_gpt, params_gpt = ckpt_utils.load_ckpt(path, tp_size, pp_size, 'gpt')
    args_gpt = getattr(params_gpt, 'args', None)
    if args_gpt is not None:
        if tp_size != args_gpt.tensor_model_parallel_size:
            raise ValueError(f'tp_size ({tp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.tensor_model_parallel_size}).')
        if pp_size != args_gpt.pipeline_model_parallel_size:
            raise ValueError(f'pp_size ({pp_size}) cannot match the attributes saved in the CKPT '
                             f'({args_gpt.pipeline_model_parallel_size}).')
    print('Gpt ckpts loaded.')
    ckpts_gpt = ckpt_utils.merge_by_pp(ckpts_gpt, tp_size, pp_size, keys_full_prefix_on_pp_layer)
    print('Gpt ckpts merged by pp.')
    ckpts_gpt = ckpt_utils.merge_by_tp(ckpts_gpt, tp_size, 1, keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('Gpt ckpts merged by tp.')
    ckpt_utils.print_keys(ckpts_gpt)

    # split pp
    ckpts_vit_pp = ckpt_utils.split_by_pp(ckpts_vit, 1, args.target_vit_pp_layers, keys_full_prefix_on_pp_process,
                                          keys_full_prefix_on_pp_postprocess)
    print('Get vit ckpts split by pp.')
    ckpts_gpt_pp = ckpt_utils.split_by_pp(ckpts_gpt, 1, args.target_gpt_pp_layers, keys_full_prefix_on_pp_process,
                                          keys_full_prefix_on_pp_postprocess)
    print('Get gpt ckpts split by pp.')

    # merge
    ckpts = ckpt_utils.merge_to_mm([ckpts_vit_pp, ckpts_gpt_pp], args.target_tp_size, args.target_pp_size)
    print('MM ckpts split by pp.')

    # mm
    ckpts = ckpt_utils.split_by_tp(ckpts, args.target_tp_size, args.target_pp_size,
                                   keys_part_on_tp_dim_0, keys_part_on_tp_dim_1)
    print('MM ckpts split by tp and pp.')
    state_dicts = ckpt_utils.add_extra_params(ckpts, params_vit, args.target_tp_size, args.target_pp_size,
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
    parser.add_argument('--target-vit-pp-layers', type=str, required=True,
                        help='PP layers of the vit part to be converted.')
    parser.add_argument('--target-gpt-pp-layers', type=str, required=True,
                        help='PP layers of the gpt part to be converted.')
    args, unrecognized_args = parser.parse_known_args()
    if unrecognized_args:
        print(f"Unrecognized Args: {unrecognized_args}")
    args.target_vit_pp_layers = ast.literal_eval(args.target_vit_pp_layers)
    args.target_gpt_pp_layers = ast.literal_eval(args.target_gpt_pp_layers)
    if not (len(args.target_vit_pp_layers) == len(args.target_gpt_pp_layers) == args.target_pp_size):
        raise ValueError(f'len({args.target_vit_pp_layers}) and len({args.target_gpt_pp_layers}) '
                         f'must equals to {args.target_pp_size=}')

    main(args)
