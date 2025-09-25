import inspect
from functools import wraps
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from mindspeed.megatron_adaptor import get_mindspeed_args
from mindspeed.patch_utils import MindSpeedPatchesManager as mspm
from megatron.training import get_args, print_rank_0
from megatron.core.parallel_state import initialize_model_parallel, is_initialized
import megatron.core.parallel_state as mpu
from mindspeed_mm import training


_ParallelStatesDict = {}
_HeteroParallelModules = ['image_encoder', 'audio_encoder', 'text_decoder']


def apply_hetero_parallel_hooks(model):

    if hasattr(model, 'image_encoder'):
        model.image_encoder.register_forward_pre_hook(image_encoder_forward_pre_hook)
        model.image_encoder.register_forward_hook(image_encoder_forward_hook)
    if hasattr(model, 'audio_encoder'):
        model.audio_encoder.register_forward_pre_hook(audio_encoder_forward_pre_hook)
        model.audio_encoder.register_forward_hook(audio_encoder_forward_hook)


def image_encoder_forward_pre_hook(module, input):
    pixel_values, image_grid_thw = input
            
    change_parallel_state('text_decoder')
    pixel_values, _ = all_gather_dp_group(pixel_values, pad_dim=0, remove_padding=True)
    image_grid_thw, _ = all_gather_dp_group(image_grid_thw)
    change_parallel_state('image_encoder')

    chunk_seq_lens = []
    for chunk in torch.chunk(image_grid_thw, chunks=mpu.get_data_parallel_world_size(), dim=0):
        chunk_seq_lens.append(chunk.prod(dim=1).sum())
    chunk_seq_lens = torch.stack(chunk_seq_lens).tolist()

    pixel_values = split_tensor_dp_group(pixel_values, pad_dim=0, chunk_seq_lens=chunk_seq_lens)  # [B, S]
    image_grid_thw = split_tensor_dp_group(image_grid_thw, split_dim=0)

    return pixel_values, image_grid_thw


def image_encoder_forward_hook(module, input, output):    
    output, all_lens = all_gather_dp_group(output, cat_dim=0, pad_dim=0, remove_padding=True)

    change_parallel_state('text_decoder')

    chunk_seq_lens = []
    origin_len = len(all_lens)
    for i in range(0, origin_len, origin_len // mpu.get_data_parallel_world_size()):
        length = sum(all_lens[i: i + origin_len // mpu.get_data_parallel_world_size()])
        chunk_seq_lens.append(length)

    output = split_tensor_dp_group(output, pad_dim=0, split_dim=0, chunk_seq_lens=chunk_seq_lens)


    return output


def audio_encoder_forward_pre_hook(module, input):
    input_features, feature_attention_mask = input
    change_parallel_state('text_decoder')
    input_features, _ = all_gather_dp_group(input_features)
    feature_attention_mask, _ = all_gather_dp_group(feature_attention_mask)
    change_parallel_state('audio_encoder')
    input_features = split_tensor_dp_group(input_features)
    feature_attention_mask = split_tensor_dp_group(feature_attention_mask)

    return input_features, feature_attention_mask


def audio_encoder_forward_hook(module, input, output):
    output, all_lens = all_gather_dp_group(output, pad_token_id=0.0, cat_dim=0, pad_dim=0, remove_padding=True)
    change_parallel_state('text_decoder')

    chunk_seq_lens = []
    origin_len = len(all_lens)
    for i in range(0, origin_len, origin_len // mpu.get_data_parallel_world_size()):
        length = sum(all_lens[i: i + origin_len // mpu.get_data_parallel_world_size()])
        chunk_seq_lens.append(length)

    output = split_tensor_dp_group(output, pad_dim=0, split_dim=0, chunk_seq_lens=chunk_seq_lens)
    
    return output


def parallel_config_extract(args_dict):
    targets = ["tp", "cp", "pp"]
    results = []

    def dfs(curr, par_key=None):
        if isinstance(curr, dict):
            if all(k in curr for k in targets) and par_key:
                results.append({
                    par_key: {k: curr[k] for k in targets}
                })
            for k, v in curr.items():
                dfs(v, k)
        elif isinstance(curr, list):
            for item in curr:
                dfs(item, par_key)

    dfs(args_dict)
    return results


def initial_modules_mpu(reuse_module, args):
    args_dict = args.to_dict()
    
    if is_initialized:
        _ParallelStatesDict[reuse_module] = {}
        state_snapshot = {
            k: v for k, v in vars((mpu)).items()
            if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
        }
        _ParallelStatesDict[reuse_module].update(state_snapshot)
    
    parallel_configs = parallel_config_extract(args_dict)
    for parallel_config in parallel_configs:
        module = next(iter(parallel_config))
        TP = parallel_config[module]["tp"]
        CP = parallel_config[module]["cp"]
        PP = parallel_config[module]["pp"]

        if module not in _ParallelStatesDict:
            _ParallelStatesDict[module] = {}
            mpu.destroy_model_parallel()
            initialize_model_parallel(
                tensor_model_parallel_size=TP,
                pipeline_model_parallel_size=PP,
                virtual_pipeline_model_parallel_size=None,
                pipeline_model_parallel_split_rank=None,
                use_sharp=False,
                context_parallel_size=CP,
                expert_model_parallel_size=1,
                nccl_communicator_config_path=None,
                distributed_timeout_minutes=30,
                order="tp-cp-ep-dp-pp")

        state_snapshot = {
            k: v for k, v in vars((mpu)).items()
            if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
        }
        _ParallelStatesDict[module].update(state_snapshot)


def change_parallel_state(module):
    target_globals = vars(mpu)
    source_globals = _ParallelStatesDict[module]

    for k, v in source_globals.items():
        if k in target_globals:
            target_globals[k] = v


def initial_megatron_hetero_parallel_wrapper(fn):
    print_rank_0('initial_megatron_hetero_parallel_wrapper activated')

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        args = get_args()
        vlm_config = deepcopy(args.mm.model)
        from pretrain_vlm import _configure_modules
        _configure_modules(vlm_config, _HeteroParallelModules)
        initial_modules_mpu(reuse_module='text_decoder', args=vlm_config)
        return 
    return wrapper


if hasattr(get_mindspeed_args(), 'hetero_parallel') and get_mindspeed_args().hetero_parallel:
    mspm.register_patch('mindspeed_mm.training.initialize_megatron',
                        initial_megatron_hetero_parallel_wrapper, force_patch=True)
    mspm.apply_patches()


def all_gather_dp_group(tensor, 
                        pad_token_id=None, 
                        cat_dim=0, 
                        pad_dim=1, 
                        remove_padding=False,
                        parallel_state=None,
                        ):
    """Gather tensors 
        暂时只支持BSH、BD
    """

    if parallel_state is None:
        group = mpu.get_data_parallel_group()
        world_size = mpu.get_data_parallel_world_size()
    else:
        group = parallel_state['_DATA_PARALLEL_GROUP']
        world_size = torch.distributed.get_world_size(group=group)
    if tensor is None:
        return None, None
    
    if pad_token_id is not None or remove_padding:
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        local_len = torch.tensor([tensor.shape[pad_dim]], device='cuda')
        all_lens = [torch.zeros_like(local_len) for _ in range(world_size)]

        dist.all_gather(all_lens, local_len, group=group)
        all_lens = [length.item() for length in all_lens]
        max_len = max(all_lens)

        pad_size = max_len - local_len
        if pad_size > 0:
            pad_dims = [0] * (2 * tensor.dim())
            # pad_dims: [B, S, H], [D_left, D_right, S_left, S_right, H_left, H_right]
            pad_dims[2 * (tensor.dim() - pad_dim) - 1] = pad_size  
            tensor = F.pad(tensor, pad_dims, value=pad_token_id)

    if tensor.requires_grad:
        if remove_padding:
            raise NotImplementedError('tensors that require grad and need removing padding are not implemented') 
        output = _AllGatherDp.apply(tensor, cat_dim)
    else:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=group)

        if remove_padding:
            gathered = [g[:length] for g, length in zip(gathered, all_lens)]
        output = torch.cat(gathered, dim=cat_dim).contiguous()

    if remove_padding:
        return output, all_lens
    return output, None


def split_tensor_dp_group(tensor, 
                          split_dim=0, 
                          pad_dim=1,
                          chunk_seq_lens=None,
                          all_lens=None,
                          parallel_state=None):
    """split tensors 
        暂时只支持bsh
        chunk_seq_lens: split tensor sliding chunk_seq_lens
        all_lens: all tensor origin lens(cat_dim)
                  if all_lens is None, split tensor per device equal or not remove padding,
                  if all_lens is not None, remove padding intra-dp, do not remove padding inter-dp
    """
    
    if parallel_state is None:
        world_size = mpu.get_data_parallel_world_size()
        group = mpu.get_data_parallel_group()
    else:
        group = parallel_state['_DATA_PARALLEL_GROUP']
        world_size = torch.distributed.get_world_size(group=group)
    
    if tensor is None:
        return None

    rank = torch.distributed.get_rank(group)

    if chunk_seq_lens:
        chunk = torch.split(tensor, dim=split_dim, split_size_or_sections=chunk_seq_lens)[rank]
    else:
        chunks = torch.chunk(tensor, world_size, dim=split_dim)
        chunk = chunks[rank]
        if all_lens is not None:
            # for not equal split, need remove padding
            local_lens_num = len(all_lens) // world_size
            start_idx = rank * local_lens_num
            end_idx = start_idx + local_lens_num
            local_lens = all_lens[start_idx: end_idx]
            index = [slice(None)] * chunk.ndim
            index[pad_dim] = slice(0, max(local_lens))  # for inner-mbs, not remove padding
            chunk = chunk[tuple(index)]
    return chunk


class _AllGatherDp(torch.autograd.Function):
    """
    all gahter for dp for diff cat dim and padding dim
    """
    @staticmethod
    def forward(ctx, _input, cat_dim=0):
        group = mpu.get_data_parallel_group()
        world_size = mpu.get_data_parallel_world_size()
        group_rank = torch.distributed.get_rank(group)
        ctx.world_size = world_size
        ctx.group = group
        ctx.group_rank = group_rank
        ctx.cat_dim = cat_dim
        ctx.original_batch_size = _input.shape[cat_dim]


        gathered = [torch.zeros_like(_input) for _ in range(world_size)]
        dist.all_gather(gathered, _input, group=group)
        output = torch.cat(gathered, dim=cat_dim).contiguous()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size, group, group_rank, cat_dim, original_batch_size \
            = ctx.world_size, ctx.group, ctx.group_rank, ctx.cat_dim, ctx.original_batch_size, \

        start = group_rank * original_batch_size
        end = start + original_batch_size

        idx = [slice(None)] * grad_output.dim()
        idx[cat_dim] = slice(start, end)
        grad_input = grad_output[tuple(idx)]

        return grad_input, None
    

def hetero_align_config(config_inner, config_outer):
    config_inner.pipeline_model_parallel_size = config_outer.pp
    config_inner.context_parallel_size = config_outer.cp
    config_inner.tensor_model_parallel_size = config_outer.tp
