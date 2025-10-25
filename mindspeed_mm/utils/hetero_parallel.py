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
from mindspeed_mm.models.common.communications import gather_forward_split_backward


_ParallelStatesDict = {}
_HeteroParallelModules = ['image_encoder', 'audio_encoder', 'text_decoder']


def apply_hetero_parallel_hooks(model):

    if hasattr(model, 'image_encoder'):
        model.image_encoder.register_forward_pre_hook(image_encoder_forward_pre_hook)
        model.image_encoder.register_forward_hook(image_encoder_forward_hook)
        model.image_encoder.register_full_backward_pre_hook(audio_encoder_backward_hook)

    if hasattr(model, 'audio_encoder'):
        model.audio_encoder.register_forward_pre_hook(audio_encoder_forward_pre_hook)
        model.audio_encoder.register_forward_hook(audio_encoder_forward_hook)
        model.audio_encoder.register_full_backward_pre_hook(audio_encoder_backward_hook)


def image_encoder_forward_pre_hook(module, input):
    pixel_values, image_grid_thw = input
    change_parallel_state('image_encoder')
    chunk_seq_lens = []
    for chunk in torch.chunk(image_grid_thw, chunks=mpu.get_data_parallel_world_size(), dim=0):
        chunk_seq_lens.append(chunk.prod(dim=1).sum())
    chunk_seq_lens = torch.stack(chunk_seq_lens).tolist()

    pixel_values = DataStreamReComm.hetero_comm(
                    pixel_values,
                    'text_decoder',
                    'image_encoder',
                    cat_split_dim=0,
                    pad_dim=0,
                    all_lens=chunk_seq_lens
    )

    image_grid_thw = DataStreamReComm.hetero_comm(
                    image_grid_thw,
                    'text_decoder',
                    'image_encoder',
                    cat_split_dim=0,
                    pad_dim=1
    )
    change_parallel_state('image_encoder')
    return pixel_values, image_grid_thw


def image_encoder_forward_hook(module, input, output):    
    output = DataStreamReComm.hetero_comm(
                    output,
                    'image_encoder',
                    'text_decoder',
                    cat_split_dim=0,
                    pad_dim=0
    )
    change_parallel_state('text_decoder')
    return output


def audio_encoder_forward_pre_hook(module, input):
    input_features, feature_attention_mask = input

    input_features = DataStreamReComm.hetero_comm(
                    input_features,
                    'text_decoder',
                    'audio_encoder',
                    cat_split_dim=0,
                    pad_dim=1
    )
    feature_attention_mask = DataStreamReComm.hetero_comm(
                    feature_attention_mask,
                    'text_decoder',
                    'audio_encoder',
                    cat_split_dim=0,
                    pad_dim=1
    )
    change_parallel_state('audio_encoder')
    return input_features, feature_attention_mask


def audio_encoder_forward_hook(module, input, output):
    output = DataStreamReComm.hetero_comm(
                    output,
                    'audio_encoder',
                    'text_decoder',
                    cat_split_dim=0,
                    pad_dim=0
    )
    change_parallel_state('text_decoder')
    return output


def audio_encoder_backward_hook(module, input):
    change_parallel_state('audio_encoder')
    return input


def image_encoder_backward_hook(module, input):
    change_parallel_state('image_encoder')
    return input


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
    

def hetero_align_config(config_inner, config_outer):
    config_inner.pipeline_model_parallel_size = config_outer.pp
    config_inner.context_parallel_size = config_outer.cp
    config_inner.tensor_model_parallel_size = config_outer.tp


class DataStreamReComm:
    _is_initialized = False
    _COMM_DP_WORLD_SIZE = {}
    _COMM_SUBGROUP_CACHE = {}

    @classmethod
    def _init(cls):
        if cls._is_initialized:
            return 
        
        for module in _HeteroParallelModules:
            change_parallel_state(module)
            cls._COMM_DP_WORLD_SIZE[module] = mpu.get_data_parallel_world_size()
        cls._is_initialized = True

        if cls._COMM_DP_WORLD_SIZE['text_decoder'] > cls._COMM_DP_WORLD_SIZE['audio_encoder'] \
            or cls._COMM_DP_WORLD_SIZE['text_decoder'] > cls._COMM_DP_WORLD_SIZE['image_encoder']:
            raise AssertionError("text decoder's DP world size should be no more than DP world size of encoders")
    
    @classmethod
    def compute_comm_subgroup(cls, from_module, to_module):
        if (from_module, to_module) in cls._COMM_SUBGROUP_CACHE:
            return cls._COMM_SUBGROUP_CACHE[(from_module, to_module)]
        large_DP_world_size = cls._COMM_DP_WORLD_SIZE[from_module] if cls._COMM_DP_WORLD_SIZE[from_module] > \
                cls._COMM_DP_WORLD_SIZE[to_module] else cls._COMM_DP_WORLD_SIZE[to_module]
        small_DP_world_size = cls._COMM_DP_WORLD_SIZE[to_module] if cls._COMM_DP_WORLD_SIZE[from_module] > \
                cls._COMM_DP_WORLD_SIZE[to_module] else cls._COMM_DP_WORLD_SIZE[from_module]
        
        world_size = dist.get_world_size()
        stride = world_size // large_DP_world_size
        group_len = large_DP_world_size // small_DP_world_size
        rank = dist.get_rank()

        non_dp_group_ranks_list = [list(range(i * (world_size // small_DP_world_size), (i + 1) * (world_size // small_DP_world_size)))
                                   for i in range(small_DP_world_size)]
        
        rank_indexes = [[stride * j + i for j in range(group_len)]for i in range(stride)]
        ranks_list = []
        for non_dp_group_ranks in non_dp_group_ranks_list:
            for rank_index in rank_indexes:
                ranks_list.append([non_dp_group_ranks[i] for i in rank_index])
        
        for ranks in ranks_list:
            group = mpu.create_group(
                ranks, 
                group_desc='NON_DP_GROUP',
            )
            if rank in ranks:
                subgroup = group
        cls._COMM_SUBGROUP_CACHE[(from_module, to_module)] = subgroup

        return subgroup
    
    @classmethod
    def hetero_comm(cls,
                    tensor,
                    from_module,
                    to_module,
                    cat_split_dim=0,
                    pad_dim=1,
                    **kwargs):
        cls._init()

        from_dp_size = cls._COMM_DP_WORLD_SIZE[from_module]
        to_dp_size = cls._COMM_DP_WORLD_SIZE[to_module]

        if from_dp_size == to_dp_size:
            return tensor
        
        subgroup_comm = cls.compute_comm_subgroup(from_module, to_module)

        if from_dp_size < to_dp_size: # dp_1 < dp_2
            return cls._hetero_split(
                        tensor,
                        subgroup_comm,
                        split_dim=cat_split_dim,
                        pad_dim=pad_dim,
                        all_lens=kwargs.get('all_lens', None)
                    )
        else:
            return cls._hetero_all_gather(
                        tensor,
                        subgroup_comm,
                        cat_dim=cat_split_dim,
                        pad_dim=pad_dim,
                        pad_token_id=kwargs.get('pad_token_id', None)
                    )
        
    @classmethod
    def _hetero_split(cls,
                      tensor,
                      subgroup_comm,
                      split_dim=0,
                      pad_dim=1,
                      all_lens=None):
        world_size = dist.get_world_size(subgroup_comm)
        rank = dist.get_rank(subgroup_comm)

        if tensor.requires_grad:
            raise AssertionError("split with tensor requiring grad is not implemented")
        
        if all_lens is None:
            chunks = torch.chunk(tensor, world_size, dim=split_dim)
            chunk = chunks[rank]
        else:
            step = len(all_lens) // world_size
            all_lens = [all_lens[i: i + step] for i in range(0, len(all_lens), step)]

            if split_dim == pad_dim:
                local_lens = [sum(all_len) for all_len in all_lens]
                chunks = torch.split(tensor, dim=split_dim, split_size_or_sections=local_lens)
                chunk = chunks[rank]
            else:
                chunks = torch.chunk(tensor, world_size, dim=split_dim)
                chunk = chunks[rank]
                local_lens = all_lens[rank]

                index = [slice(None)] * chunk.ndim
                index[pad_dim] = slice(0, max(local_lens))
                chunk = chunk[tuple(index)]
        return chunk
    
    @classmethod
    def _hetero_all_gather(cls,
                           tensor,
                           subgroup_comm,
                           cat_dim=0,
                           pad_dim=1,
                           pad_token_id=None):
        world_size = dist.get_world_size(subgroup_comm)
        pad_token_id = 0 if pad_token_id is None else pad_token_id
        local_len = torch.tensor([tensor.shape[pad_dim]], device='cuda')
        all_lens = [torch.empty_like(local_len) for _ in range(world_size)]

        dist.all_gather(all_lens, local_len, group=subgroup_comm)
        all_lens = [length.item() for length in all_lens]
        
        output = gather_forward_split_backward(input_=tensor,
                                               process_group=subgroup_comm,
                                               dim=pad_dim,
                                               gather_sizes=all_lens)
        return output