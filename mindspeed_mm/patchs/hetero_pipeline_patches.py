# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import contextlib
import inspect
from functools import wraps, partial
from collections import deque
import torch
import torch.distributed as dist
import megatron.core.parallel_state as mpu
from megatron.training import get_args, print_rank_0
from megatron.core import parallel_state
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.utils import (
    get_model_config,
    get_model_type,
    get_attr_wrapped_model,
    get_model_xattn,
)
from megatron.core.pipeline_parallel.schedules import (
    deallocate_output_tensor,
    forward_step,
    backward_step,
    check_first_val_step,
    get_forward_backward_func,
    get_tensor_shapes,
    recv_forward,
    recv_backward,
    send_forward,
    send_backward,
    send_forward_recv_backward,
    send_backward_recv_forward,
    clear_embedding_activation_buffer,
    finish_embedding_wgrad_compute
)
from megatron.core.pipeline_parallel import schedules
from mindspeed_mm.utils.hetero_parallel import change_parallel_state
from mindspeed_mm.utils.hetero_parallel import _HeteroParallelModules as MODULE_LIST
from mindspeed_mm.utils.hetero_parallel import _ParallelStatesDict


class PipelineMeta:
    def __init__(self, module_name=None, state_snapshot=None, is_first_pipeline=False, is_last_pipeline=False):
        self.module_name = module_name
        self.state_snapshot = state_snapshot


class ReplayIterator:
    def __init__(self, data_iterator):
        self.real_iter = data_iterator
        self.cached_batch = None
        self.has_cached = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.has_cached:
            try:
                self.cached_batch = next(self.real_iter)
                self.has_cached = True
                return self.cached_batch
            except StopIteration as e:
                raise RuntimeError(f"Can not next, because of {e}") from e
        else:
            return self.cached_batch

    @property
    def current_batch(self):
        if not self.has_cached:
            raise RuntimeError("Try to get current_batch, but no batch cached.")
        return self.cached_batch

    def consume_and_reset(self):
        self.cached_batch = None
        self.has_cached = False


def recovery_parallel_state(source_globals):
    target_globals = vars(mpu)

    for k, v in source_globals.items():
        if k in target_globals:
            target_globals[k] = v


def store_state_snapshot():
    state_snapshot = {
        k: v
        for k, v in vars((mpu)).items()
        if k.startswith('_') and not k.startswith('__') and not inspect.isfunction(v)
    }
    return state_snapshot


def get_update_data_iterator_func(module_meta_pre, module_meta):
    return '_'.join(['update_data_iterator', 'from', module_meta_pre.module_name, 'to', module_meta.module_name])


def get_backward_func(forward_backward_pipeline):
    return forward_backward_pipeline + '_backward'


def update_data_iterator_from_image_encoder_to_text_decoder(outputs, data_batchs, pp_batch):
    if not outputs:
        raise ValueError('Image encoder outputs is empty.')
    if not data_batchs:
        raise ValueError('GBS data_batchs is empty.')

    if data_batchs:
        keys = [k
                for k in data_batchs[0].keys()
                if k != 'pixel_values' and isinstance(data_batchs[0][k], torch.Tensor)]
    else:
        keys = []
    all_keys = ['vit_embeds'] + [k for k in keys if k != 'vit_embeds']

    split_tensors_queues = deque()
    for key in all_keys:
        if key == 'vit_embeds':
            local_tensors = outputs
        else:
            local_tensors = []
            for i in range(len(outputs)):
                t = data_batchs[i].get(key)
                if isinstance(t, torch.Tensor):
                    local_tensors.append(t)
                else:
                    raise ValueError(f"Key '{key}' missing in micro-batch {i} in in dp rank: {dist.get_rank()}")
        if not local_tensors:
            continue
        split_tensors_queues.append((key, deque(local_tensors)))

    pp_batch['pp_batchs_queues'] = split_tensors_queues


def mpu_wrapper():
    mpu._HETERO_PIPELINE = False
    mpu._IS_LAST_PIPELINE = False
    mpu._IS_FIRST_PIPELINE = False
    return mpu


def mpu_is_pipeline_last_stage_wrapper(original_func):
    @wraps(original_func)
    def wrapper(*args, **kwargs):
        return original_func(*args, **kwargs) and mpu._IS_LAST_PIPELINE

    return wrapper


original_is_pipeline_last_stage = mpu.is_pipeline_last_stage
mpu = mpu_wrapper()
parallel_state = mpu_wrapper()
parallel_state.is_pipeline_last_stage = mpu_is_pipeline_last_stage_wrapper(original_is_pipeline_last_stage)


def get_forward_backward_func_wrapper(original_func):
    @wraps(original_func)
    def wrapper(parallel_states_dict=None, *args, **kwargs):

        if parallel_states_dict is None:
            return original_func(*args, **kwargs)

        pipeline_meta_list = []
        forward_backward_func_list = []
        origin_state_snapshot = store_state_snapshot()
        if isinstance(parallel_states_dict, dict):
            for module in MODULE_LIST:
                # audio_encoder is considered as a part of pp stage0
                if module in parallel_states_dict and module != 'audio_encoder':
                    pipeline_meta_list.append(
                        PipelineMeta(module_name=module, state_snapshot=parallel_states_dict[module])
                    )
                    change_parallel_state(module)
                    forward_backward_func_list.append(original_func(*args, **kwargs))
        recovery_parallel_state(origin_state_snapshot)

        if len(forward_backward_func_list) < 1:
            raise ValueError(
                'get_forward_backward_func_wrapper is Error, please check parallel_states_dict: ', parallel_states_dict
            )
        elif len(forward_backward_func_list) == 1:
            return forward_backward_func_list[0]
        else:
            for meta_info in pipeline_meta_list:
                meta_info.state_snapshot['_HETERO_PIPELINE'] = True
            pipeline_meta_list[0].state_snapshot['_IS_FIRST_PIPELINE'] = True
            pipeline_meta_list[0].state_snapshot['_IS_LAST_PIPELINE'] = False
            pipeline_meta_list[-1].state_snapshot['_IS_FIRST_PIPELINE'] = False
            pipeline_meta_list[-1].state_snapshot['_IS_LAST_PIPELINE'] = True
            return partial(
                hetero_pipeline,
                pipeline_meta_list=pipeline_meta_list,
                forward_backward_func_list=forward_backward_func_list
            )

    return wrapper


def hetero_pipeline(
        pipeline_meta_list,
        forward_backward_func_list,
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    if len(pipeline_meta_list) != len(forward_backward_func_list):
        raise ValueError("module_meta num is not equal num of forward_backward_func in hetero_pipeline")

    backward_func_list, backward_pipeline_meta_list, output_tensors_list, num_microbatches_list = [], [], [], []
    total_num_tokens, forward_data_store = None, None
    module_meta_pre = None
    update_data_iterator_func = None

    for module_meta, forward_backward_func in zip(pipeline_meta_list, forward_backward_func_list):

        if module_meta_pre is not None:
            update_data_iterator_func = get_update_data_iterator_func(module_meta_pre, module_meta)
        if update_data_iterator_func is not None:
            globals()[update_data_iterator_func](output_tensors, current_batchs, data_iterator.current_batch)
        else:
            data_iterator = ReplayIterator(data_iterator)

        change_parallel_state(module_meta.module_name)
        forward_only_for_global = (not mpu._IS_LAST_PIPELINE) or forward_only

        output = forward_backward_func(forward_step_func=forward_step_func, data_iterator=data_iterator,
                                       model=model, num_microbatches=num_microbatches, seq_length=seq_length,
                                       micro_batch_size=micro_batch_size,
                                       decoder_seq_length=decoder_seq_length, forward_only=forward_only_for_global,
                                       collect_non_loss_data=collect_non_loss_data, first_val_step=first_val_step)

        if module_meta_pre:
            data_iterator.consume_and_reset()

        forward_data_store, output_tensors, total_num_tokens, current_batchs = output
        output_tensors_list.append(output_tensors)
        module_meta_pre = module_meta

        if forward_only_for_global and (not forward_only):
            backward_func_list.append(get_backward_func(forward_backward_func.__name__))
            backward_pipeline_meta_list.append(module_meta)
            num_microbatches_list.append(num_microbatches)

    return forward_data_store


def forward_backward_no_pipelining_patch(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,  # unused
        micro_batch_size: int,  # unused
        decoder_seq_length: int = None,  # unused
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    """Run forward and backward passes with no pipeline parallelism
    (no inter-stage communication).

    Returns dictionary with losses.


    See get_forward_backward_func() for argument details
    """
    if isinstance(model, list):
        if len(model) != 1:
            raise ValueError("non-pipeline-parallel schedule does not support model chunking")
        model = model[0]

    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            raise ValueError("non-pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    output_tensors = []
    current_batch = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    with no_sync_func():
        for i in range(num_microbatches - 1):
            output_tensor, num_tokens = forward_step(
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
            )
            total_num_tokens += num_tokens
            if not forward_only:
                backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
            elif mpu._HETERO_PIPELINE and not mpu._IS_LAST_PIPELINE:
                output_tensors.append(output_tensor)
                current_batch.append(data_iterator.current_batch)
                data_iterator.consume_and_reset()

    # Run computation for last microbatch out of context handler (want to
    # synchronize gradients).
    data_iterator = ReplayIterator(data_iterator)
    output_tensor, num_tokens = forward_step(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data,
        is_first_microbatch=check_first_val_step(
            first_val_step, forward_only, num_microbatches == 1
        ),
        current_microbatch=num_microbatches - 1,
    )
    total_num_tokens += num_tokens

    if not forward_only:
        backward_step(input_tensor, output_tensor, output_tensor_grad, model_type, config)
    elif mpu._HETERO_PIPELINE and not mpu._IS_LAST_PIPELINE:
        output_tensors.append(output_tensor)
        current_batch.append(data_iterator.current_batch)

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model], total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store, output_tensors, total_num_tokens, current_batch


def forward_backward_pipelining_without_interleaving_patch(
        *,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches: int,
        seq_length: int,
        micro_batch_size: int,
        decoder_seq_length: int = None,
        forward_only: bool = False,
        collect_non_loss_data: bool = False,
        first_val_step: bool = None,
):
    """
    Run non-interleaved 1F1B schedule, with communication between pipeline stages.
    Returns dictionary with losses if the last stage, empty dict otherwise.
    """

    if isinstance(model, list):
        if len(model) != 1:
            raise ValueError("non-interleaved pipeline-parallel schedule does not support model chunking")
        model = model[0]

    if isinstance(data_iterator, list):
        if len(data_iterator) != 1:
            raise ValueError("non-interleaved pipeline-parallel schedule does not support model chunking")
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "non-interleaved pipeline-parallel schedule does not support communication"
        )

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    def get_unwrapped_model(model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def split_batch(split_tensors_queues, current_batch):
        for key, q in split_tensors_queues:
            if len(q) == 0:
                raise RuntimeError(f"Queue for key '{key}' is empty, cannot pop.")
            current_split_tensor = q.popleft()
            current_batch[key] = current_split_tensor

    def set_decoder_input_tensor(model, input_tensor):
        if not mpu.is_pipeline_first_stage():
            vlm_model = get_unwrapped_model(model)
            decoder_model = vlm_model.text_decoder
            set_input_tensor = get_attr_wrapped_model(decoder_model, "set_input_tensor")
            set_input_tensor(input_tensor[0])

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
            parallel_state.get_pipeline_model_parallel_world_size()
            - parallel_state.get_pipeline_model_parallel_rank()
            - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model)
    encoder_decoder_xattn = get_model_xattn(model)

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    split_tensors_queues = data_iterator.current_batch['pp_batchs_queues']

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                    i % max_outstanding_backprops
                    >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = recv_forward(recv_tensor_shapes, config)
        set_decoder_input_tensor(model, input_tensor)

        split_batch(split_tensors_queues, data_iterator.current_batch)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )

        send_forward(output_tensor, send_tensor_shapes, config)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = recv_forward(recv_tensor_shapes, config)

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                                                        (i + num_warmup_microbatches) % max_outstanding_backprops
                                                ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        set_decoder_input_tensor(model, input_tensor)
        split_batch(split_tensors_queues, data_iterator.current_batch)
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            collect_non_loss_data,
            checkpoint_activations_microbatch,
            check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            encoder_decoder_xattn=encoder_decoder_xattn,
        )
        total_num_tokens += num_tokens

        if forward_only:
            send_forward(output_tensor, send_tensor_shapes, config)

            if not last_iteration:
                input_tensor = recv_forward(recv_tensor_shapes, config)

        else:
            output_tensor_grad = send_forward_recv_backward(
                output_tensor, send_tensor_shapes, config
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            if last_iteration:
                input_tensor = None
                send_backward(input_tensor_grad, recv_tensor_shapes, config)
            else:
                input_tensor = send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, config
                )

    # Run cooldown backward passes.
    input_tensor_grads = []
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = recv_backward(send_tensor_shapes, config)

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward(input_tensor_grad, recv_tensor_shapes, config)

            if rank == 0 and mpu._HETERO_PIPELINE:
                input_tensor_grads.append(output_tensor_grad)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        if (not hasattr(parallel_state, "_IS_FIRST_PIPELINE") or (
        getattr(parallel_state, "_IS_FIRST_PIPELINE", False))):
            config.finalize_model_grads_func(
                [model], total_num_tokens if config.calculate_per_token_loss else None
            )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    current_batch = []

    return forward_data_store, input_tensor_grads, total_num_tokens, current_batch

print_rank_0("hetero pipeline patches is activated...")
hp_get_forward_backward_func = partial(
    get_forward_backward_func_wrapper(get_forward_backward_func),
    parallel_states_dict=_ParallelStatesDict
)
schedules.forward_backward_pipelining_without_interleaving = forward_backward_pipelining_without_interleaving_patch
schedules.forward_backward_no_pipelining = forward_backward_no_pipelining_patch
