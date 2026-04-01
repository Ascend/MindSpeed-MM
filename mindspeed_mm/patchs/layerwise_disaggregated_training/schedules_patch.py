# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
import contextlib
from datetime import timedelta
from functools import partial
from typing import Callable, List, Optional, Iterator, Union, Dict, Any

import torch

from megatron.core import mpu, parallel_state
from megatron.core.enums import ModelType
from megatron.core.parallel_state import (
    RankGenerator,
    create_group,
    default_embedding_ranks,
    default_position_embedding_ranks,
    get_nccl_options,
    get_pipeline_model_parallel_group,
)
from megatron.core.pipeline_parallel.schedules import (
    backward_step,
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_backward_no_pipelining,
    get_tensor_shapes,
    set_current_microbatch,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    get_model_xattn,
)
from megatron.training import get_args
from megatron.training.utils import average_losses_across_data_parallel_group

from mindspeed_mm.patchs.layerwise_disaggregated_training import p2p_communication_patch
from mindspeed_mm.utils.utils import compute_token_level_loss

_PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = None
_PIPELINE_GLOBAL_RANKS_NEW_STREAM = None
_PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = None
_PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = None
_PIPELINE_MODEL_PARALLEL_DECODER_START = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None
stream_ping = None
stream_pang = None
stream_last_to_first = None
stream_first_to_last = None
default_stream = None


def move_to_device(batch: Dict[str, Any], float_dtype: str):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            dtype = float_dtype if torch.is_floating_point(v) else None
            batch[k] = v.to(device=torch.cuda.current_device(), dtype=dtype)
        elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
            batch[k] = [t.to(device=torch.cuda.current_device(),
                             dtype=float_dtype if torch.is_floating_point(t) else None)
                        for t in v]


def get_batch(data_iterator, is_vit_last_stage=False):
    """Generate a batch."""
    if data_iterator is not None:
        batch = next(data_iterator)
    else:
        raise ValueError("Data iterator is None. Unable to retrieve batch.")
    move_to_device(batch, get_args().params_dtype)
    has_video = 'pixel_values_videos' in batch and 'video_grid_thw' in batch
    if has_video:
        batch['pixel_values'] = batch.pop('pixel_values_videos')
        batch['image_grid_thw'] = batch.pop('video_grid_thw')
    if (mpu.is_pipeline_first_stage() or is_vit_last_stage) and get_args().encoder_dp_balance:
        batch['pixel_values'], batch['tranfer'] = EncoderBalanceComm.apply(
            batch['pixel_values'],
            mpu.get_data_parallel_group())
    else:
        batch['tranfer'] = None
    return batch


def get_tps(output_tensor):
    """Get the tokens per sample"""
    B, S, _ = output_tensor.shape
    dp_size = torch.distributed.get_world_size(group=mpu.get_data_parallel_group())
    cp_size = torch.distributed.get_world_size(group=mpu.get_context_parallel_group())
    tokens_per_sample = torch.tensor(S, device=output_tensor.device) / dp_size * cp_size
    torch.distributed.all_reduce(tokens_per_sample, group=mpu.get_data_parallel_group())
    return tokens_per_sample


def loss_func(output_tensor):
    """Loss function."""
    args = get_args()
    loss_dict = output_tensor['loss_dict']

    loss_dir = {}
    if args.log_tps:
        tokens_per_sample = get_tps(output_tensor['logits'])
        loss_dir["tokens per sample"] = tokens_per_sample

    if args.calculate_per_token_loss:
        loss, local_num_tokens, reporting_loss = compute_token_level_loss(loss_dict)
        loss_dir["loss"] = (reporting_loss[0], reporting_loss[1])
        return (
            loss[0].clone(),
            local_num_tokens,
            loss_dir
        )

    loss = loss_dict['loss']
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dir["loss"] = averaged_loss[0]
    loss = loss.unsqueeze(0).clone()
    return loss / mpu.get_context_parallel_world_size(), loss_dir


def get_forward_backward_func():
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
    if pipeline_model_parallel_size > 1:
        forward_backward_func = forward_backward_pipelining_without_interleaving
        group_initialize(
            tensor_model_parallel_size=parallel_state.get_tensor_model_parallel_world_size(),
            pipeline_model_parallel_size=parallel_state.get_pipeline_model_parallel_world_size(),
            virtual_pipeline_model_parallel_size=parallel_state.get_virtual_pipeline_model_parallel_world_size(),
            context_parallel_size=parallel_state.get_expert_model_parallel_world_size(),
            expert_tensor_parallel_size=parallel_state.get_expert_tensor_parallel_world_size(),
        )
    else:
        forward_backward_func = forward_backward_no_pipelining

    return forward_backward_func


def group_initialize(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: int = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[
        Callable[[List[int], Optional[int]], List[int]]
    ] = None,
    get_position_embedding_ranks: Optional[
        Callable[[List[int], Optional[int]], List[int]]
    ] = None,
) -> None:
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if (
        encoder_tensor_model_parallel_size == 0
        and encoder_pipeline_model_parallel_size > 0
    ):
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(
            default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks,
            split_rank=pipeline_model_parallel_split_rank,
        )

    if encoder_pipeline_model_parallel_size > 0:
        global _PIPELINE_MODEL_PARALLEL_DECODER_START
        _PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    world_size: int = torch.distributed.get_world_size()

    if encoder_tensor_model_parallel_size > 0:
        if not (
            encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ):
            raise RuntimeError("We do not support encoders with more TP than the decoder.")

    encoder_model_size = (
        encoder_tensor_model_parallel_size
        * encoder_pipeline_model_parallel_size
        * context_parallel_size
    )
    decoder_model_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by {total_model_size}"
        )

    data_parallel_size: int = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    if not (
        encoder_world_size + decoder_world_size == world_size
    ):
        raise RuntimeError(f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}")

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size
        )

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            ) from e

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size
        * expert_model_parallel_size
        * pipeline_model_parallel_size
    )
    expert_data_parallel_size = (
        decoder_world_size // expert_tensor_model_pipeline_parallel_size
    )
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    if not (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ):
        raise RuntimeError("When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same")

    if not (decoder_rank_generator.get_ranks(
        "pp"
    ) == expert_decoder_rank_generator.get_ranks(
        "pp"
    )):
        raise RuntimeError(f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}")

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp.

        Let's say we have a total of 6 GPUs denoted by g0 ... g5.
        For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
        g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
        The present function will create with "tp-dp-pp":
        3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
        4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
        4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
        """
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == "pp":
            # Map one encoder tp rank to several decoder tp ranks, because
            # encoder tp and decoder tp won't be the same size.
            # Assign this way to avoid getting the DP ranks mixed up with the PP ranks.
            # For example, if e_ranks = [0,1,2] and d_ranks = [3,4,5,6]
            # Should yield [0,3], [0,4], [1,5], [2,6]
            rep = len(d_ranks) // len(e_ranks)
            remain = len(d_ranks) % len(e_ranks)
            e_ind = 0
            e_rep = rep + int(e_ind < remain)
            for i, y in enumerate(d_ranks):
                x = e_ranks[e_ind]
                e_rep -= 1
                if e_rep == 0:
                    e_ind += 1
                    e_rep = rep + int(e_ind < remain)
                yield x + y
        elif group_type == "tp-pp":
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            if len(e_ranks) != len(d_ranks):
                raise RuntimeError("Length of encoder ranks and decoder ranks must be the same for tp-pp group")
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # global variables for communication stream
    global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST
    global _PIPELINE_GLOBAL_RANKS_NEW_STREAM
    global _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST
    global _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST

    if pipeline_model_parallel_comm_backend == "ucc":
        # The UCC backend provides two key benefits:
        # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
        # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
        #    with overlapping compute kernels.

        # The UCC backend is recommended in the following cases:
        # 1) When the exposed pipeline-parallel (PP) communications are significant.
        #    - E.g., Pipeline parallelism with very less gradient accumulation steps.
        #    - It may provide better performance due to improved bandwidth utilization.
        # 2) When the critical-path pipeline stage has substantial PP-communication overlap.
        #    - E.g., Uneven pipeline parallelism.
        #    - It may provide better performance due to zero SM resource usage.
        if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
            # UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS variable to be larger than 1,
            # to gurantee the overlapped UCC communications. If this environment variable is set to 1,
            # all the UCC communication will be serialized.
            if os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1":
                raise RuntimeError("UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1")

        # Setting up required environment variables for ucc backend
        #
        # "TORCH_UCC_BLOCKING_WAIT=none" allows non-blocking waits of the communiction handle
        # "UCC_EC_CUDA_STREAM_TASK_MODE" controls how CUDA execution engines (EC)
        # schedule tasks on CUDA streams.
        # "UCX_TLS" controls transport layer selection
        # "NSYS_UCP_COMM_PARAMS=1" enables capturing ucx tracing in nsys profiling
        # "UCX_RNDV_THRESH" controls threshold threshold for switching between
        # eager and rendezvous (RNDV) communication protocols.
        # "UCX_NET_DEVICES" select which network interfaces UCX should use.
        # "UCC_CL_BASIC_TLS" controls which Transport Layers are used by
        # the Basic Collective libraray

        os.environ["TORCH_UCC_BLOCKING_WAIT"] = (
            os.environ["TORCH_UCC_BLOCKING_WAIT"]
            if "TORCH_UCC_BLOCKING_WAIT" in os.environ
            else "none"
        )
        os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"] = (
            os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"]
            if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
            else "driver"
        )
        os.environ["UCX_TLS"] = (
            os.environ["UCX_TLS"] if "UCX_TLS" in os.environ else "ib,cuda_copy"
        )  # cuda_ipc (i.e., NVLink-enablement) will be later supported
        os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
        os.environ["UCX_RNDV_THRESH"] = "0"
        os.environ["UCX_NET_DEVICES"] = "all"
        os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

    for ranks in generator_wrapper("pp"):
        # create pg for different communication streams
        group_new = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_new
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_NEW_STREAM[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE.append(group_new)
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE,
                    group_new,
                ]
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM = [
                    _PIPELINE_GLOBAL_RANKS_NEW_STREAM,
                    ranks,
                ]

        group_last_to_first = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_last_to_first
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST.append(
                    group_last_to_first
                )
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST,
                    group_last_to_first,
                ]
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = [
                    _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST,
                    ranks,
                ]

        group_first_to_last = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_first_to_last
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST.append(
                    group_first_to_last
                )
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST,
                    group_first_to_last,
                ]
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = [
                    _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST,
                    ranks,
                ]


def get_pipeline_model_parallel_group_alternate():
    """Get the alternate pipeline model parallel communication group.

    This function returns the alternate pipeline model parallel group used for
    double-buffering communication in pipeline parallel training. It works in
    conjunction with the default pipeline model parallel group to enable
    efficient alternating communication streams.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The alternate pipeline model parallel communication group(s).
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used in double-buffering communication to improve performance
        - It is typically used alongside the default pipeline model parallel group
        - The two groups are alternated based on the pipeline parallel rank parity
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE


def get_pipeline_model_parallel_group_last_to_first():
    """Get the pipeline model parallel communication group for last-to-first direction.

    This function returns the pipeline model parallel group used for communication
    in the last-to-first direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for last-to-first direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from last stage to first stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST


def get_pipeline_model_parallel_group_first_to_last():
    """Get the pipeline model parallel communication group for first-to-last direction.

    This function returns the pipeline model parallel group used for communication
    in the first-to-last direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for first-to-last direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from first stage to last stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST


def forward_step_impl(data_iterator, model, batch=None):
    """Forward step."""
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    
    if batch is None:
        output_tensor = model(**get_batch(data_iterator, is_vit_last_stage))
    elif parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        output_tensor = model(**batch)
    else:
        output_tensor = model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            image_grid_thw=batch['image_grid_thw']
        )
    return output_tensor, loss_func


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    is_end_stage=False,
    batch=None,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_function = forward_step_func(data_iterator, model, batch)
        else:
            output_tensor, loss_function = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    # U-shaped split scenario, the first and last layers deploy on pp first stage,
    normal_last_stage = (
        not config.layerwise_disaggregated_training 
        and parallel_state.is_pipeline_last_stage()
    )
    disaggregated_end_stage = (
        config.layerwise_disaggregated_training 
        and parallel_state.is_pipeline_first_stage() 
        and is_end_stage
    )
    if normal_last_stage or disaggregated_end_stage:
        if not collect_non_loss_data:
            outputs = loss_function(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                if not len(outputs) == 2:
                    raise ValueError()
                output_tensor, loss_reduced = outputs
                output_tensor *= parallel_state.get_context_parallel_world_size()
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_function(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def recv_forward_with_reqs(tensor_shapes, config, is_end_stage: bool = False, **kwargs):
    """Wrapper for p2p_communication_patch.recv_forward used with non-interleaving schedule."""
    input_tensors = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensor, reqs = p2p_communication_patch.recv_forward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            input_tensors.append(input_tensor)
            reps_list.append(reqs)
    return input_tensors, reps_list


def recv_backward_with_reqs(tensor_shapes, config, is_end_stage=False, **kwargs):
    """Wrapper for p2p_communication_patch.recv_backward used with non-interleaving schedule."""
    output_tensor_grads = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grad, reqs = p2p_communication_patch.recv_backward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            output_tensor_grads.append(output_tensor_grad)
            reps_list.append(reqs)
    return output_tensor_grads, reps_list


def send_forward(
    output_tensors, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication_patch.send_forward used with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication_patch.send_forward(output_tensor, config, is_end_stage, **kwargs)


def send_backward(
    input_tensor_grads, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication_patch.send_backward used with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication_patch.send_backward(input_tensor_grad, config, is_end_stage, **kwargs)


def get_all_batchs(mbn, data_iterator, model, config):

    device = f"npu:{torch.cuda.current_device()}"
    data_type = torch.int64
    hidden_size = config.hidden_size

    all_batchs = [[], []]
    recv_forward_tensor_shapes = []
    recv_backward_tensor_shapes = []

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(),
                                        group=parallel_state.get_pipeline_model_parallel_group())
    
    def get_batch_infos(attention_infos, thws, shapes, i_forward):
        seq_len, mbs = shapes[i_forward][0][0], shapes[i_forward][0][1]
        attention_mask = torch.ones(mbs, seq_len, device=device, dtype=data_type)

        for i, padding_info in enumerate(attention_infos[i_forward]):
            padding_side, padding_num = padding_info[0], padding_info[1]

            if padding_num == 0:
                continue

            if padding_side == 0:
                attention_mask[i, :padding_num] = torch.zeros(padding_num, device=device, dtype=data_type)
            else:
                attention_mask[i, -padding_num:] = torch.zeros(padding_num, device=device, dtype=data_type)

        image_grid_thw = torch.tensor(thws[i_forward], device=device, dtype=data_type)
        return attention_mask, image_grid_thw
    
    is_vit_last_stage = False
    if model.module.module.add_image_encoder:
        is_vit_last_stage = model.module.module.image_encoder.post_process
    
    tensor_shapes = torch.empty(
        mbn,
        3 + 5 * config.micro_batch_size,
        device=device,
        dtype=data_type
    )

    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        for i in range(mbn):
            batch = get_batch(data_iterator[0], is_vit_last_stage)
            mbs, seq_len = batch["input_ids"].shape[0], batch["input_ids"].shape[1]
            tensor_shapes[i, :3] = torch.tensor([seq_len, mbs, hidden_size], device=device, dtype=data_type)

            attention_mask = batch["attention_mask"]  # [mbs, seq_len]
            image_grid_thw = batch["image_grid_thw"]  # [mbs, 3]

            padding_side = (attention_mask[:, 0] != 0).long().unsqueeze(1)  # [mbs, 1]
            padding_num = (seq_len - attention_mask.sum(dim=1)).unsqueeze(1)  # [mbs, 1]
            tensor_shapes[i][3:] = torch.cat([padding_side, padding_num, image_grid_thw], dim=1).flatten()  # [mbs * 5, ]

            tensor_shape = [(seq_len, mbs, config.hidden_size)]

            all_batchs[0].append(batch)
            all_batchs[1].append(batch)
            recv_forward_tensor_shapes.append(tensor_shape)
            recv_backward_tensor_shapes.append(tensor_shape)

        _broadcast(tensor_shapes)
    else:
        _broadcast(tensor_shapes)

        tensor_shapes_tolist = tensor_shapes.tolist()

        shapes = [[tuple(shape[:3])] for shape in tensor_shapes_tolist]
        recv_forward_tensor_shapes = shapes
        recv_backward_tensor_shapes = shapes.copy()

        attention_infos = [[(shape[3 + 5 * i: 5 + 5 * i]) for i in range(config.micro_batch_size)] for shape in tensor_shapes_tolist]
        thws = [[(shape[5 + 5 * i: 8 + 5 * i]) for i in range(config.micro_batch_size)] for shape in tensor_shapes_tolist]

        for i in range(mbn):
            seq_len, mbs = shapes[i][0][0], shapes[i][0][1]
            input_ids = torch.zeros(mbs, seq_len, device=device, dtype=data_type)
            attention_mask, image_grid_thw = get_batch_infos(attention_infos, thws, shapes, i)
            batch = {
                'input_ids': input_ids,
                'labels': None,
                'pixel_values': None,
                'attention_mask': attention_mask,
                'image_grid_thw': image_grid_thw,
                'tranfer': None
            }

            all_batchs[0].append(batch)
            all_batchs[1].append(batch)
    
    return all_batchs, recv_forward_tensor_shapes, recv_backward_tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """
    Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise.
    """

    parallel_state.set_virtual_pipeline_model_parallel_rank(0)
    
    if not isinstance(model, list):
        raise TypeError("cloud-edge pipeline parallelism expected model chunking")
    if not all(isinstance(chunk, torch.nn.Module) for chunk in model):
        raise TypeError("invalid model chunking")

    if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        data_iterator = [None]

    config = get_model_config(model[0])
    config.variable_seq_lengths = False
    config.layerwise_disaggregated_training = True
    forward_step_func = forward_step_impl

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
        if isinstance(no_sync_func, list):
            for func in no_sync_func:
                no_sync_context = func()
                no_sync_context.__enter__()
        else:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
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

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

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
        from collections import defaultdict
        input_tensors = defaultdict(list)
        output_tensors = defaultdict(list)
    forward_data_store = []

    global default_stream
    if default_stream is None:
        default_stream = torch.cuda.default_stream()

    global stream_ping
    if stream_ping is None:
        stream_ping = torch.cuda.Stream()

    global stream_pang
    if stream_pang is None:
        stream_pang = torch.cuda.Stream()
    
    global stream_last_to_first
    if stream_last_to_first is None:
        stream_last_to_first = torch.cuda.Stream()
    
    global stream_first_to_last
    if stream_first_to_last is None:
        stream_first_to_last = torch.cuda.Stream()
    
    group_ping = get_pipeline_model_parallel_group()
    group_pang = get_pipeline_model_parallel_group_alternate()
    group_last_to_first = get_pipeline_model_parallel_group_last_to_first()
    group_first_to_last = get_pipeline_model_parallel_group_first_to_last()

    if parallel_state.get_pipeline_model_parallel_rank() % 2 == 0:
        receive_forward_stream = receive_backward_stream = stream_ping
        send_forward_stream = send_backward_stream = stream_pang
        receive_forward_group = receive_backward_group = group_ping
        send_forward_group = send_backward_group = group_pang
    else:
        receive_forward_stream = receive_backward_stream = stream_pang
        send_forward_stream = send_backward_stream = stream_ping
        receive_forward_group = receive_backward_group = group_pang
        send_forward_group = send_backward_group = group_ping
    
    # PP为奇数时，需要特殊处理
    if parallel_state.get_pipeline_model_parallel_world_size() % 2 == 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            receive_forward_stream = stream_last_to_first
            receive_forward_group = group_last_to_first
            send_backward_stream = stream_first_to_last
            send_backward_group = group_first_to_last
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            receive_backward_stream = stream_first_to_last
            receive_backward_group = group_first_to_last
            send_forward_stream = stream_last_to_first
            send_forward_group = group_last_to_first

    if not isinstance(receive_forward_group, list):
        receive_forward_group = [receive_forward_group]
    if not isinstance(receive_backward_group, list):
        receive_backward_group = [receive_backward_group]
    if not isinstance(send_forward_group, list):
        send_forward_group = [send_forward_group]
    if not isinstance(send_backward_group, list):
        send_backward_group = [send_backward_group]

    def wait_helper(reqs_list):
        is_wait = False
        recv_prev = False
        for reqs in reqs_list:
            if reqs is None:
                continue
            if "recv_prev" in reqs.keys():
                recv_prev = True
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
                is_wait = True
        if is_wait:
            if recv_prev:
                default_stream.wait_stream(receive_forward_stream)
            else:
                default_stream.wait_stream(receive_backward_stream)
        reqs_list = []
    
    def send_forward_with_stream(output_tensor, send_tensor_shapes, config, is_end_stage=False, **kwargs):
        with torch.cuda.stream(send_forward_stream):
            send_forward_stream.wait_stream(default_stream)
            send_forward(output_tensor, send_tensor_shapes, config, is_end_stage, **kwargs)
            if output_tensor is not None:
                if isinstance(output_tensor, list):
                    for output_tensor_i in output_tensor:
                        if output_tensor_i is not None:
                            output_tensor_i.record_stream(send_forward_stream)
                else:
                    output_tensor.record_stream(send_forward_stream)
    
    def recv_forward_with_stream(recv_tensor_shapes, config, is_end_stage=False, **kwargs):
        with torch.cuda.stream(receive_forward_stream):
            input_tensor, reqs_list = recv_forward_with_reqs(recv_tensor_shapes, config, is_end_stage, **kwargs)
            for input_tensor_i in input_tensor:
                if input_tensor_i is not None:
                    input_tensor_i.record_stream(default_stream)
        if 'wait_on_reqs' in kwargs.keys():
            if kwargs['wait_on_reqs'] is True:
                default_stream.wait_stream(receive_forward_stream)
                return input_tensor
        else:
            default_stream.wait_stream(receive_forward_stream)
            return input_tensor
        return input_tensor, reqs_list

    def send_backward_with_stream(input_tensor_grad, recv_tensor_shapes, config, is_end_stage=False, **kwargs):
        with torch.cuda.stream(send_backward_stream):
            send_backward_stream.wait_stream(default_stream)
            send_backward(input_tensor_grad, recv_tensor_shapes, config, is_end_stage, **kwargs)
            if input_tensor_grad is not None:
                if isinstance(input_tensor_grad, list):
                    for input_tensor_grad_i in input_tensor_grad:
                        if input_tensor_grad_i is not None:
                            input_tensor_grad_i.record_stream(send_backward_stream)
                else:
                    input_tensor_grad.record_stream(send_backward_stream)
    
    def recv_backward_with_stream(recv_tensor_shapes, config, is_end_stage=False, **kwargs):
        with torch.cuda.stream(receive_backward_stream):
            output_tensor_grad, reqs_list = recv_backward_with_reqs(
                recv_tensor_shapes, config, is_end_stage, **kwargs
                )
            for output_tensor_grad_i in output_tensor_grad:
                if output_tensor_grad_i is not None:
                    output_tensor_grad_i.record_stream(default_stream)
        default_stream.wait_stream(receive_backward_stream)
        return output_tensor_grad, reqs_list
    
    all_batchs, recv_forward_tensor_shapes, recv_backward_tensor_shapes = get_all_batchs(
        num_microbatches, data_iterator, model[0], config)

    pp_group = get_pipeline_model_parallel_group()
    if not isinstance(pp_group, list):
        pp_group = [pp_group]

    num_forward_end_backward_start = int(
        (4 * parallel_state.get_pipeline_model_parallel_world_size() + 1) / 6 + .00001
    )
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        num_2f2b = num_microbatches - num_forward_end_backward_start
        cooldown_iter_num = num_forward_end_backward_start
    else:
        num_2f2b = num_microbatches_remaining
        cooldown_iter_num = num_warmup_microbatches
    
    input_tensor_tmp = None
    reqs_list = []
    vdp_input_tensor_tmp = None
    vdp_reqs_list = []
    pp_group_name = "".join(str(i) for i in torch.distributed.get_process_group_ranks(pp_group[0]))

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        last_iteration = i == (num_warmup_microbatches - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None
        
        if i == 0:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            input_tensor = recv_forward_with_stream(recv_tensor_shapes, config, group=receive_forward_group)
        
        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            this_iterator = None
            this_model = model[0]

            wait_helper(reqs_list)
            if input_tensor_tmp is not None:
                input_tensor = input_tensor_tmp
                input_tensor_tmp = None

            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            input_tensor_tmp, reqs_list = recv_forward_with_stream(
                recv_tensor_shapes, config, group=receive_forward_group, wait_on_reqs=False
            )

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batchs[0].pop(0)
            )

            send_forward_with_stream(output_tensor, send_tensor_shapes, config, group=send_forward_group)
            total_num_tokens += num_tokens

            if not forward_only:
                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
        else:
            if last_iteration and num_forward_end_backward_start > 0:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes, config, group=receive_forward_group, is_end_stage=True, wait_on_reqs=False
                )
            
            this_iterator = None
            this_model = model[0]

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batchs[0].pop(0)
            )
            send_forward_with_stream(output_tensor, send_tensor_shapes, config, group=send_forward_group)

            if not forward_only:
                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

    # Run forward-end-backward-start at end stage for PP0
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        if num_warmup_microbatches == 0:
            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            vdp_input_tensor_tmp = recv_forward_with_stream(
                recv_tensor_shapes, config, group=receive_forward_group, is_end_stage=True
            )
        
        for i in range(num_forward_end_backward_start):
            last_iteration = i == (num_forward_end_backward_start - 1)

            wait_helper(vdp_reqs_list)
            if vdp_input_tensor_tmp is not None:
                input_tensor_end = vdp_input_tensor_tmp
                vdp_input_tensor_tmp = None
            
            if not last_iteration:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes, config, group=receive_forward_group, is_end_stage=True, wait_on_reqs=False
                )
            
            this_iterator = None
            this_model = model[1]

            output_tensor_end, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor_end,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                is_end_stage=True,
                batch=all_batchs[1].pop(0)
            )
            total_num_tokens += num_tokens

            if not forward_only:
                output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                if num_2f2b == 0 and cooldown_iter_num == 0 and last_iteration:
                    if config.grad_sync_func is None or rank == 0:
                        enable_grad_sync()
                
                deallocate_output_tensor(output_tensor_end[0], config.deallocate_pipeline_outputs)

                input_tensor_grad_end = backward_step(
                    input_tensor_end, output_tensor_end, output_tensor_grad_end, model_type, config
                )

                if last_iteration:
                    input_tensor_end = None

                send_backward_with_stream(
                    input_tensor_grad_end, send_tensor_shapes, config, group=send_backward_group, is_end_stage=True
                )

    # Run 2F2B in steady state
    output_tensor_grad_tmp = None
    vdp_output_tensor_grad_tmp = None

    for i in range(num_2f2b):
        last_iteration = i == (num_2f2b - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )

        else:
            checkpoint_activations_microbatch = None
        
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                recv_tensor_shapes, config, group=receive_forward_group, is_end_stage=True, wait_on_reqs=False
            )
        
        if i < num_microbatches_remaining:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                wait_helper(reqs_list)
                if input_tensor_tmp is not None:
                    input_tensor = input_tensor_tmp
                    input_tensor_tmp = None
                
                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                output_tensor_grad, reqs_list = recv_backward_with_stream(
                    recv_tensor_shapes, config, group=receive_backward_group, wait_on_reqs=False
                )

                this_iterator = None
                this_model = model[0]

                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
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
                    batch=all_batchs[0].pop(0)
                )
                total_num_tokens += num_tokens

                send_forward_with_stream(output_tensor, send_tensor_shapes, config, group=send_forward_group)

                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            else:
                input_tensor = [None] * len(recv_tensor_shapes)
                this_iterator = None
                this_model = model[0]

                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batchs[0].pop(0)
                )
                total_num_tokens += num_tokens

                send_forward_with_stream(output_tensor, send_tensor_shapes, config, group=send_forward_group)

                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            wait_helper(vdp_reqs_list)
            if vdp_input_tensor_tmp is not None:
                input_tensor_end = vdp_input_tensor_tmp
                vdp_input_tensor_tmp = None
            
            recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
            vdp_output_tensor_grad_tmp, vdp_reqs_list = recv_backward_with_stream(
                recv_tensor_shapes, config, group=receive_backward_group, wait_on_reqs=False
            )

            this_iterator = None
            this_model = model[1]

            output_tensor_end, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor_end,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                is_end_stage=True,
                batch=all_batchs[1].pop(0)
            )
            total_num_tokens += num_tokens

            if not forward_only:
                deallocate_output_tensor(output_tensor_end[0], config.deallocate_pipeline_outputs)

                output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                input_tensor_grad_end = backward_step(
                    input_tensor_end, output_tensor_end, output_tensor_grad_end, model_type, config
                )

                send_backward_with_stream(
                    input_tensor_grad_end, send_tensor_shapes, config, group=send_backward_group, is_end_stage=True
                )

        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            wait_helper(reqs_list)

            input_tensor = input_tensors[pp_group_name].pop(0)
            output_tensor = output_tensors[pp_group_name].pop(0)

            if cooldown_iter_num == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()
            
            if not last_iteration:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                input_tensor_tmp, reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes, config, group=receive_forward_group, wait_on_reqs=False
                )
            elif cooldown_iter_num > 0:
                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                output_tensor_grad_tmp, reqs_list = recv_backward_with_stream(
                    recv_tensor_shapes, config, group=receive_backward_group, wait_on_reqs=False
                )
            
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward_with_stream(input_tensor_grad, send_tensor_shapes, config, group=send_backward_group)
        else:
            wait_helper(vdp_reqs_list)
            if vdp_output_tensor_grad_tmp is not None:
                output_tensor_grad = vdp_output_tensor_grad_tmp
                vdp_output_tensor_grad_tmp = None

            input_tensor = input_tensors[pp_group_name].pop(0)
            output_tensor = output_tensors[pp_group_name].pop(0)

            if cooldown_iter_num == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()
            
            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward_with_stream(input_tensor_grad, send_tensor_shapes, config, group=send_backward_group)


    input_tensor_end = None
    input_tensor = None

    # Run cooldown backward passes
    if not forward_only:
        for i in range(cooldown_iter_num):
            last_iteration = i == (cooldown_iter_num - 1)
            
            if last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()
            
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                vdp_output_tensor_grad_tmp, _ = recv_backward_with_stream(
                    recv_tensor_shapes, config, group=receive_backward_group
                )
            
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                input_tensor = input_tensors[pp_group_name].pop(0)
                output_tensor = output_tensors[pp_group_name].pop(0)

                if num_2f2b == 0 and not last_iteration:
                    recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                    output_tensor_grad, reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes, config, group=receive_backward_group
                    )
                
                wait_helper(reqs_list)
                if output_tensor_grad_tmp is not None:
                    output_tensor_grad = output_tensor_grad_tmp
                    output_tensor_grad_tmp = None

                if not last_iteration:
                    recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                    output_tensor_grad_tmp, reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes, config, group=receive_backward_group, wait_on_reqs=False
                    )

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

                send_backward_with_stream(input_tensor_grad, send_tensor_shapes, config, group=send_backward_group)
            
            else:
                input_tensor = input_tensors[pp_group_name].pop(0)
                output_tensor = output_tensors[pp_group_name].pop(0)
                
                wait_helper(vdp_reqs_list)
                if vdp_output_tensor_grad_tmp is not None:
                    output_tensor_grad = vdp_output_tensor_grad_tmp
                    vdp_output_tensor_grad_tmp = None

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

                send_backward_with_stream(input_tensor_grad, send_tensor_shapes, config, group=send_backward_group)
        
        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                for this_model in model:
                    config.grad_sync_func(this_model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        this_model = model if parallel_state.is_pipeline_first_stage(ignore_virtual=True) else [model[0]]
        config.finalize_model_grads_func(
            this_model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'enable_cuda_graph') and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store
