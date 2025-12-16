import os

import torch
import mindspeed.megatron_adaptor
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from tests_extend.unit_tests.common import DistributedTest
from tests_extend.commons import set_random_seed, initialize_model_parallel

from tests.ut.utils import judge_expression
from mindspeed_mm.models.common.attention import ParallelAttention


TP_SIZE = 2
CP_SIZE = 4
HIDDEN_SIZE = 2304
NUM_ATTENTION_HEADS = 24
SEQUENCE_LENGTH = 128
MICRO_BATCH_SIZE = 2
DTYPE = torch.bfloat16
OUTPUT_SUM = -161.0
ATTENTION_DATA_PATH = "/home/ci_resource/data/attention"


class TestParallelAttention(DistributedTest):
    world_size = 8

    def test_forward(self):
        args = parse_args(None, True)
        args.tensor_model_parallel_size = TP_SIZE
        args.context_parallel_size = CP_SIZE
        args.hidden_size = HIDDEN_SIZE
        args.num_attention_heads = NUM_ATTENTION_HEADS
        args.sequence_parallel = True
        args.context_parallel_algo = "ulysses_cp_algo"
        args.sparse_mode = 0
        args.params_dtype = DTYPE
        args.num_layers = 1
        args.seq_length = SEQUENCE_LENGTH
        set_args(args)

        initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            context_parallel_size=args.context_parallel_size
        )

        set_random_seed(1234)
        model_parallel_cuda_manual_seed(1234)

        # init ParallelAttention
        parallelattention = ParallelAttention(
            query_dim=args.hidden_size,
            key_dim=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            hidden_size=args.hidden_size,
            proj_q_bias=True,
            proj_k_bias=True,
            proj_v_bias=True,
            proj_out_bias=True,
            dropout=0.0,
            use_qk_norm=True,
            norm_type="layernorm",
            fa_layout="sbh"
        )
        
        # load state dict
        attention_state_dict = torch.load(f"{ATTENTION_DATA_PATH}/attention_state_dict_rank_{torch.distributed.get_rank()}.pt", map_location="cpu")
        parallelattention.load_state_dict(attention_state_dict)
        parallelattention = parallelattention.to("npu")

        # [sequence length, batch size, hidden size]
        hidden_states = torch.load(f"{ATTENTION_DATA_PATH}/attention_input.pt").to(DTYPE).to("npu")
        attention_mask = None

        out = parallelattention(query=hidden_states, mask=attention_mask)

        # output judge
        judge_expression(out.sum().item() == OUTPUT_SUM)