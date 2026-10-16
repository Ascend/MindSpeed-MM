import copy
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers.models.glm4v_moe.configuration_glm4v_moe import (
    Glm4vMoeConfig,
    Glm4vMoeTextConfig,
    Glm4vMoeVisionConfig,
)
from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionEmbeddings
from transformers.models.glm4v_moe import modeling_glm4v_moe as hf_modeling


REPO_ROOT = Path(__file__).parents[4]
MODEL_PATH = REPO_ROOT / "mindspeed_mm/fsdp/models/glm4_5v/modeling_glm4_5v.py"
LAUNCHER_PATH = REPO_ROOT / "examples/glm4.5v/finetune_glm4_5v_106B_v1.sh"
CONFIG_PATH = REPO_ROOT / "examples/glm4.5v/glm4_5v_106B_config_v1.yaml"


class _ModelRegister:
    @staticmethod
    def register(_name):
        return lambda model_cls: model_cls


def _load_source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_chunkloss_modules():
    module_paths = [
        "mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss",
        "mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_fused",
        "mindspeed_mm.fsdp.utils.constants",
        "mindspeed_mm.fsdp.utils.str_match",
    ]
    dependencies = {
        name: _load_source_module(name, name.replace(".", "/") + ".py")
        for name in module_paths
    }
    parallel_state = types.ModuleType("mindspeed_mm.fsdp.distributed.parallel_state")
    parallel_state.get_parallel_state = lambda: SimpleNamespace(is_cp_enable=lambda: False)
    communication = types.ModuleType("mindspeed_mm.fsdp.distributed.context_parallel.communication")
    communication.split_forward_gather_backward_with_cp = lambda tensor, dim: tensor
    log = types.ModuleType("mindspeed_mm.fsdp.log")
    log.print_rank = lambda log_func, message: log_func(message)
    dependencies.update({module.__name__: module for module in (parallel_state, communication, log)})
    with patch.dict(sys.modules, dependencies):
        loss = _load_source_module("_glm4v_loss", "mindspeed_mm/fsdp/loss/loss_func.py")
        head = _load_source_module(
            "_glm4v_chunk_head", "mindspeed_mm/fsdp/features/memory/chunkloss/chunkloss_lm_head.py"
        )
    return loss, head


def _load_modeling_module():
    register_module = types.ModuleType("mindspeed_mm.fsdp.utils.register")
    register_module.model_register = _ModelRegister()
    device_module = types.ModuleType("mindspeed_mm.fsdp.utils.device")
    device_module.IS_NPU_AVAILABLE = False
    parallel_state_module = types.ModuleType("mindspeed_mm.fsdp.distributed.parallel_state")
    parallel_state_module.get_parallel_state = lambda: None
    communication_module = types.ModuleType("mindspeed_mm.fsdp.distributed.context_parallel.communication")
    communication_module.split_forward_gather_backward_with_cp = lambda tensor, dim: tensor
    communication_module.gather_forward_split_backward = lambda tensor, group, dim: tensor
    communication_module.packed_data_split_forward_gather_backward_with_cp = (
        lambda tensor, dim, seq_lens: tensor
    )
    communication_module.packed_data_gather_forward_split_backward_with_cp = (
        lambda tensor, dim, seq_lens: tensor
    )
    cp_utils_module = types.ModuleType("mindspeed_mm.fsdp.distributed.context_parallel.utils")
    cp_utils_module.cal_split_sizes_multi = lambda seq_lens, world_size: torch.tensor(
        [[int(seq_len) for seq_len in seq_lens]] * world_size
    )

    dependencies = {
        register_module.__name__: register_module,
        device_module.__name__: device_module,
        parallel_state_module.__name__: parallel_state_module,
        communication_module.__name__: communication_module,
        cp_utils_module.__name__: cp_utils_module,
    }
    spec = importlib.util.spec_from_file_location("_glm4_5v_modeling_under_test", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


def _tiny_text_config(num_hidden_layers=0):
    return Glm4vMoeTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=2,
        moe_intermediate_size=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_routed_experts=4,
        first_k_dense_replace=0,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 0, 0]},
    )


def _tiny_config():
    vision_config = Glm4vMoeVisionConfig(
        depth=0,
        hidden_size=8,
        num_heads=2,
        image_size=4,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=8,
        intermediate_size=16,
    )
    return Glm4vMoeConfig(
        text_config=_tiny_text_config().to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=29,
        video_token_id=30,
    )


@pytest.mark.parametrize("routes", [
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
    [[0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
])
def test_stacked_experts_match_reference_forward_and_backward(routes):
    modeling = _load_modeling_module()
    torch.manual_seed(0)
    experts = modeling.Glm4vMoeTextExperts(_tiny_text_config(), intermediate_size=4).double()
    experts.gate_up_proj.data.normal_()
    experts.down_proj.data.normal_()
    hidden_states = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    topk_indices = torch.tensor(routes)
    topk_weights = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)
    inputs = (hidden_states, topk_weights, experts.gate_up_proj, experts.down_proj)
    ref_hidden, ref_routing, ref_gate_up, ref_down = [x.detach().clone().requires_grad_() for x in inputs]

    expected_tokens = []
    for token_idx in range(hidden_states.shape[0]):
        routed_outputs = []
        for route_idx in range(topk_indices.shape[1]):
            expert_idx = topk_indices[token_idx, route_idx]
            gate_weight, up_weight = ref_gate_up[expert_idx].T.chunk(2, dim=0)
            gate = F.linear(ref_hidden[token_idx], gate_weight)
            up = F.linear(ref_hidden[token_idx], up_weight)
            expert_output = F.linear(F.silu(gate) * up, ref_down[expert_idx].T)
            routed_outputs.append(ref_routing[token_idx, route_idx] * expert_output)
        expected_tokens.append(torch.stack(routed_outputs).sum(0))
    expected = torch.stack(expected_tokens)

    actual = experts(hidden_states, topk_weights, topk_indices)
    torch.testing.assert_close(actual, expected)
    grad_output = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, inputs, grad_output)
    expected_grads = torch.autograd.grad(
        expected, (ref_hidden, ref_routing, ref_gate_up, ref_down), grad_output
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)
    unused_experts = sorted(set(range(experts.num_experts)) - set(topk_indices.flatten().tolist()))
    if unused_experts:
        assert torch.count_nonzero(actual_grads[2][unused_experts]) == 0
        assert torch.count_nonzero(actual_grads[3][unused_experts]) == 0


@pytest.mark.parametrize("routes", [
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3]],
    [[0, 1], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]],
])
def test_grouped_experts_match_reference_forward_and_backward(routes):
    modeling = _load_modeling_module()
    config = _tiny_text_config()
    config.use_grouped_expert_matmul = True
    parallel_state = MagicMock()
    parallel_state.is_ep_enable.return_value = False
    with patch.object(modeling, "get_parallel_state", return_value=parallel_state):
        experts = modeling.Glm4vMoeTextExperts(config, intermediate_size=4).double()
    torch.manual_seed(0)
    experts.gate_up_proj.data.normal_()
    experts.down_proj.data.normal_()
    reference = copy.deepcopy(experts)
    reference.use_grouped_expert_matmul = False

    calls = []

    class _NPUInput:
        """Select the NPU branch, then use real CPU tensors for its arithmetic."""

        def __init__(self, tensor):
            self.tensor = tensor
            self.shape = tensor.shape
            self.dtype = tensor.dtype
            self.device = SimpleNamespace(type="npu")

        def reshape(self, *shape):
            return self.tensor.reshape(*shape)

    def permute(hidden, indices):
        calls.append("permute")
        order = indices.flatten().argsort(stable=True)
        return hidden[order // indices.shape[1]], order

    def swiglu(hidden, dim):
        calls.append("swiglu")
        gate, up = hidden.chunk(2, dim=dim)
        return F.silu(gate) * up

    def unpermute(hidden, row_map, probs):
        calls.append("unpermute")
        restored = torch.zeros_like(hidden).index_copy(0, row_map, hidden)
        return (restored.reshape(*probs.shape, -1) * probs.unsqueeze(-1)).sum(dim=1)

    fake_torch_npu = SimpleNamespace(
        npu_moe_token_permute=permute,
        npu_swiglu=swiglu,
        npu_moe_token_unpermute=unpermute,
    )

    def fake_group_gemm(hidden, weight, group_list):
        calls.append("group_gemm")
        chunks = hidden.split(group_list.to(torch.int64).tolist())
        return torch.cat([chunk @ weight[index] for index, chunk in enumerate(chunks)])

    hidden_states = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    topk_weights = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)
    topk_indices = torch.tensor(routes)
    ref_hidden = hidden_states.detach().clone().requires_grad_()
    ref_weights = topk_weights.detach().clone().requires_grad_()
    expected = reference(ref_hidden, ref_weights, topk_indices).reshape_as(hidden_states)

    def fake_histc(values, bins, min, max):
        return torch.bincount(values.reshape(-1).long(), minlength=bins).float()

    with (
        patch.object(modeling, "IS_NPU_AVAILABLE", True),
        patch.object(modeling, "torch_npu", fake_torch_npu, create=True),
        patch.object(modeling, "fused_group_gemm", fake_group_gemm, create=True),
        patch("torch.histc", side_effect=fake_histc),
    ):
        output = experts(_NPUInput(hidden_states), topk_weights, topk_indices)

    torch.testing.assert_close(output, expected)
    actual_grads = torch.autograd.grad(
        output.square().sum(), (hidden_states, topk_weights, experts.gate_up_proj, experts.down_proj)
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (ref_hidden, ref_weights, reference.gate_up_proj, reference.down_proj)
    )
    for actual, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected_grad)
    unused = sorted(set(range(config.n_routed_experts)) - set(topk_indices.flatten().tolist()))
    if unused:
        assert torch.count_nonzero(actual_grads[2][unused]) == 0
        assert torch.count_nonzero(actual_grads[3][unused]) == 0
    assert calls == ["permute", "group_gemm", "swiglu", "group_gemm", "unpermute"]


def test_text_model_applies_cp_split_and_loss_gather():
    modeling = _load_modeling_module()
    model = modeling.Glm4vMoeFSDPForConditionalGeneration(_tiny_config())
    parallel_state = MagicMock()
    parallel_state.is_cp_enable.return_value = True
    parallel_state.get_cp_group.return_value = object()

    with (
        patch.object(modeling, "get_parallel_state", return_value=parallel_state, create=True),
        patch.object(
            modeling,
            "split_forward_gather_backward_with_cp",
            side_effect=lambda tensor, dim: tensor,
            create=True,
        ) as split,
        patch.object(
            modeling,
            "gather_forward_split_backward",
            side_effect=lambda tensor, group, dim: tensor,
            create=True,
        ) as gather,
    ):
        input_ids = torch.tensor([[1, 2, 3, 4]])
        model(input_ids=input_ids, labels=input_ids)

    assert [call.kwargs["dim"] for call in split.call_args_list] == [2, 1, 1]
    gather.assert_called_once()


def test_text_model_matches_hf_forward_and_backward():
    modeling = _load_modeling_module()
    torch.manual_seed(1)
    config = _tiny_text_config(num_hidden_layers=1)
    config._attn_implementation_internal = "eager"
    reference = hf_modeling.Glm4vMoeTextModel(copy.deepcopy(config))
    model = modeling.Glm4vFusedMoeTextModel(config)
    state = dict(reference.state_dict())
    gate_up, down = [], []
    for expert_idx in range(config.n_routed_experts):
        prefix = f"layers.0.mlp.experts.{expert_idx}"
        gate = state.pop(f"{prefix}.gate_proj.weight")
        up = state.pop(f"{prefix}.up_proj.weight")
        gate_up.append(torch.cat((gate, up), dim=0).T)
        down.append(state.pop(f"{prefix}.down_proj.weight").T)
    state["layers.0.mlp.experts.gate_up_proj"] = torch.stack(gate_up)
    state["layers.0.mlp.experts.down_proj"] = torch.stack(down)
    model.load_state_dict(state, strict=True)
    parallel_state = MagicMock()
    parallel_state.is_cp_enable.return_value = False

    inputs = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]]),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "position_ids": torch.arange(4).view(1, 1, 4).expand(3, 2, 4),
    }
    with patch.object(modeling, "get_parallel_state", return_value=parallel_state):
        actual = model(**inputs).last_hidden_state
        expected = reference(**inputs).last_hidden_state

    torch.testing.assert_close(actual, expected)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(model.embed_tokens.weight.grad, reference.embed_tokens.weight.grad)
    torch.testing.assert_close(model.layers[0].mlp.gate.weight.grad, reference.layers[0].mlp.gate.weight.grad)
    for expert_idx, ref_expert in enumerate(reference.layers[0].mlp.experts):
        actual_gate, actual_up = model.layers[0].mlp.experts.gate_up_proj.grad[expert_idx].T.chunk(2)
        for actual_grad, ref_grad in (
            (actual_gate, ref_expert.gate_proj.weight.grad),
            (actual_up, ref_expert.up_proj.weight.grad),
            (model.layers[0].mlp.experts.down_proj.grad[expert_idx].T, ref_expert.down_proj.weight.grad),
        ):
            torch.testing.assert_close(actual_grad, ref_grad if ref_grad is not None else torch.zeros_like(actual_grad))


def test_vision_model_applies_cp_to_packed_tokens():
    modeling = _load_modeling_module()
    vision = modeling.Glm4vFusedMoeVisionModel(_tiny_config().vision_config)

    class _VisionEmbeddings(nn.Module):
        def forward(self, embeddings, lengths, image_shapes, h_coords, w_coords):
            return embeddings

    class _Downsample(nn.Module):
        def forward(self, hidden_states):
            return hidden_states.mean(dim=(-1, -2))

    class _CaptureBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.kwargs = None

        def forward(self, hidden_states, **kwargs):
            self.kwargs = kwargs
            return hidden_states

    vision.patch_embed = nn.Identity()
    vision.post_conv_layernorm = nn.Identity()
    vision.embeddings = _VisionEmbeddings()
    vision.post_layernorm = nn.Identity()
    vision.downsample = _Downsample()
    vision.merger = nn.Identity()
    capture_block = _CaptureBlock()
    vision.blocks = nn.ModuleList([capture_block])

    parallel_state = MagicMock()
    parallel_state.is_cp_enable.return_value = True
    parallel_state.is_ring_enable.return_value = True
    parallel_state.get_ring_group_size.return_value = 2
    parallel_state.get_ring_rank.return_value = 0

    with (
        patch.object(modeling, "get_parallel_state", return_value=parallel_state),
        patch.object(
            modeling,
            "packed_data_split_forward_gather_backward_with_cp",
            side_effect=lambda tensor, dim, seq_lens: tensor,
            create=True,
        ) as split,
        patch.object(
            modeling,
            "packed_data_gather_forward_split_backward_with_cp",
            side_effect=lambda tensor, dim, seq_lens: tensor,
            create=True,
        ) as gather,
        patch.object(
            modeling,
            "cal_split_sizes_multi",
            return_value=torch.tensor([[2, 2], [2, 2]], dtype=torch.int32),
        ),
    ):
        output = vision(torch.randn(8, 8), torch.tensor([[2, 2, 2]]))

    assert split.call_count == 3
    assert gather.call_count == 2
    assert split.call_args_list[-1].kwargs["seq_lens"].tolist() == [1, 1]
    assert gather.call_args_list[-1].kwargs["seq_lens"].tolist() == [1, 1]
    assert capture_block.kwargs["cu_seqlens"].tolist() == [0, 2, 4]
    assert capture_block.kwargs["seq_split_lens"].tolist() == [[2, 2], [2, 2]]
    assert output.shape == (2, 8)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_chunk_loss_matches_dense_loss_and_gradients(mode, batch_size):
    modeling = _load_modeling_module()
    loss_module, head_module = _load_chunkloss_modules()
    torch.manual_seed(2)
    baseline = modeling.Glm4vMoeFSDPForConditionalGeneration(_tiny_config())
    config = _tiny_config()
    feature_args = SimpleNamespace(
        enable_chunk_loss=mode == "static",
        enable_dynamic_chunk_loss=mode == "dynamic",
    )
    modeling.Glm4vMoeFSDPForConditionalGeneration.overwrite_transformer_config(
        config, SimpleNamespace(), feature_args
    )
    model = modeling.Glm4vMoeFSDPForConditionalGeneration(config)
    model.load_state_dict(baseline.state_dict(), strict=True)
    input_ids = torch.randint(1, 28, (batch_size, 7))
    labels = input_ids.clone()
    labels[:, :2] = -100
    labels[0, 4] = -100
    baseline.loss_function = loss_module.build_loss_func("default", labels=labels)
    model.loss_function = loss_module.build_loss_func(
        "default", labels=labels,
        chunk_size=2 if mode == "static" else None,
        total_chunk_size=4 if mode == "dynamic" else None,
    )
    head_module.apply_chunkloss_module(model.lm_head)
    parallel_state = MagicMock()
    parallel_state.is_cp_enable.return_value = False

    with (
        patch.object(modeling, "get_parallel_state", return_value=parallel_state),
        patch.object(loss_module, "chunk_loss", wraps=loss_module.chunk_loss) as chunk_loss,
    ):
        expected = baseline(input_ids=input_ids, labels=labels)
        output = model(input_ids=input_ids, labels=labels)

    assert output.logits is None
    torch.testing.assert_close(output.loss, expected.loss)
    manual_loss = F.cross_entropy(expected.logits[:, :-1].reshape(-1, 32), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(expected.loss, manual_loss)
    assert chunk_loss.call_args.kwargs["chunk_size"] == (1 if mode == "dynamic" and batch_size == 3 else 2)
    (output.loss * 1.7).backward()
    (expected.loss * 1.7).backward()
    baseline_parameters = dict(baseline.named_parameters())
    for name, parameter in model.named_parameters():
        reference_grad = baseline_parameters[name].grad
        if reference_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, reference_grad)
    assert torch.count_nonzero(model.lm_head.weight.grad) > 0


def test_ep_balance_configuration_and_dispatch():
    modeling = _load_modeling_module()
    config = _tiny_config()
    feature_args = SimpleNamespace(
        loss_cfg=SimpleNamespace(router_aux_loss_coef=0.0, router_aux_loss_offload=False),
        enable_chunk_loss=False,
        enable_dynamic_chunk_loss=False,
        enable_ep_balance=True,
        ep_balance_plan=SimpleNamespace(max_dup_experts_num=3),
    )
    modeling.Glm4vMoeFSDPForConditionalGeneration.overwrite_transformer_config(
        config, SimpleNamespace(), feature_args
    )
    assert config.text_config.use_grouped_expert_matmul is True
    assert config.text_config.enable_ep_balance is True
    assert config.text_config.max_dup_experts_num == 3

    disabled_config = _tiny_config()
    modeling.Glm4vMoeFSDPForConditionalGeneration.overwrite_transformer_config(
        disabled_config, SimpleNamespace(use_grouped_expert_matmul=False), feature_args
    )
    assert disabled_config.text_config.use_grouped_expert_matmul is False

    parallel_state = MagicMock()
    parallel_state.is_ep_enable.return_value = False
    with patch.object(modeling, "get_parallel_state", return_value=parallel_state):
        experts = modeling.Glm4vMoeTextExperts(config.text_config, intermediate_size=4)
    experts.enable_ep_balance = False
    dispatcher = MagicMock(side_effect=lambda *args, **kwargs: args[3].to(args[1].dtype))
    dispatcher_module = types.ModuleType("mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher")
    dispatcher_module.ep_forward = dispatcher
    dispatcher_module.ep_mc2_forward = MagicMock()
    dispatcher_module.ep_allgather_forward = MagicMock()
    hidden_states = torch.randn(1, 4, 8, dtype=torch.bfloat16)
    topk_weights = torch.rand(4, 2)
    topk_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    ep_plan = SimpleNamespace(dispatcher="alltoall", use_npu_fused_ops=True)

    with patch.dict(sys.modules, {dispatcher_module.__name__: dispatcher_module}):
        output = experts.ep_forward(
            hidden_states,
            topk_weights,
            topk_indices,
            ep_group=object(),
            ep_plan=ep_plan,
        )

    assert output.shape == hidden_states.shape
    assert output.dtype == hidden_states.dtype
    assert dispatcher.call_args.kwargs["ep_balance_strategy"] is None

    strategy = SimpleNamespace(
        executor=MagicMock(),
        planner=MagicMock(),
    )
    experts.enable_ep_balance = True
    experts.ep_balance_strategy = strategy
    with patch.dict(sys.modules, {dispatcher_module.__name__: dispatcher_module}):
        experts.ep_forward(
            hidden_states,
            topk_weights,
            topk_indices,
            ep_group=object(),
            ep_plan=ep_plan,
        )

    assert dispatcher.call_args.kwargs["ep_balance_strategy"] is strategy
    assert strategy.executor.register_backward_dup_experts_grad_acc_hook.call_args_list[0].kwargs == {
        "name": "fc1"
    }
    assert strategy.executor.register_backward_dup_experts_grad_acc_hook.call_args_list[1].kwargs == {
        "name": "fc2"
    }


def test_feature_plan_covers_visual_and_text_towers():
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    features = config["features"]

    assert config["data"]["dataloader_param"]["collate_param"]["collator_id"] == "qwen3vl"
    assert config["model"]["attn_implementation"] == "flash_attention_2"
    assert config["parallel"]["ring_attention_size"] == 1
    assert config["parallel"]["ulysses_parallel_size"] == 1
    assert features["enable_chunk_loss"] is True
    assert features["enable_dynamic_chunk_loss"] is False
    assert features["chunkloss_plan"]["total_chunk_size"] == 4096
    assert features["recompute"] is True
    assert features["recompute_plan"]["apply_modules"] == [
        "model.visual.blocks.{*}",
        "model.language_model.layers.{*}",
    ]
    assert features["enable_activation_offload"] is False
    assert features["activation_offload_plan"]["apply_modules"] == [
        "model.visual.blocks.{*}",
        "model.language_model.layers.{*}",
    ]
    assert features["enable_ep_balance"] is False
    assert config["training"]["save"] == "./save_dir"
    assert config["training"]["save_interval"] == 10000


def test_feature_plan_module_paths_exist_on_model():
    modeling = _load_modeling_module()
    config = _tiny_config()
    config.vision_config.depth = 1
    config.text_config.num_hidden_layers = 1
    model = modeling.Glm4vMoeFSDPForConditionalGeneration(config)
    module_names = dict(model.named_modules())

    assert "model.visual.blocks.0" in module_names
    assert "model.language_model.layers.0" in module_names
    assert "model.language_model.layers.0.mlp.experts" in module_names
    assert isinstance(module_names["lm_head"], nn.Linear)


def test_vision_embeddings_use_native_forward_off_npu():
    modeling = _load_modeling_module()
    torch.manual_seed(3)
    embeddings = modeling.Glm4vFusedMoeVisionEmbeddings(_tiny_config().vision_config)
    reference = Glm4vMoeVisionEmbeddings(_tiny_config().vision_config)
    reference.load_state_dict(embeddings.state_dict())
    inputs = torch.randn(4, 8, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    grid = {
        "lengths": [4],
        "image_shapes": [[1, 2, 2]],
        "h_coords": torch.tensor([0, 0, 1, 1]),
        "w_coords": torch.tensor([0, 1, 0, 1]),
    }

    with patch.object(modeling, "IS_NPU_AVAILABLE", False):
        actual = embeddings(inputs, **grid)
    expected = reference(reference_inputs, **grid)
    torch.testing.assert_close(actual, expected)
    grad_output = torch.randn_like(actual)
    actual.backward(grad_output)
    expected.backward(grad_output)
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    torch.testing.assert_close(embeddings.position_embedding.weight.grad, reference.position_embedding.weight.grad)


@pytest.mark.parametrize("logits_to_keep", [0, 2, torch.tensor([0, 2])])
def test_unlabeled_forward_returns_selected_logits(logits_to_keep):
    modeling = _load_modeling_module()
    torch.manual_seed(4)
    model = modeling.Glm4vMoeFSDPForConditionalGeneration(_tiny_config()).eval()
    parallel_state = SimpleNamespace(is_cp_enable=lambda: False)
    inputs = torch.tensor([[1, 2, 3, 4]])
    with patch.object(modeling, "get_parallel_state", return_value=parallel_state):
        full = model(input_ids=inputs)
        selected = model(input_ids=inputs, logits_to_keep=logits_to_keep)
    indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    assert selected.loss is None
    torch.testing.assert_close(selected.logits, full.logits[:, indices])


def _cpu_rotary_mul(value, cos, sin):
    first, second = value.chunk(2, dim=-1)
    return value * cos + torch.cat((-second, first), dim=-1) * sin


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_npu_vision_rope_matches_hf_with_cpu_primitive(dtype):
    npu_patch = _load_source_module("_glm4v_npu_patch", "mindspeed_mm/fsdp/models/glm4_5v/npu_patch.py")
    torch.manual_seed(5)
    q = torch.randn(5, 2, 8, dtype=dtype, requires_grad=True)
    k = torch.randn(5, 2, 8, dtype=dtype, requires_grad=True)
    ref_q = q.detach().clone().requires_grad_()
    ref_k = k.detach().clone().requires_grad_()
    cos, sin = torch.randn(5, 8), torch.randn(5, 8)
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_rotary_mul = _cpu_rotary_mul

    expected = hf_modeling.apply_rotary_pos_emb_vision(ref_q, ref_k, cos, sin)
    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        actual = npu_patch.apply_rotary_pos_emb_vision(q, k, cos, sin)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)
    grad_outputs = tuple(torch.randn_like(value) for value in actual)
    actual_grads = torch.autograd.grad(actual, (q, k), grad_outputs)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k), grad_outputs)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize("rotary_dim", [6, 12])
@pytest.mark.parametrize("unsqueeze_dim", [1, 2])
def test_npu_mrope_matches_hf_with_cpu_primitive(rotary_dim, unsqueeze_dim):
    npu_patch = _load_source_module("_glm4v_npu_patch", "mindspeed_mm/fsdp/models/glm4_5v/npu_patch.py")
    torch.manual_seed(6)
    shape_q = (2, 4, 5, 12) if unsqueeze_dim == 1 else (2, 5, 4, 12)
    shape_k = (2, 2, 5, 12) if unsqueeze_dim == 1 else (2, 5, 2, 12)
    q = torch.randn(shape_q, requires_grad=True)
    k = torch.randn(shape_k, requires_grad=True)
    ref_q = q.detach().clone().requires_grad_()
    ref_k = k.detach().clone().requires_grad_()
    cos, sin = torch.randn(3, 2, 5, rotary_dim), torch.randn(3, 2, 5, rotary_dim)
    sections = [rotary_dim // 6] * 3
    torch_npu = types.ModuleType("torch_npu")
    torch_npu.npu_rotary_mul = _cpu_rotary_mul

    expected = hf_modeling.apply_multimodal_rotary_pos_emb(ref_q, ref_k, cos, sin, sections, unsqueeze_dim)
    with patch.dict(sys.modules, {"torch_npu": torch_npu}):
        actual = npu_patch.apply_multimodal_rotary_pos_emb(q, k, cos, sin, sections, unsqueeze_dim)
    for value, reference, original in zip(actual, expected, (q, k)):
        torch.testing.assert_close(value, reference)
        torch.testing.assert_close(value[..., rotary_dim:], original[..., rotary_dim:], rtol=0, atol=0)
    grad_outputs = tuple(torch.randn_like(value) for value in actual)
    actual_grads = torch.autograd.grad(actual, (q, k), grad_outputs)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k), grad_outputs)
    for value, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(value, reference)


def test_npu_patch_requires_explicit_registration():
    original_vision = hf_modeling.apply_rotary_pos_emb_vision
    original_text = hf_modeling.apply_multimodal_rotary_pos_emb
    npu_patch = _load_source_module("_glm4v_npu_patch", "mindspeed_mm/fsdp/models/glm4_5v/npu_patch.py")
    assert hf_modeling.apply_rotary_pos_emb_vision is original_vision
    assert hf_modeling.apply_multimodal_rotary_pos_emb is original_text

    with (
        patch.object(hf_modeling, "apply_rotary_pos_emb_vision", original_vision),
        patch.object(hf_modeling, "apply_multimodal_rotary_pos_emb", original_text),
    ):
        npu_patch.apply_glm4v_npu_patch()
        assert hf_modeling.apply_rotary_pos_emb_vision is npu_patch.apply_rotary_pos_emb_vision
        assert hf_modeling.apply_multimodal_rotary_pos_emb is npu_patch.apply_multimodal_rotary_pos_emb


def test_multinode_launcher_requires_external_rendezvous_settings():
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert 'MASTER_ADDR=${MASTER_ADDR:?' in launcher
    assert 'NODE_RANK=${NODE_RANK:?' in launcher
    assert 'NNODES=${NNODES:-8}' in launcher
    source = 'source /usr/local/Ascend/ascend-toolkit/set_env.sh || exit $?'
    assert source in launcher
    assert launcher.index(source) < launcher.index('set -euo pipefail')
