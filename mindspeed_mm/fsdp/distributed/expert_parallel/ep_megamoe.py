# Copyright 2026 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A2/A3 MegaMoE adapter for the common EP dispatcher weight layout."""

import weakref

import torch


# Executors own no parameters; each call supplies the current materialized weights.
_MEGAMOE_EXECUTORS = weakref.WeakValueDictionary()


def ep_megamoe_forward(
    num_experts,
    routing_weights,
    selected_experts,
    hidden_states,
    fc1_weight,
    fc2_weight,
    ep_group=None,
    fused=True,
    swiglu_limit=None,
    swiglu_alpha=None,
    ep_balance_strategy=None,
    skip_moe_pad_tokens=False,
    seq_mask=None,
    activation=None,
    *,
    owner,
):
    """Execute standard SwiGLU experts with [E, in, out] dispatcher weights.

    The owning expert module retains the parameterless executor without registering
    it as a child module. Weight views remain in autograd and are never cached.
    """
    if hidden_states.device.type != "npu":
        raise ValueError("megamoe dispatcher requires NPU tensors")
    if not fused:
        raise ValueError("megamoe dispatcher requires use_npu_fused_ops=True")
    if ep_balance_strategy is not None or skip_moe_pad_tokens:
        raise NotImplementedError("megamoe dispatcher does not support expert duplication or padding removal")
    if activation is not None or swiglu_alpha is not None or swiglu_limit not in (None, 0):
        raise NotImplementedError("megamoe dispatcher supports standard SwiGLU only")
    if ep_group is None:
        raise ValueError("megamoe dispatcher requires an explicit EP process group")
    ep_size = torch.distributed.get_world_size(ep_group)
    if ep_size <= 1 or num_experts <= 0 or num_experts % ep_size:
        raise ValueError("megamoe requires EP size > 1 dividing the number of experts")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("megamoe dispatcher currently supports BF16 only")
    if any(w.device != hidden_states.device or w.dtype != hidden_states.dtype
           for w in (fc1_weight, fc2_weight)):
        raise ValueError("megamoe weights must share the input device and BF16 dtype")
    if hidden_states.ndim != 2 or fc1_weight.ndim != 3 or fc2_weight.ndim != 3:
        raise ValueError("megamoe expects [tokens, hidden] inputs and 3D expert weights")
    hidden_size = hidden_states.shape[-1]
    local_experts = num_experts // ep_size
    intermediate_size = fc2_weight.shape[1]
    if (tuple(fc1_weight.shape) != (local_experts, hidden_size, 2 * intermediate_size)
            or tuple(fc2_weight.shape) != (local_experts, intermediate_size, hidden_size)):
        raise ValueError("megamoe expects EP-local [E, in, out] SwiGLU weights")
    if (selected_experts.ndim != 2 or routing_weights.ndim != 2
            or selected_experts.shape[0] != hidden_states.shape[0]
            or routing_weights.shape[0] != hidden_states.shape[0]
            or not 0 < selected_experts.shape[1] <= num_experts):
        raise ValueError("megamoe expects matching token counts and a valid top-k routing shape")
    if selected_experts.device != hidden_states.device or routing_weights.device != hidden_states.device:
        raise ValueError("megamoe routing tensors must be on the input device")
    if selected_experts.dtype not in (torch.int32, torch.int64):
        raise ValueError("megamoe expert indices must be int32 or int64")

    if routing_weights.shape != selected_experts.shape:
        if routing_weights.shape[1] != num_experts:
            raise ValueError("megamoe routing weights must contain top-k or all expert probabilities")
        routing_weights = routing_weights.gather(1, selected_experts.long())

    try:
        from hyper_multicore import MegaMoeExperts, MegaMoeNumericsOptions
    except ImportError as exc:
        raise ImportError("MegaMoE requires hyper_multicore with parameterless expert_weights support") from exc

    key = (id(ep_group), hidden_states.device, hidden_states.dtype, hidden_size,
           intermediate_size, num_experts, selected_experts.shape[-1])
    executor = _MEGAMOE_EXECUTORS.get(key)
    if executor is None:
        executor = MegaMoeExperts(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            num_experts=num_experts, top_k=selected_experts.shape[-1], ep_group=ep_group,
            precision="bf16",
            numerics_options=MegaMoeNumericsOptions(apply_probs_before_fc2=False),
        )
        _MEGAMOE_EXECUTORS[key] = executor
        print(f"[QWEN_MEGAMOE_ACTIVE] dispatcher=megamoe ep={ep_size} "
              f"tokens={hidden_states.shape[0]} weight_layout=EOI", flush=True)
    object.__setattr__(owner, "_megamoe_executor", executor)
    # Qwen4Exp stores EOI weights, so these transposes undo its dispatcher views.
    # Other Qwen experts use EIO storage; contiguous() adapts those without detach.
    weights = (fc1_weight.transpose(1, 2).contiguous(), fc2_weight.transpose(1, 2).contiguous())
    return executor(hidden_states, selected_experts.contiguous(), routing_weights.contiguous(),
                    expert_weights=weights)
