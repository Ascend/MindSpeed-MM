"""MindSpeed-MM FSDP MoE patch for Kimi-K3.

This module provides an alternative implementation of Kimi's sparse MoE block
that keeps the original routing logic, latent-MoE projection and shared-expert
behaviour, but reorganises routed-expert weights as 3-D tensors (the same layout
as Qwen3.5-MoE) so that it can be used with:

* non-EP and EP execution paths,
* NPU grouped GEMM / token permute / token unpermute kernels,
* the existing MindSpeed-MM FSDP2/EP tooling.

The patch is applied by ``apply_kimi_k3_moe_patch()`` which monkey-patches the
class reference in ``modeling_kimi``; the original source classes are left
untouched.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor

from mindspeed_mm.fsdp.distributed.parallel_state import get_parallel_state
from mindspeed_mm.fsdp.ops.moe_ops.gemm import grouped_matmul
from mindspeed_mm.fsdp.ops.moe_ops.permute import permute
from mindspeed_mm.fsdp.ops.moe_ops.unpermute import unpermute
from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

from .configuration_kimi_k3 import KimiLinearConfig
from .modeling_kimi_linear import (
    ACT2FN,
    KimiMLP,
    KimiMoEGate,
    KimiRMSNorm,
    SituAndMul,
    _get_situ_activation_params,
)


def _is_situ(config: KimiLinearConfig) -> bool:
    return config.hidden_act == "situ"


class PatchKimiMoEGate(KimiMoEGate):
    """Drop-in replacement for ``KimiMoEGate`` used by ``PatchKimiSparseMoeBlock``.

    The routing logic is intentionally identical to the original gate; this
    class exists only so that the patched MoE block can be constructed from
    explicitly named patch classes without modifying the original source.
    """
    pass


class PatchKimiMoeExperts(nn.Module):
    """Routed experts stored as 3-D tensors, compatible with GMM/permute/unpermute and EP.

    Weight layout:
        * ``gate_up_proj``: ``[num_experts, hidden_size, 2 * intermediate_size]``
        * ``down_proj``   : ``[num_experts, intermediate_size, hidden_size]``

    The module transparently loads old per-expert checkpoints that store
    separate ``w1/w2/w3`` matrices (see ``_load_from_state_dict``).
    """

    def __init__(self, config: KimiLinearConfig, hidden_size: int | None = None, intermediate_size: int | None = None):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.hidden_size = hidden_size if hidden_size is not None else config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.moe_intermediate_size

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_size)
        )

        if _is_situ(config):
            beta, linear_beta = _get_situ_activation_params(config)
            self.act_fn = SituAndMul(beta=beta, linear_beta=linear_beta)
        else:
            self.act_fn = ACT2FN[config.hidden_act]

        self.use_grouped_expert_matmul = getattr(config, "use_grouped_expert_matmul", False)
        self.fused = self.use_grouped_expert_matmul and IS_NPU_AVAILABLE

        ps = get_parallel_state()
        if ps.is_ep_enable() and not _is_situ(config):
            raise ValueError(
                f"Kimi-K3 MoE with expert parallelism only supports the 'situ' activation, "
                f"got hidden_act={config.hidden_act!r}."
            )
        self.enable_ep_balance = getattr(config, "enable_ep_balance", False) and ps.is_ep_enable()
        if self.enable_ep_balance:
            from mindspeed_mm.fsdp.distributed.expert_parallel.ep_balance.ep_balance_strategy import EPBalanceStrategy
            self.ep_balance_strategy = EPBalanceStrategy(
                ep_group=ps.get_ep_group(),
                num_experts=self.num_experts,
                max_dup_experts_num=getattr(config, "max_dup_experts_num", 2),
            )

            def hook_fn(*args, **kwargs):
                self.ep_balance_strategy.planner.pop_plan_cache()
            self.register_full_backward_hook(hook_fn)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize expert weights like the original Kimi linear layers."""
        import torch.nn.init as init
        if self.gate_up_proj.device.type == "meta":
            return
        std = getattr(self.config, "initializer_range", 0.02)
        init.normal_(self.gate_up_proj, mean=0.0, std=std)
        init.normal_(self.down_proj, mean=0.0, std=std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Non-EP routed-expert forward.

        Args:
            hidden_states: ``[num_tokens, hidden_size]``.
            top_k_index: ``[num_tokens, top_k]`` selected expert ids.
            top_k_weights: ``[num_tokens, top_k]`` routing weights.

        Returns:
            ``[num_tokens, hidden_size]`` aggregated expert outputs.
        """
        if self.fused:
            return self._forward_fused(hidden_states, top_k_index, top_k_weights)
        return self._forward_eager(hidden_states, top_k_index, top_k_weights)

    def _forward_fused(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:

        permuted_hidden_states, row_ids_map = permute(
            hidden_states, top_k_index.to(torch.int32), fused=True
        )
        tokens_per_expert = torch.histc(
            top_k_index, bins=self.num_experts, min=0, max=self.num_experts
        )
        intermediate = grouped_matmul(
            permuted_hidden_states, self.gate_up_proj, tokens_per_expert, fused=True
        )
        if _is_situ(self.config):
            intermediate_activations = self.act_fn(intermediate)
        else:
            gate, up = intermediate.chunk(2, dim=-1)
            intermediate_activations = self.act_fn(gate) * up
        output = grouped_matmul(
            intermediate_activations, self.down_proj, tokens_per_expert, fused=True
        )
        return unpermute(output, row_ids_map, probs=top_k_weights, fused=True)

    def _forward_eager(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # Transpose weights to F.linear layout: [num_experts, out_dim, in_dim].
        gate_up_proj = self.gate_up_proj.permute(0, 2, 1).contiguous()
        down_proj = self.down_proj.permute(0, 2, 1).contiguous()

        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(
                top_k_index, num_classes=self.num_experts
            ).permute(2, 1, 0)
            expert_hit = torch.greater(
                expert_mask.sum(dim=(-1, -2)), 0
            ).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx.item()
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = F.linear(current_state, gate_up_proj[expert_idx])
            if _is_situ(self.config):
                current_hidden_states = self.act_fn(gate_up)
            else:
                gate, up = gate_up.chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[
                token_idx, top_k_pos, None
            ].to(current_hidden_states.dtype)
            final_hidden_states.index_add_(
                0,
                token_idx,
                current_hidden_states.to(final_hidden_states.dtype),
            )

        return final_hidden_states

    def ep_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
        ep_group: Any,
        ep_plan: Any,
    ) -> torch.Tensor:
        """EP routed-expert forward.

        This method is discovered by ``expert_parallelize_modules`` and bound to
        ``forward`` when expert parallelism is enabled.  It reuses the generic
        EP dispatchers (``alltoall`` / ``mc2`` / ``allgather``) and passes the
        configured activation (including ``situ``) down to the dispatcher.
        """
        gate_up_proj = (
            self.gate_up_proj.to_local()
            if isinstance(self.gate_up_proj, DTensor)
            else self.gate_up_proj
        )
        down_proj = (
            self.down_proj.to_local()
            if isinstance(self.down_proj, DTensor)
            else self.down_proj
        )

        from mindspeed_mm.fsdp.distributed.expert_parallel.ep_dispatcher import (
            ep_allgather_forward,
            ep_forward,
            ep_mc2_forward,
        )

        ep_dispatcher_dict = {
            "alltoall": ep_forward,
            "mc2": ep_mc2_forward,
            "allgather": ep_allgather_forward,
        }

        if self.enable_ep_balance:
            self.ep_balance_strategy.executor.register_backward_dup_experts_grad_acc_hook(gate_up_proj, name="fc1")
            self.ep_balance_strategy.executor.register_backward_dup_experts_grad_acc_hook(down_proj, name="fc2")

            if ep_plan.dispatcher in ["mc2", "allgather"]:
                raise NotImplementedError("EP load balancing strategy currently only supports alltoall dispatch.")

        if ep_plan.dispatcher not in ep_dispatcher_dict:
            raise NotImplementedError(
                f"EP dispatcher {ep_plan.dispatcher} is not implemented for Kimi-K3 MoE."
            )

        dispatcher_func = ep_dispatcher_dict[ep_plan.dispatcher]
        hidden_states = dispatcher_func(
            self.num_experts,
            top_k_weights,
            top_k_index,
            hidden_states,
            fc1_weight=gate_up_proj,
            fc2_weight=down_proj,
            ep_group=ep_group,
            fused=ep_plan.use_npu_fused_ops,
            ep_balance_strategy=self.ep_balance_strategy if self.enable_ep_balance else None,
            activation=self.act_fn,
        )

        return hidden_states


class PatchKimiSparseMoeBlock(nn.Module):
    """Patched Kimi sparse MoE block with 3-D expert tensors, GMM and EP support.

    Behavioural invariants compared with the original ``KimiSparseMoeBlock``:

    * Same gating logic (via ``PatchKimiMoEGate``).
    * Same latent-MoE down/up projection and optional RMSNorm.
    * Same shared-expert residual addition.
    * Same output shape and dtype semantics.

    The only structural change is that routed experts are stored as 3-D tensors
    inside ``PatchKimiMoeExperts`` instead of a ``ModuleList`` of per-expert MLPs.
    """

    def __init__(self, config: KimiLinearConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_token
        self.moe_renormalize = config.moe_renormalize

        self.use_latent_moe = getattr(config, "routed_expert_hidden_size", None) is not None
        self.moe_hidden_size = (
            config.routed_expert_hidden_size
            if self.use_latent_moe else config.hidden_size
        )
        self.latent_moe_use_norm = getattr(config, "latent_moe_use_norm", False)

        self.experts = PatchKimiMoeExperts(
            config,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
        )
        self.gate = PatchKimiMoEGate(config)

        if config.num_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.num_shared_experts
            self.shared_experts = KimiMLP(
                config=config, intermediate_size=intermediate_size
            )

        if self.use_latent_moe:
            self.routed_expert_down_proj = nn.Linear(
                config.hidden_size, self.moe_hidden_size, bias=False
            )
            self.routed_expert_up_proj = nn.Linear(
                self.moe_hidden_size, config.hidden_size, bias=False
            )
            if self.latent_moe_use_norm:
                self.routed_expert_norm = KimiRMSNorm(
                    self.moe_hidden_size, eps=config.rms_norm_eps
                )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        if self.use_latent_moe:
            hidden_states = self.routed_expert_down_proj(hidden_states)

        y = self.experts(hidden_states, topk_idx, topk_weight)

        if self.use_latent_moe:
            if self.latent_moe_use_norm:
                y = self.routed_expert_norm(y)
            y = self.routed_expert_up_proj(y)

        y = y.view(*orig_shape)

        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y


# -----------------------------------------------------------------------------
# Patch application helpers
# -----------------------------------------------------------------------------

def apply_kimi_k3_moe_patch():
    """Replace Kimi sparse MoE classes with their patched counterparts.

    Must be called after ``modeling_kimi`` has been imported (so the original
    class objects exist) but before the Kimi model is instantiated.
    """
    from . import modeling_kimi_linear

    modeling_kimi_linear.KimiSparseMoeBlock = PatchKimiSparseMoeBlock
    modeling_kimi_linear.KimiMoEGate = PatchKimiMoEGate
