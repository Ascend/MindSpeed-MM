import torch


_AUX_LOSS_TYPE_LOCAL = "local"
_AUX_LOSS_TYPE_GLOBAL = "global"
_GLOBAL_AUX_LOSS_TRACKER = {}


def _get_group_size(group) -> int:
    if (
        group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
    ):
        return 1
    return torch.distributed.get_world_size(group=group)


def _scale_gradient(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Scale the backward gradient without changing the forward value."""
    detached = tensor.detach()
    return detached + scale * (tensor - detached)


def reset_global_aux_loss_tracker(tracker_key: str | None = None) -> None:
    """Reset accumulated expert counts at an optimizer-step boundary."""
    if tracker_key is None:
        _GLOBAL_AUX_LOSS_TRACKER.clear()
    else:
        _GLOBAL_AUX_LOSS_TRACKER.pop(tracker_key, None)


def _update_global_tokens_per_expert(
        tracker_key: str,
        global_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    tracker = _GLOBAL_AUX_LOSS_TRACKER.get(tracker_key)
    if (
        tracker is None
        or tracker["tokens_per_expert"].shape != global_tokens_per_expert.shape
        or tracker["tokens_per_expert"].device != global_tokens_per_expert.device
        or tracker["tokens_per_expert"].dtype != global_tokens_per_expert.dtype
    ):
        tracker = {
            "tokens_per_expert": torch.zeros_like(global_tokens_per_expert),
            "steps": 0,
        }
        _GLOBAL_AUX_LOSS_TRACKER[tracker_key] = tracker

    tracker["tokens_per_expert"] += global_tokens_per_expert.detach()
    tracker["steps"] += 1
    return tracker["tokens_per_expert"] / tracker["steps"]


def _load_balancing_loss_func_local(
        gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
        num_experts: int | None = None,
        top_k: int = 2,
        attention_mask: torch.Tensor | None = None,
        context_parallel_group=None,
) -> torch.Tensor | int:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    num_layers = len(gate_logits)
    if num_layers == 0:
        return 0

    compute_device = gate_logits[0].device

    tokens_selected = torch.zeros(top_k, num_experts, device=compute_device)

    if attention_mask is None:
        tokens_total = torch.zeros(top_k, num_experts, device=compute_device)
    else:
        tokens_total = torch.zeros(top_k, num_experts, device=compute_device, dtype=attention_mask.dtype)

    expert_attention_mask = None

    for layer_gate in gate_logits:
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)  # [batch*seq_len, num_experts]
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)  # [batch*seq_len, top_k]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)  # [batch*seq_len, top_k, num_experts]

        if attention_mask is None:
            num_tokens = layer_gate.shape[0]  # batch_size * sequence_length
            if expert_attention_mask is None or expert_attention_mask.shape[0] != num_tokens:
                expert_attention_mask = torch.ones(
                    num_tokens, top_k, num_experts,
                    device=compute_device, dtype=torch.float32
                ).reshape(-1, top_k, num_experts)
                layer_tokens_total = torch.sum(expert_attention_mask, dim=0)

            layer_tokens_selected = torch.sum(expert_mask.float(), dim=0)
        else:
            batch_size, sequence_length = attention_mask.shape
            if expert_attention_mask is None:
                expert_attention_mask = (
                    attention_mask[None, :, :, None, None]
                    .expand((1, batch_size, sequence_length, top_k, num_experts))
                    .reshape(-1, top_k, num_experts)
                    .to(compute_device)
                )
                layer_tokens_total = torch.sum(expert_attention_mask, dim=0)

            layer_tokens_selected = torch.sum(expert_mask.float() * expert_attention_mask, dim=0)

        tokens_selected += layer_tokens_selected
        tokens_total += layer_tokens_total

    if context_parallel_group is not None and torch.distributed.get_world_size(group=context_parallel_group) > 1:
        torch.distributed.all_reduce(
            tokens_total,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

        torch.distributed.all_reduce(
            tokens_selected,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

    tokens_per_expert = tokens_selected / tokens_total

    # calculate router_prob_per_expert
    compute_device = gate_logits[0].device
    routing_weights = torch.cat([torch.nn.functional.softmax(layer_gate, dim=-1).to(compute_device) for layer_gate in gate_logits], dim=0)

    if attention_mask is not None:
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_selected = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0)
        router_total = torch.sum(router_per_expert_attention_mask, dim=0)
    else:
        num_tokens = gate_logits[0].shape[0]  # batch_size * sequence_length
        router_per_expert_attention_mask = torch.ones(
            num_layers, num_tokens, num_experts,
            device=compute_device, dtype=torch.float32
        ).reshape(-1, num_experts)
        router_selected = torch.sum(routing_weights, dim=0)
        router_total = torch.sum(router_per_expert_attention_mask, dim=0)

    if context_parallel_group is not None and torch.distributed.get_world_size(group=context_parallel_group) > 1:
        torch.distributed.all_reduce(
            router_selected,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

        torch.distributed.all_reduce(
            router_total,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group
        )

    router_prob_per_expert = router_selected / router_total

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def _load_balancing_loss_func_global(
        gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
        num_experts: int | None = None,
        top_k: int = 2,
        attention_mask: torch.Tensor | None = None,
        context_parallel_group=None,
        global_aux_loss_group=None,
        data_parallel_group=None,
        global_step_num_tokens: torch.Tensor | None = None,
        gradient_accumulation_steps: torch.Tensor | int | None = None,
        router_aux_loss_use_attention_mask: bool = False,
        tracker_key: str = "default",
) -> torch.Tensor | int:
    """Compute Megatron-style global router load-balancing loss.

    Reference: NVIDIA/Megatron-LM, ``Router._apply_global_aux_loss`` and
    ``switch_load_balancing_loss_func``:
    https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/router.py
    Training requires per-token LM loss normalization and router logits that
    retain their autograd graph.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    num_layers = len(gate_logits)
    if num_layers == 0:
        return 0

    if torch.is_grad_enabled() and not all(layer_gate.requires_grad for layer_gate in gate_logits):
        raise RuntimeError(
            "Global router aux loss requires captured router logits with an autograd graph. "
            "Set features.recompute_plan.use_reentrant to false."
        )

    compute_device = gate_logits[0].device
    local_tokens_per_expert = torch.zeros(
        num_layers, num_experts, device=compute_device, dtype=torch.float32
    )
    local_num_tokens = torch.zeros(num_layers, device=compute_device, dtype=torch.float32)
    local_router_probability_sum = torch.zeros(
        num_layers, num_experts, device=compute_device, dtype=torch.float32
    )

    valid_token_mask = None
    if router_aux_loss_use_attention_mask:
        if attention_mask is None:
            raise ValueError(
                "Global router aux loss requires attention_mask when "
                "router_aux_loss_use_attention_mask is enabled."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                "Global router aux loss requires a 2D [batch, sequence] attention_mask "
                "when router_aux_loss_use_attention_mask is enabled."
            )
        expected_num_tokens = gate_logits[0].shape[0]
        if attention_mask.numel() != expected_num_tokens:
            raise ValueError(
                "Global router aux loss attention_mask token count must match router logits: "
                f"mask={attention_mask.numel()}, router={expected_num_tokens}."
            )
        if any(layer_gate.shape[0] != expected_num_tokens for layer_gate in gate_logits):
            raise ValueError("All global router aux loss layers must have the same token count.")
        valid_token_mask = attention_mask.reshape(-1).to(compute_device).bool()

    for layer_idx, layer_gate in enumerate(gate_logits):
        routing_weights = torch.nn.functional.softmax(layer_gate, dim=-1)
        _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts).float()

        if valid_token_mask is None:
            local_tokens_per_expert[layer_idx] = torch.sum(expert_mask, dim=(0, 1))
            local_num_tokens[layer_idx] = layer_gate.new_tensor(
                layer_gate.shape[0], dtype=torch.float32
            )
            local_router_probability_sum[layer_idx] = torch.sum(
                routing_weights.float(), dim=0
            )
        else:
            valid_tokens = valid_token_mask.float()
            local_tokens_per_expert[layer_idx] = torch.sum(
                expert_mask * valid_tokens[:, None, None], dim=(0, 1)
            )
            local_num_tokens[layer_idx] = torch.sum(valid_tokens)
            local_router_probability_sum[layer_idx] = torch.sum(
                routing_weights.float() * valid_tokens[:, None], dim=0
            )

    global_aux_loss_group_size = _get_group_size(global_aux_loss_group)
    global_tokens_per_expert = local_tokens_per_expert.detach().clone()
    if global_aux_loss_group_size > 1:
        torch.distributed.all_reduce(
            global_tokens_per_expert,
            op=torch.distributed.ReduceOp.SUM,
            group=global_aux_loss_group,
        )

    averaged_tokens_per_expert = _update_global_tokens_per_expert(
        tracker_key, global_tokens_per_expert
    )

    context_parallel_group_size = _get_group_size(context_parallel_group)
    if global_aux_loss_group_size % context_parallel_group_size != 0:
        raise ValueError(
            "Global aux group size must be divisible by context parallel group size: "
            f"global={global_aux_loss_group_size}, cp={context_parallel_group_size}."
        )

    cp_sample_num_tokens = local_num_tokens.detach().clone()
    if context_parallel_group_size > 1:
        torch.distributed.all_reduce(
            cp_sample_num_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=context_parallel_group,
        )

    logical_data_parallel_size = global_aux_loss_group_size // context_parallel_group_size
    if router_aux_loss_use_attention_mask:
        denominator_num_tokens = global_tokens_per_expert.sum(dim=1) / top_k
    else:
        denominator_num_tokens = cp_sample_num_tokens * logical_data_parallel_size
    denominator = torch.clamp(
        top_k * denominator_num_tokens * denominator_num_tokens,
        min=1.0,
    )
    aux_loss_per_layer = torch.sum(
        local_router_probability_sum * averaged_tokens_per_expert, dim=1
    ) * (num_experts / denominator)

    if torch.is_grad_enabled():
        if not isinstance(global_step_num_tokens, torch.Tensor):
            raise ValueError(
                "Global router aux loss requires global_step_num_tokens from "
                "PrefetchGradAccDataLoader; set features.loss_cfg.loss_type to per_token_loss."
            )
        if gradient_accumulation_steps is None:
            raise ValueError(
                "Global router aux loss requires gradient_accumulation_steps from "
                "PrefetchGradAccDataLoader."
            )

        global_label_tokens = global_step_num_tokens.detach().float().clone()
        if _get_group_size(data_parallel_group) > 1:
            torch.distributed.all_reduce(
                global_label_tokens,
                op=torch.distributed.ReduceOp.SUM,
                group=data_parallel_group,
            )

        accumulation_steps = torch.as_tensor(
            gradient_accumulation_steps,
            dtype=torch.float32,
            device=compute_device,
        )
        backward_scale = accumulation_steps * cp_sample_num_tokens / torch.clamp(
            global_label_tokens, min=1.0
        )
        aux_loss_per_layer = _scale_gradient(aux_loss_per_layer, backward_scale)

    aux_loss = torch.sum(aux_loss_per_layer)

    # FSDP/HSDP averages gradients after backward. Compensate so the effective
    # aux gradient is the sum of the rank-local contributions.
    return aux_loss * global_aux_loss_group_size


def load_balancing_loss_func_optimized(
        gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
        num_experts: int | None = None,
        top_k: int = 2,
        attention_mask: torch.Tensor | None = None,
        context_parallel_group=None,
        aux_loss_type: str = _AUX_LOSS_TYPE_LOCAL,
        global_aux_loss_group=None,
        data_parallel_group=None,
        global_step_num_tokens: torch.Tensor | None = None,
        gradient_accumulation_steps: torch.Tensor | int | None = None,
        router_aux_loss_use_attention_mask: bool = False,
        tracker_key: str = "default",
) -> torch.Tensor | int:
    if aux_loss_type == _AUX_LOSS_TYPE_LOCAL:
        return _load_balancing_loss_func_local(
            gate_logits,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask,
            context_parallel_group=context_parallel_group,
        )
    if aux_loss_type == _AUX_LOSS_TYPE_GLOBAL:
        return _load_balancing_loss_func_global(
            gate_logits,
            num_experts=num_experts,
            top_k=top_k,
            attention_mask=attention_mask,
            context_parallel_group=context_parallel_group,
            global_aux_loss_group=global_aux_loss_group,
            data_parallel_group=data_parallel_group,
            global_step_num_tokens=global_step_num_tokens,
            gradient_accumulation_steps=gradient_accumulation_steps,
            router_aux_loss_use_attention_mask=router_aux_loss_use_attention_mask,
            tracker_key=tracker_key,
        )
    raise ValueError(
        f"Invalid aux_loss_type='{aux_loss_type}'. Must be one of: local, global."
    )
