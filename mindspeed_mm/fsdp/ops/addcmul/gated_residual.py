"""
Gated residual injection operations based on addcmul operations.
"""

from mindspeed_mm.fsdp.utils.device import IS_NPU_AVAILABLE

def eager_residual_injection(hidden_states, injection_weights, hyper_input):
    injection = hidden_states.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)

def fused_residual_injection(hidden_states, injection_weights, hyper_input):
    from .addcmul_triton import triton_addcmul
    return triton_addcmul(hidden_states, injection_weights, hyper_input, round_product=True)


def gated_residual_injection(hidden_states, injection_weights, hyper_input, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_residual_injection(hidden_states, injection_weights, hyper_input)
    else:
        return eager_residual_injection(hidden_states, injection_weights, hyper_input)
