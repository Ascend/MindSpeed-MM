import inspect
from functools import wraps
import megatron.core.parallel_state as mpu
from mindspeed.patch_utils import MindSpeedPatchesManager as mspm


def hetero_attention_init_wrapper(fn):
    from mindspeed.core.context_parallel.adaptor import attention_init_wrapper
    
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn_orig = inspect.unwrap(fn)
        if mpu.get_context_parallel_world_size() > 1:
            return attention_init_wrapper(fn_orig)(*args, **kwargs)
        else:
            return fn_orig(*args, **kwargs)
    return wrapper

    
def hetero_spec_wrapper(spec):
    @wraps(spec)
    def wrapper(*args, **kwargs):
        from mindspeed_mm.patchs.ulysses_patches import UlyssesContextAttention as mm_UlyssesContextAttention
        mspm.register_patch('mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel.UlyssesContextAttention', \
                                mm_UlyssesContextAttention, force_patch=True)
        if mpu.get_context_parallel_world_size() > 1:
            from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
            mspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                MindSpeedCPDotProductAttention, force_patch=True)
        else:
            from megatron.core.transformer.dot_product_attention import DotProductAttention
            mspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                    DotProductAttention, force_patch=True)
            from mindspeed.core.transformer.flash_attention.generate_mask.adaptor import dot_product_attention_forward_wrapper
            mspm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward', 
                                dot_product_attention_forward_wrapper, force_patch=True)
        mspm.apply_patches()    
        return spec(*args, **kwargs)
    return wrapper