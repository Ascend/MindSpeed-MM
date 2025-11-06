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
        from mindspeed.core.transformer.flash_attention.generate_mask.adaptor import dot_product_attention_forward_wrapper
        from mindspeed.core.transformer.flash_attention.flash_attention.adaptor import \
            dot_product_attention_forward_impl
        from mindspeed_mm.models.common.module_spec import qwen2_5omni_layer_spec, qwen2vl_layer_spec
        from mindspeed.core.context_parallel.adaptor import MindSpeedCPDotProductAttention
        
        if 'megatron.core.transformer.dot_product_attention.DotProductAttention' in mspm.patches_info:
            DotProductAttention = mspm.patches_info['megatron.core.transformer.dot_product_attention.DotProductAttention'].orig_func
        else:
            from megatron.core.transformer.dot_product_attention import DotProductAttention
        orig_DotProductAttention = DotProductAttention
        from mindspeed.core.context_parallel.adaptor import attention_init_wrapper

        attention_init_wrapper.__globals__['UlyssesContextAttention'] = mm_UlyssesContextAttention

        if 'vit' in spec.__name__:
            globals = qwen2vl_layer_spec.get_qwen2vl_layer_spec.__globals__
        if 'audio' in spec.__name__:
            globals = qwen2_5omni_layer_spec.get_qwen_omni_audio_layer_spec.__globals__
        if 'llm' in spec.__name__:
            globals = qwen2vl_layer_spec.get_qwen2vl_llm_layer_spec.__globals__
        
        if mpu.get_context_parallel_world_size() > 1:
            globals['DotProductAttention'] = MindSpeedCPDotProductAttention
        else:
            globals['DotProductAttention'] = orig_DotProductAttention
            globals['DotProductAttention'].forward = dot_product_attention_forward_impl
            globals['DotProductAttention'].forward = dot_product_attention_forward_wrapper(globals['DotProductAttention'].forward)
   
        return spec(*args, **kwargs)
    return wrapper