def _initialize_npu_plugin():

    from verl_npu.plugin import apply_npu_plugin
    apply_npu_plugin()

# Initialize on module import
_initialize_npu_plugin()
