def verify_attn_layout(name: str, layout: str):
    if name not in ALL_ATTENTION_LAYOUT:
        raise NotImplementedError(f"Unrecognized attention function: {name}")
    if layout not in ALL_ATTENTION_LAYOUT[name]:
        raise NotImplementedError(f"Unsupported layout: {layout}, {name} attention only support {ALL_ATTENTION_LAYOUT[name]}")
    return ALL_ATTENTION_LAYOUT[name]


ALL_ATTENTION_LAYOUT = {
    "eager": ["BNSD"],
    "flash_attention_2": ["BNSD"],
}