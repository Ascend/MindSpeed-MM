"""CPU regression tests for the BSC cached convolution fallback.

Only the repository-owned fallback definition is loaded: importing the full
model module also imports accelerator/distributed dependencies unrelated to
this pure PyTorch helper. This does not test model or kernel dispatch.
"""

import ast
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


MODEL_FILES = (
    "qwen3_5/modeling_qwen3_5.py",
    "qwen3_5_moe/modeling_qwen3_5_moe.py",
)


@pytest.fixture(params=MODEL_FILES, ids=("dense", "moe"))
def cached_update(request):
    root = Path(__file__).resolve().parents[4]
    source = root / "mindspeed_mm/fsdp/models" / request.param
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "torch_causal_conv1d_update"
    )
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["torch_causal_conv1d_update"]


def _inputs(seq_len, dtype, with_bias):
    generator = torch.Generator().manual_seed(42)
    # Different sequence/channel/kernel sizes expose accidental layout swaps.
    tokens = torch.randn(2, seq_len, 5, generator=generator, dtype=dtype)
    state = torch.randn(2, 5, 4, generator=generator, dtype=dtype)
    weight = torch.randn(5, 4, generator=generator, dtype=dtype)
    bias = torch.randn(5, generator=generator, dtype=dtype) if with_bias else None
    return tokens, state, weight, bias


def _reference(tokens, state, weight, bias):
    history = torch.cat((state, tokens.transpose(1, 2)), dim=-1)
    # Explicit sliding-window depthwise convolution, independent of F.conv1d.
    windows = history.unfold(-1, weight.shape[-1], 1)[:, :, -tokens.shape[1]:, :]
    output = (windows * weight[None, :, None, :]).sum(dim=-1)
    if bias is not None:
        output = output + bias[None, :, None]
    return F.silu(output).transpose(1, 2), history[:, :, -state.shape[-1]:]


@pytest.mark.parametrize("seq_len", (1, 3, 7))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("with_bias", (False, True))
def test_cached_update_layout_and_state(cached_update, seq_len, dtype, with_bias):
    tokens, state, weight, bias = _inputs(seq_len, dtype, with_bias)
    expected, expected_state = _reference(tokens, state.clone(), weight, bias)
    state_storage = state.data_ptr()

    actual = cached_update(tokens, state, weight, bias, activation="silu")

    assert actual.shape == tokens.shape
    assert actual.dtype == tokens.dtype
    assert actual.is_contiguous()
    assert state.data_ptr() == state_storage
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)


@pytest.mark.parametrize("with_bias", (False, True))
def test_cached_update_chunked_decode(cached_update, with_bias):
    tokens, state, weight, bias = _inputs(10, torch.float32, with_bias)
    expected, expected_state = _reference(tokens, state.clone(), weight, bias)
    outputs = []
    offset = 0
    for chunk_len in (1, 3, 6):
        # Slices are non-contiguous for batch > 1, as in streaming callers.
        chunk = tokens[:, offset:offset + chunk_len, :]
        outputs.append(cached_update(chunk, state, weight, bias, activation="silu"))
        offset += chunk_len

    torch.testing.assert_close(torch.cat(outputs, dim=1), expected)
    torch.testing.assert_close(state, expected_state)
