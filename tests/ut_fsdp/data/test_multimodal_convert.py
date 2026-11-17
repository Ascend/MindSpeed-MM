# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Tests for multimodal Agentic SFT data handling."""

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import (
    DatasetAttr,
    MultiModalToolDatasetConverter,
)
from mindspeed_mm.fsdp.data.data_utils.func_utils.template import TEMPLATES


class _StubDataArgs:
    dataset_dir = "."


def _messages():
    return [
        {"role": "user", "content": "u"},
        {"role": "tool_call", "content": "c1"},
        {"role": "tool_response", "content": "r1"},
        {"role": "tool_call", "content": "c2"},
        {"role": "tool_response", "content": "r2"},
        {"role": "assistant", "content": "a"},
    ]


def _converter(tools=None):
    attr = DatasetAttr(
        messages="messages",
        images="images",
        tools=tools,
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool_response",
        system_tag="system",
        formatting="multimodal_tool",
    )
    return MultiModalToolDatasetConverter(dataset_attr=attr, data_args=_StubDataArgs())


def test_qwen3_omni_tool_prompt():
    for name in ("qwen3_omni", "qwen3_omni_nothink"):
        assert TEMPLATES[name].format_tool is not None
        assert "<tools>" in TEMPLATES[name].format_tool.apply(content="{}")[0]


def test_qwen3_vl_tool_prompt():
    for name in ("qwen3_vl", "qwen3_vl_nothink"):
        assert TEMPLATES[name].format_tool is not None
        assert "<tools>" in TEMPLATES[name].format_tool.apply(content="{}")[0]
