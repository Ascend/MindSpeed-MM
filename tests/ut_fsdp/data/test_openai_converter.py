# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Unit tests for the OpenAI agentical-trace dataset converter and Qwen3.6 templates.

Covers ``mindspeed_mm.fsdp.data.data_utils.func_utils.convert.OpenAIDatasetConverter``
(flattening OpenAI ChatCompletion messages into Role-tagged turns) and the
``qwen3_6`` / ``qwen3_6_nothink`` template registrations.
"""

from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import (
    OpenAIDatasetConverter,
    DatasetAttr,
    DATASET_CONVERTERS,
    Role,
)
from mindspeed_mm.fsdp.data.data_utils.func_utils.template import TEMPLATES



class _StubDataArgs:
    """Minimal data_args stand-in.

    OpenAIDatasetConverter only touches data_args via ``_find_media_files``,
    which is never reached for the text-only samples exercised here.
    """

    dataset_dir = "."


def _make_converter():
    """Build a converter wired to OpenAI-style column / tag names."""
    attr = DatasetAttr(
        messages="messages",
        tools="tools",
        role_tag="role",
        content_tag="content",
        user_tag="user",
        assistant_tag="assistant",
        observation_tag="tool",
        system_tag="system",
    )
    return OpenAIDatasetConverter(dataset_attr=attr, data_args=_StubDataArgs())


class TestOpenAIConverterRegistration:
    def test_registered_under_openai_key(self):
        assert DATASET_CONVERTERS.get("openai") is OpenAIDatasetConverter

    def test_formatting_literal_accepts_openai(self):
        # Constructing with formatting="openai" must not raise (Literal accepts it).
        attr = DatasetAttr(formatting="openai")
        assert attr.formatting == "openai"

    def test_tools_attr_defaults_to_none(self):
        assert DatasetAttr().tools is None


class TestOpenAIConverterBasic:
    def test_system_and_simple_turn(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]})
        assert out["_system"] == "You are helpful."
        assert out["_prompt"] == [{"role": Role.USER.value, "content": "hi"}]
        assert out["_response"] == [{"role": Role.ASSISTANT.value, "content": "hello"}]

    def test_content_is_trimmed(self):
        # Match the inference chat_template's |trim: leading/trailing whitespace
        # on every message content must be stripped.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "  hi\n\n"},
            {"role": "assistant", "content": "\nhello  "},
        ]})
        assert out["_prompt"][0]["content"] == "hi"
        assert out["_response"][0]["content"] == "hello"

    def test_no_system_message(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]})
        assert out["_system"] == ""

    def test_leading_system_is_trimmed(self):
        # The leading system message content is trimmed, consistent with the
        # per-message trim policy and SharegptConverter.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "system", "content": "  You are helpful.  \n"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]})
        assert out["_system"] == "You are helpful."


class TestOpenAIConverterToolCalls:
    def test_tool_call_serialized_as_qwen36_xml(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "SH"}}}
            ]},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]})
        tc_turn = out["_prompt"][1]
        assert tc_turn["role"] == Role.TOOL_CALL.value
        assert tc_turn["content"] == (
            "<tool_call>\n<function=get_weather>\n"
            "<parameter=city>\nSH\n</parameter>\n"
            "</function>\n</tool_call>"
        )

    def test_tool_response_role(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "SH"}}}
            ]},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ]})
        assert out["_prompt"][2]["role"] == Role.TOOL_RESPONSE.value
        assert out["_prompt"][2]["content"] == "sunny"

    def test_tool_call_arguments_as_json_string(self):
        # arguments may arrive as a JSON string instead of a dict.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "get_weather", "arguments": '{"city": "SH"}'}}
            ]},
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "ok"},
        ]})
        assert "<parameter=city>\nSH\n</parameter>" in out["_prompt"][1]["content"]

    def test_consecutive_tool_responses_merged(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "content": "r1"},
            {"role": "tool", "content": "r2"},
            {"role": "assistant", "content": "done"},
        ]})
        resp_turn = out["_prompt"][2]
        assert resp_turn["role"] == Role.TOOL_RESPONSE.value
        # The two tool outputs are joined by the tool-response separator.
        assert "r1" in resp_turn["content"] and "r2" in resp_turn["content"]
        assert conv.TOOL_RESP_SEP in resp_turn["content"]


class TestOpenAIConverterReasoning:
    def test_reasoning_content_merged_as_think_block(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans", "reasoning_content": "think"},
        ]})
        assert out["_response"][0]["content"] == "<think>\nthink\n</think>\n\nans"

    def test_no_reasoning_leaves_content_unchanged(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "ans"},
        ]})
        assert out["_response"][0]["content"] == "ans"


class TestOpenAIConverterUserInterjection:
    def test_midtrajectory_user_folded_into_tool_response(self):
        # A user message following tool outputs (e.g. a system reminder) must be
        # folded into the pending tool-response block rather than breaking
        # alternation and dropping the whole sample.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "f", "arguments": {}}}
            ]},
            {"role": "tool", "content": "tool out"},
            {"role": "user", "content": "extra reminder"},
            {"role": "assistant", "content": "final"},
        ]})
        # Sample survives (non-empty) and ends on the assistant turn.
        assert out["_response"] == [{"role": Role.ASSISTANT.value, "content": "final"}]
        assert "extra reminder" in out["_prompt"][2]["content"]


class TestOpenAIConverterTools:
    def test_tools_column_rendered_as_list_of_str(self):
        conv = _make_converter()
        out = conv({
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "tools": [{"name": "f", "description": "d"}],
        })
        assert isinstance(out["_tools"], list)
        assert len(out["_tools"]) == 1
        assert isinstance(out["_tools"][0], str)

    def test_missing_tools_column_yields_none(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]})
        assert out["_tools"] is None


class TestOpenAIConverterBrokenSamples:
    def test_trailing_user_turn_dropped(self):
        # Odd turn count not ending in assistant -> dropped (empty prompt/response).
        conv = _make_converter()
        out = conv({"messages": [{"role": "user", "content": "only user"}]})
        assert out["_prompt"] == []
        assert out["_response"] == []

    def test_unknown_role_dropped(self):
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "bogus", "content": "x"},
        ]})
        assert out["_prompt"] == []
        assert out["_response"] == []

    def test_tool_call_missing_name_dropped(self):
        # A malformed tool_call (function dict without "name") must drop the
        # whole sample rather than crash the batch.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"arguments": {}}}
            ]},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "a"},
        ]})
        assert out["_prompt"] == []
        assert out["_response"] == []

    def test_tool_call_not_a_dict_dropped(self):
        # A tool_call element that is not a dict at all must not crash.
        conv = _make_converter()
        out = conv({"messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "tool_calls": ["not a dict"]},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "a"},
        ]})
        assert out["_prompt"] == []
        assert out["_response"] == []


class TestQwen36Templates:
    def test_qwen3_6_templates_registered(self):
        # Importing template.py registers the new templates into TEMPLATES.
        assert "qwen3_6" in TEMPLATES
        assert "qwen3_6_nothink" in TEMPLATES

    def test_preexisting_templates_preserved(self):
        # Templates that existed on master must not be clobbered.
        assert "qwen3_vl" in TEMPLATES
        assert "step3_vl" in TEMPLATES
