# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import bisect
import json
import os
import copy
from abc import abstractmethod, ABC
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import chain
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Optional, Union, Tuple, Literal, List, Dict, Type, TypedDict

import torch
from transformers import PreTrainedTokenizer, ProcessorMixin, AutoProcessor, AutoConfig, AutoTokenizer, PretrainedConfig
from mindspeed_mm.config.arguments.base_args import BaseArguments

from .log import get_logger
from .model_args import ProcessorArguments


IGNORE_INDEX = -100

logger = get_logger(__file__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESPONSE = "tool_response"


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import Seq2SeqTrainingArguments
    from .template import Template
    from .mm_plugin import AudioInput, ImageInput, VideoInput

    MediaType = Union[ImageInput, VideoInput, AudioInput]


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_media_files(self, media_files: Union["MediaType", List["MediaType"], None]) -> Optional[List["MediaType"]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if media_files is None:
            return None
        elif not isinstance(media_files, list):
            media_files = [media_files]
        elif len(media_files) == 0:
            return None
        else:
            media_files = media_files[:]
        for i, media in enumerate(media_files):
            if os.path.isfile(os.path.join(self.data_args.dataset_dir, media)):
                media_files[i] = os.path.join(self.data_args.dataset_dir, media)
            else:
                logger.warning(f"Media {media} does not exist in `media_dir`. Use original path.")
        return media_files

    @abstractmethod
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if (
                self.dataset_attr.ranking
                and isinstance(example[self.dataset_attr.chosen], str)
                and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(example[self.dataset_attr.response], str):  # normal example
            response = [{"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_images": self._find_media_files(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_media_files(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_media_files(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
                self.dataset_attr.system_tag
                and len(messages) != 0
                and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = (messages[0].get(self.dataset_attr.content_tag) or "").strip()
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                # Report only what is needed to diagnose (turn index, the bad role
                # tag, expected tags) instead of dumping the full conversation,
                # which can be tens of KB per sample for long agent traces.
                logger.warning_rank0(
                    f"Skipping sample: invalid role tag at turn {turn_idx} "
                    f"(got {message[self.dataset_attr.role_tag]!r}, "
                    f"expected one of {accept_tags[turn_idx % 2]})."
                )
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping.get(message.get(self.dataset_attr.role_tag)),
                    # Match the inference chat_template's `render_content(...)|trim`:
                    # strip leading/trailing whitespace on every message content.
                    # Agent traces (e.g. terminal SFT) often carry trailing '\n'
                    # on user turns that the jinja template would discard; without
                    # the trim the training tokens diverge from inference output.
                    "content": (message.get(self.dataset_attr.content_tag) or "").strip(),
                }
            )

        is_invalid_message_count = (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or \
                                   (self.dataset_attr.ranking and len(aligned_messages) % 2 == 0)
        if is_invalid_message_count:
            logger.warning_rank0(
                f"Skipping sample: invalid message count {len(aligned_messages)} "
                f"(ranking={self.dataset_attr.ranking})."
            )
            broken_data = True

        if broken_data:
            prompt, response = [], []
        elif (
                self.dataset_attr.ranking
                and isinstance(example[self.dataset_attr.chosen], dict)
                and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                    chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                    or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(
                    f"Skipping pairwise sample: invalid chosen/rejected role tag "
                    f"(chosen={chosen.get(self.dataset_attr.role_tag)!r}, "
                    f"rejected={rejected.get(self.dataset_attr.role_tag)!r}, "
                    f"expected one of {accept_tags[-1]})."
                )
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping.get(chosen.get(self.dataset_attr.role_tag)),
                    "content": (chosen.get(self.dataset_attr.content_tag) or "").strip(),
                },
                {
                    "role": tag_mapping.get(rejected.get(self.dataset_attr.role_tag)),
                    "content": (rejected.get(self.dataset_attr.content_tag) or "").strip(),
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_images": self._find_media_files(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_media_files(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_media_files(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output


@dataclass
class MultiModalToolDatasetConverter(DatasetConverter):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        messages = example[self.dataset_attr.messages]
        if (
                self.dataset_attr.system_tag
                and len(messages) != 0
                and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = messages

        prompt = aligned_messages[:-1]
        response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_images": self._find_media_files(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_media_files(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_media_files(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            "_tools": example['tools'] or None
        }
        return output


@dataclass
class OpenAIDatasetConverter(DatasetConverter):
    r"""Convert OpenAI ChatCompletion-style data into Role-tagged messages.

    Supports:
      - assistant.tool_calls (structured field; arguments may be dict or JSON string)
      - assistant.reasoning_content (Qwen3 thinking; merged as <think>...</think>)
      - role: "tool" responses (consecutive tool messages merged into one TOOL_RESPONSE)
      - tools schema column (rendered as list[str], joined with newlines downstream)

    Tool calls are serialized in Qwen3.6 XML form:
        <tool_call>
        <function=NAME>
        <parameter=KEY>
        VALUE
        </parameter>
        ...
        </function>
        </tool_call>
    """

    THINK_OPEN = "<think>\n"
    THINK_CLOSE = "\n</think>\n\n"
    TOOL_RESP_SEP = "\n</tool_response>\n<tool_response>\n"

    def _serialize_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> Optional[str]:
        """Serialize tool calls into Qwen3.6 XML. Returns None if any tool call
        is malformed (not a dict, or missing the function name), so the caller
        can drop the whole sample rather than crash or emit a corrupted turn."""
        blocks = []
        for tc in tool_calls:
            fn = tc.get("function", tc) if isinstance(tc, dict) else tc
            if not isinstance(fn, dict) or "name" not in fn:
                logger.warning_rank0(
                    "Skipping sample: malformed tool_call (missing function name)."
                )
                return None
            name = fn["name"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (TypeError, ValueError):
                    args = {}
            # Function-call arguments must be a JSON object. Anything else
            # (None from "null", a list, a number, an empty string, etc.) is
            # treated as "no arguments" — this both avoids silently dropping a
            # falsy value into {} inconsistently and prevents an AttributeError
            # on the args.items() call below for truthy non-dict values.
            if not isinstance(args, dict):
                args = {}
            param_lines = []
            for k, v in args.items():
                v_str = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                param_lines.append(f"<parameter={k}>\n{v_str}\n</parameter>")
            params_block = "\n".join(param_lines)
            if params_block:
                body = f"<function={name}>\n{params_block}\n</function>"
            else:
                body = f"<function={name}>\n</function>"
            blocks.append(f"<tool_call>\n{body}\n</tool_call>")
        return "\n".join(blocks)

    def _merge_thinking(self, content: str, reasoning: Optional[str]) -> str:
        content = content or ""
        if not reasoning:
            return content
        return self.THINK_OPEN + reasoning.strip("\n") + self.THINK_CLOSE + content

    def _extract_system(self, example: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
        """Pull a leading system message out of the trajectory, if present.

        Returns ``(system_text, remaining_messages)``.
        """
        messages = example[self.dataset_attr.messages]
        if (
                self.dataset_attr.system_tag
                and len(messages) > 0
                and messages[0].get(self.dataset_attr.role_tag) == self.dataset_attr.system_tag
        ):
            return (messages[0].get(self.dataset_attr.content_tag) or "").strip(), messages[1:]
        system = example[self.dataset_attr.system] if self.dataset_attr.system else ""
        return (system or "").strip(), messages

    def _build_assistant_turn(self, msg: Dict[str, Any], content: str) -> Optional[Dict[str, str]]:
        """Build a single ASSISTANT or TOOL_CALL turn from an assistant message.

        Returns the turn dict, or ``None`` if the message carries a malformed
        tool call (the caller drops the whole sample in that case).
        """
        tool_calls = msg.get("tool_calls")
        reasoning = msg.get("reasoning_content") or ""
        if not tool_calls:
            return {"role": Role.ASSISTANT.value, "content": self._merge_thinking(content, reasoning)}

        tc_block = self._serialize_tool_calls(tool_calls)
        if tc_block is None:
            # _serialize_tool_calls already logged the reason.
            return None
        if content.strip():
            body = content.rstrip() + "\n\n" + tc_block
        else:
            body = tc_block
        return {"role": Role.TOOL_CALL.value, "content": self._merge_thinking(body, reasoning)}

    def _flatten_messages(self, messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, str]]]:
        """Flatten OpenAI messages into Role.{USER, TOOL_CALL, TOOL_RESPONSE, ASSISTANT} turns.

        Returns the aligned turn list, or ``None`` if the trajectory is
        malformed (unknown role, or a malformed tool call).
        """
        aligned: List[Dict[str, str]] = []
        pending_tool_responses: List[str] = []

        def flush_tool_responses():
            if pending_tool_responses:
                aligned.append({
                    "role": Role.TOOL_RESPONSE.value,
                    "content": self.TOOL_RESP_SEP.join(pending_tool_responses),
                })
                pending_tool_responses.clear()

        for msg in messages:
            role = msg.get(self.dataset_attr.role_tag)
            # Match the inference chat_template's `render_content(...)|trim`: trim
            # every message's content before further processing. Without this,
            # trailing whitespace in tool outputs causes byte-level divergence vs
            # the official chat_template.
            content = (msg.get(self.dataset_attr.content_tag, "") or "").strip()

            if role == self.dataset_attr.observation_tag:
                pending_tool_responses.append(content)
                continue

            # Agent traces often inject role="user" messages mid-trajectory
            # (system reminders, follow-up instructions, harness hook output).
            # Without this branch they land at an odd slot in aligned[] -> the
            # alternation check trips and the whole sample is dropped. Fold them
            # into the surrounding input stream instead so alternation holds.
            if role == self.dataset_attr.user_tag and (
                pending_tool_responses
                or (aligned and aligned[-1]["role"] in (Role.TOOL_RESPONSE.value, Role.USER.value))
            ):
                if pending_tool_responses:
                    # Treat the user interjection as another tool_response-style
                    # block; it is wrapped by TOOL_RESP_SEP at flush time.
                    pending_tool_responses.append(content)
                else:
                    # Last aligned entry is already a user-side block — concatenate.
                    sep = "\n" if aligned[-1]["content"] else ""
                    aligned[-1]["content"] = aligned[-1]["content"] + sep + content
                continue

            flush_tool_responses()

            if role == self.dataset_attr.user_tag:
                aligned.append({"role": Role.USER.value, "content": content})
            elif role == self.dataset_attr.assistant_tag:
                turn = self._build_assistant_turn(msg, content)
                if turn is None:
                    return None
                aligned.append(turn)
            else:
                logger.warning_rank0(f"Unknown role {role!r} in OpenAI sample, dropping example.")
                return None

        flush_tool_responses()
        return aligned

    def _is_valid_alignment(self, aligned: List[Dict[str, str]]) -> bool:
        """Validate strict role alternation and an even turn count ending in
        assistant. Logs the specific failing rule and returns False if invalid.
        """
        odd_roles = (Role.USER.value, Role.TOOL_RESPONSE.value)
        even_roles = (Role.ASSISTANT.value, Role.TOOL_CALL.value)
        accept = (odd_roles, even_roles)
        for idx, m in enumerate(aligned):
            if m["role"] not in accept[idx % 2]:
                logger.warning_rank0(
                    f"Role alternation broken at turn {idx} role={m['role']!r}; dropping example."
                )
                return False
        if len(aligned) == 0 or len(aligned) % 2 != 0:
            logger.warning_rank0(
                f"Sample has {len(aligned)} aligned turns; SFT requires even count ending in assistant. Dropping."
            )
            return False
        return True

    def _build_tools(self, example: Dict[str, Any]) -> Optional[List[str]]:
        """Render the tools schema column into list[str] (one JSON-serialized
        tool per element, matching '\\n'.join in the template), or None when
        the column is absent or empty.
        """
        tools_field = self.dataset_attr.tools
        raw_tools = example.get(tools_field) if tools_field else None
        if not raw_tools:
            return None
        return [
            t if isinstance(t, str) else json.dumps(t, ensure_ascii=False)
            for t in raw_tools
        ]

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        # 1) extract leading system message (if data carries one)
        system, messages = self._extract_system(example)

        # 2) flatten messages into aligned turns; None means the trajectory is
        #    malformed and the sample is dropped.
        aligned = self._flatten_messages(messages)

        # 3) validate; on any failure the sample yields empty prompt/response.
        if aligned is None or not self._is_valid_alignment(aligned):
            prompt: List[Dict[str, str]] = []
            response: List[Dict[str, str]] = []
        else:
            prompt = aligned[:-1]
            response = aligned[-1:]

        return {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_images": self._find_media_files(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_media_files(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_media_files(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
            "_tools": self._build_tools(example),
        }


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
    "multimodal_tool": MultiModalToolDatasetConverter,
    "openai": OpenAIDatasetConverter,
}


def register_dataset_converter(name: str, dataset_converter: Type["DatasetConverter"]) -> None:
    r"""Register a new dataset converter."""
    if name in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} already exists.")

    DATASET_CONVERTERS[name] = dataset_converter


def get_dataset_converter(name: str, dataset_attr: "DatasetAttr", data_args: "DataArguments") -> "DatasetConverter":
    r"""Get a dataset converter."""
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
        dataset: Union["Dataset", "IterableDataset"],
        dataset_attr: "DatasetAttr",
        data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _images: []
    _videos: []
    _audios: []
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (int(os.getenv("LOCAL_RANK", -1)) != 0),
            desc="Converting format of dataset",
        )

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )


class DatasetAttr(BaseArguments):
    r"""
    Dataset attributes.
    """
    # basic configs
    ranking: bool = False
    # common columns
    system: Optional[str] = None
    images: Optional[str] = None
    videos: Optional[str] = None
    audios: Optional[str] = None
    # alpaca columns
    prompt: Optional[str] = None
    # alpaca tags
    query: Optional[str] = None
    response: Optional[str] = None
    history: Optional[str] = None
    # sharegpt columns
    messages: Optional[str] = "conversations"
    tools: Optional[str] = None
    # sharegpt tags
    role_tag: Optional[str] = "from"
    content_tag: Optional[str] = "value"
    user_tag: Optional[str] = "human"
    assistant_tag: Optional[str] = "gpt"
    observation_tag: Optional[str] = "observation"
    function_tag: Optional[str] = "function_call"
    system_tag: Optional[str] = "system"
    # rlhf columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    formatting: Literal["alpaca", "sharegpt", "multimodal_tool", "openai"] = "sharegpt"


class DataArguments(BaseArguments):
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Directory to read/write data. Defaults to `~/.cache/huggingface/datasets`(env:HF_DATASETS_CACHE)"},
    )
    template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which template to use for constructing prompts in training and inference."},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which chat template file to use for constructing prompts in training and inference."},
    )
    enable_thinking: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    dataset: Optional[Union[str, List[str]]] = field(
        default=None,
        metadata={
            "help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    cutoff_len: int = field(
        default=1024,
        metadata={
            "help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    tool_format: Optional[str] = field(
        default=None,
        metadata={
            "help": "Tool format to use for constructing function calling examples."},
    )
    val_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the validation dataset."},
    )
    val_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes, truncate the number of examples for each validation dataset."},
    )
    val_rate: Optional[float] = field(
        default=None,
        metadata={"help": "The proportion of the dataset to be used for validation."},
    )
    packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    preprocess_on_fly: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to perform preprocess during training."},
    )
    async_preprocess: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to perform async preprocess during training."},
    )
    async_preprocess_buffer_size: Optional[int] = field(
        default=None,
        metadata={"help": "Buffer size for async preprocess. Defaults to 8 when not set and num_workers is unset."},
    )
    stage: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the validation dataset."},
    )

    def model_post_init(self, __context):
        if self.neat_packing:
            self.packing = True
        if self.stage is None:
            self.stage = "sft"
        if self.stage not in ["sft", "pretrain", "rm"]:
            raise ValueError(f"not support stage: {self.stage}")

        if self.dataset and not isinstance(self.dataset, list):
            self.dataset = self.dataset.split(",")


def search_for_fit(numbers: List[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: List[int], capacity: int) -> List[List[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
            self,
            prompt: List[Dict[str, str]],
            response: List[Dict[str, str]],
            system: Optional[str],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
            tools: List[str],
    ) -> Tuple[List[int], List[int]]:
        messages = self.template.mm_plugin.process_messages(prompt + response, images, videos, audios, self.processor)
        input_ids, labels = self.template.mm_plugin.process_token_ids(
            [], [], images, videos, audios, self.tokenizer, self.processor
        )
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system, tools)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                logger.info(
                    f"Maximum sequence length {self.data_args.cutoff_len} reached. "
                    f"Please increase seq_len or cutoff_len in config."
                )
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
        # for multiturn examples, we only mask the prompt part in each prompt-response pair.
        # Pre-seed every expected key so an all-broken batch still returns a dict whose
        # schema matches non-empty batches. With defaultdict(list), datasets.map(batched=True)
        # establishes the arrow schema from the first non-empty batch, and a later batch that
        # never wrote any key returns {}, which pyarrow rejects ("Schema and number of arrays
        # unequal").
        model_inputs: Dict[str, List[Any]] = {
            "input_ids": [], "attention_mask": [], "labels": [],
            "images": [], "videos": [], "audios": [],
        }
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    f"Dropped invalid example: prompt_turns={len(examples['_prompt'][i])} "
                    f"response_turns={len(examples['_response'][i])} "
                    f"(expected odd prompt + 1 response)."
                )
                continue

            tool_schema = []
            if '_tools' in examples:
                tool_schema = examples['_tools'][i]
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                tools=tool_schema,
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"labels:\n{self.tokenizer.decode(valid_labels, skip_special_tokens=False)}")


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels, batch_images, batch_videos, batch_audios = [], [], [], [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                logger.warning_rank0(
                    f"Dropped invalid example: prompt_turns={len(examples['_prompt'][i])} "
                    f"response_turns={len(examples['_response'][i])} "
                    f"(expected odd prompt + 1 response)."
                )
                continue

            tool_schema = []
            if "_tools" in examples:
                tool_schema = examples['_tools'][i]

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
                tools=tool_schema
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning_rank0(f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}.")
            else:
                labels[0] = IGNORE_INDEX  # mark the last token's label as ignore.
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                batch_images.append(examples["_images"][i] or [])
                batch_videos.append(examples["_videos"][i] or [])
                batch_audios.append(examples["_audios"][i] or [])
                valid_num += 1

        # Same pyarrow schema-stability fix as SupervisedDatasetProcessor: pre-seed all
        # keys so an all-broken batch still satisfies the arrow schema check.
        model_inputs: Dict[str, List[Any]] = {
            "input_ids": [], "attention_mask": [], "position_ids": [], "labels": [],
            "images": [], "videos": [], "audios": [], "cu_seqlens": [],
        }
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)
        for knapsack in knapsacks:
            packed_input_ids, packed_attention_masks, packed_position_ids, packed_labels = [], [], [], []
            packed_images, packed_videos, packed_audios, cu_seqlens = [], [], [], [0]
            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_input_ids += batch_input_ids[index]
                packed_position_ids += list(range(len(batch_input_ids[index])))  # NOTE: pad_to_multiple_of ignore this
                packed_labels += batch_labels[index]
                packed_images += batch_images[index]
                packed_videos += batch_videos[index]
                packed_audios += batch_audios[index]
                cu_seqlens += [cu_seqlens[-1] + length]
                if self.data_args.neat_packing:
                    packed_attention_masks += [i + 1] * len(batch_input_ids[index])  # start from 1
                else:
                    packed_attention_masks += [1] * len(batch_input_ids[index])

            model_inputs["input_ids"].append(packed_input_ids)
            model_inputs["attention_mask"].append(packed_attention_masks)
            model_inputs["position_ids"].append(packed_position_ids)
            model_inputs["labels"].append(packed_labels)
            model_inputs["images"].append(packed_images or None)
            model_inputs["videos"].append(packed_videos or None)
            model_inputs["audios"].append(packed_audios or None)
            model_inputs["cu_seqlens"].append(cu_seqlens)

        return model_inputs


class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
            self,
            prompt: List[Dict[str, str]],
            response: List[Dict[str, str]],
            system: Optional[str],
            images: List["ImageInput"],
            videos: List["VideoInput"],
            audios: List["AudioInput"],
    ) -> Tuple[List[int], List[int], List[int], List[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels

    def preprocess_dataset(self, examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        # Same pyarrow schema-stability fix as SupervisedDatasetProcessor.
        model_inputs: Dict[str, List[Any]] = {
            "chosen_input_ids": [], "chosen_attention_mask": [], "chosen_labels": [],
            "rejected_input_ids": [], "rejected_attention_mask": [], "rejected_labels": [],
            "images": [], "videos": [], "audios": [],
        }
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    f"Dropped invalid pairwise example: prompt_turns={len(examples['_prompt'][i])} "
                    f"response_turns={len(examples['_response'][i])} "
                    f"(expected odd prompt + >=2 responses)."
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: Dict[str, List[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> Tuple[int, int]:
    r"""
    Computes the real sequence length after truncation by the cutoff_len.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(
            cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def get_vision_feature_select_strategy(config: "PretrainedConfig") -> int:
    r"""
    Get the vision_feature_select_strategy.
    """
    vision_feature_select_strategy = getattr(config, "vision_feature_select_strategy", "default")
    return vision_feature_select_strategy


def load_tokenizer(model_args: "ProcessorArguments") -> "TokenizerModule":
    r"""
    Loads pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="right",
        local_files_only=True,
        trust_remote_code=model_args.trust_remote_code,
    )

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            local_files_only=True,
            trust_remote_code=model_args.trust_remote_code,
        )
        setattr(processor, "tokenizer", tokenizer)
        setattr(processor, "image_max_pixels", model_args.image_max_pixels)
        setattr(processor, "image_min_pixels", model_args.image_min_pixels)
        setattr(processor, "image_do_pan_and_scan", model_args.image_do_pan_and_scan)
        setattr(processor, "crop_to_patches", model_args.crop_to_patches)
        setattr(processor, "video_max_pixels", model_args.video_max_pixels)
        setattr(processor, "video_min_pixels", model_args.video_min_pixels)
        setattr(processor, "video_fps", model_args.video_fps)
        setattr(processor, "video_maxlen", model_args.video_maxlen)
        setattr(processor, "audio_sampling_rate", model_args.audio_sampling_rate)
        setattr(processor, "use_audio_in_video", model_args.use_audio_in_video)
    except Exception as e:
        logger.warning("Processor was not found: %s.", e)
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        processor = None

    return {"tokenizer": tokenizer, "processor": processor}


def update_tokenizer_with_chat_template(tokenizer: "PreTrainedTokenizer", chat_template: str) -> "PreTrainedTokenizer":
    r"""
    Update tokenizer with custom chat_template file.
    """

    if not os.path.isfile(chat_template):
        raise FileNotFoundError(f"The chat_template path {chat_template} does not exist or is not a file")

    with open(chat_template, 'r', encoding='utf-8') as f:
        chat_template_content = f.read()

    logger.info("Apply custom chat_template %s to tokenizer.", chat_template)
    tokenizer.chat_template = chat_template_content
    return tokenizer
