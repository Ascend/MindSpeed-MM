# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References:
# TextProcesser: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py
# DataSet https://github.com/OpenGVLab/InternVL/blob/main/internvl_chat/internvl/train/dataset.py


import os
import re
import html
import copy
import random
import math
import urllib.parse as ul
from collections import Counter, defaultdict
from logging import getLogger
from typing import Any, Dict, Optional, Tuple, Union, Sequence, Type, Callable

try:
    import decord
except Exception as e:
    print(f"Failed to import decord module. The reason of decord unavailable is {e}")

import av
import ftfy
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from einops import rearrange
import torch.nn.functional as F
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from packaging import version
import tokenizers
from megatron.training import get_args
from megatron.core import mpu

from mindspeed_mm.data.data_utils.transform_pipeline import get_transforms
from mindspeed_mm.data.data_utils.conversation import get_conv_template
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS


logger = getLogger(__name__)
VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
TENSOR_EXTENSIONS = (".pt", ".pth")
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


class DataFileReader:
    """get the data from different types of files such as csv/json/parquat"""

    def __init__(self, data_storage_mode="standard"):
        self.data_storage_mode = data_storage_mode

    def __call__(self, data_path, return_type="list"):
        if self.data_storage_mode == "standard":
            return self.get_datasamples(data_path, return_type=return_type)
        elif self.data_storage_mode == "combine" or self.data_storage_mode == "sorafeatured":
            return self.get_cap_list(data_path)
        elif self.data_storage_mode == "vace":
            return self.get_vace_datasamples(data_path)
        else:
            raise NotImplementedError("Not support now.")

    @staticmethod
    def get_datasamples(data_path, return_type="list"):
        if data_path.endswith(".csv"):
            data_out = pd.read_csv(data_path)
            if return_type == "list":
                return data_out.to_dict("records")
            else:
                return data_out
        elif data_path.endswith(".json"):
            data_out = pd.read_json(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".jsonl"):
            data_out = pd.read_json(data_path, lines=True)
            return data_out.to_dict("records")
        elif data_path.endswith(".parquat"):
            data_out = pd.read_parquat(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".txt"):
            with open(data_path, 'r') as f:
                data_out = f.readlines()
            data_out = [data.strip() for data in data_out]
            return data_out
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path}")

    def get_cap_list(self, data_path):
        cap_lists = []
        with open(data_path, "r") as f:
            folder_anno = [
                i.strip().split(",")
                for i in f.readlines()
                if len(i.strip()) > 0
            ]
        for folder, anno in folder_anno:
            sub_list = self.get_datasamples(anno)
            print(f"Building {anno}...")
            for sub in sub_list:
                sub["path"] = os.path.join(folder, sub["path"])
            cap_lists += sub_list
        return cap_lists

    def get_vace_datasamples(self, data_path):
        data_file_keys = ["video", "src_video", "src_video_mask", "src_ref_images"]
        cap_lists = []
        with open(data_path, "r") as f:
            folder_anno = [
                i.strip().split(",")
                for i in f.readlines()
                if len(i.strip()) > 0
            ]
        for folder, anno in folder_anno:
            sub_list = self.get_datasamples(anno)
            print(f"Building {anno}...")
            for sub in sub_list:
                for key in data_file_keys:
                    if check_none(sub[key]):
                        sub[key] = None
                    if sub[key]:
                        if isinstance(sub[key], list):
                            new_sub = []
                            for file in sub[key]:
                                new_sub.append(os.path.join(folder, file))
                            sub[key] = new_sub
                        else:
                            sub[key] = os.path.join(folder, sub[key])

            cap_lists += sub_list
        return cap_lists


class DecordInit:
    """Using Decord (https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
            f"sr={self.sr},"
            f"num_threads={self.num_threads})"
        )
        return repr_str


class DataStats:
    def __init__(self):
        self.counters = defaultdict(int)
        self.collections = defaultdict(list)
    
    def increment(self, key, value=1):
        self.counters[key] += value

    def collect(self, key, item):
        self.collections[key].append(item)

    def print_report(self):
        report = ["\n=== Data Processing Report ==="]
        for k, v in self.counters.items():
            print(f"{k.replace('_', ' ').title():<25}: {v}")
        if self.counters:
            for k, v in sorted(self.counters.items()):
                report.append(f"  {k}: {v}")

        return "\n".join(report)


class ImageProcesser:
    """Used for image data preprocessing"""

    def __init__(
            self,
            num_frames=16,
            train_pipeline=None,
            image_reader_type="torchvision",
            image_processer_type="image2video",
            dynamic_image_size=False,
            image_size=224,
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            use_thumbnail=False,
            transform_size=None,
            **kwargs,
    ):
        self.num_frames = num_frames
        self.image_transforms = get_transforms(
            is_video=False, train_pipeline=train_pipeline, transform_size=transform_size
        )
        self.video_transforms = get_transforms(
            is_video=True, train_pipeline=train_pipeline, transform_size=transform_size
        )
        self.train_pipeline = train_pipeline
        self.image_reader_type = image_reader_type
        self.image_processer_type = image_processer_type
        self.dynamic_image_size = dynamic_image_size
        self.image_size = image_size
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.use_thumbnail = use_thumbnail
        self.is_image = False

    def __call__(self, image_path, mode="", num_image=1):
        if self.image_processer_type == "image2video":
            image = self.image_to_video(image_path)
        elif self.image_processer_type == "image2image":
            image = self.image_to_image(image_path)
        else:
            raise NotImplementedError(
                f"Unsupported image processor type: {self.image_processer_type}"
            )
        return image

    def image_to_video(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        image = self.image_transforms(image)
        video = image.repeat(self.num_frames, 1, 1, 1)
        video = video.permute(1, 0, 2, 3)  # TCHW -> CTHW
        return video

    def image_to_image(self, image_path):
        image = self.image_reader(image_path)
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  # [1 c h w]
        # [1 C H W] -> num_img [1 C H W]
        if "human_images" in image_path or self.is_image:
            image = self.image_transforms(image)
        else:
            image = self.video_transforms(image)
        # [1 C H W] -> [C 1 H W]
        image = image.permute(1, 0, 2, 3)
        return image

    def image_reader(self, image_path):
        if self.image_reader_type in ["torchvision", "CLIPImageProcessor"]:
            image = pil_loader(image_path)
        elif self.image_reader_type == "Image":
            image = Image.open(image_path).convert("RGB")  # [h, w, c]
        else:
            raise NotImplementedError(
                f"Unsupported image reader type: {self.image_reader_type}"
            )
        return image


class TextProcesser:
    """Used for text data preprocessing"""

    bad_punct_regex = re.compile(
        r"["
        + "#®•©™&@·º½¾¿¡§~"
        + "\)"
        + "\("
        + "\]"
        + "\["
        + "\}"
        + "\{"
        + "\|"
        + "\\"
        + "\/"
        + "\*"
        + r"]{1,}"
    )

    def __init__(
            self,
            tokenizer=None,
            use_clean_caption=True,
            enable_text_preprocessing=True,
            padding_type="max_length",
            support_chinese=False,
            text_preprocess_methods=None,
            cfg=0.1,
    ):
        self.padding = padding_type
        self.tokenizer = tokenizer
        self.use_clean_caption = use_clean_caption
        self.support_chinese = support_chinese
        self.cfg = cfg
        self.enable_text_preprocessing = enable_text_preprocessing
        self.text_preprocess_methods = text_preprocess_methods

    def __call__(self, texts):
        if self.enable_text_preprocessing:
            if isinstance(texts, tuple) or isinstance(texts, list):
                texts_info = [
                    TextProcesser.text_preprocessing(
                        text,
                        self.use_clean_caption,
                        text_preprocess_methods=self.text_preprocess_methods
                    )
                    for text in texts
                ]
            else:
                texts_info = TextProcesser.text_preprocessing(
                    texts,
                    self.use_clean_caption,
                    text_preprocess_methods=self.text_preprocess_methods
                )
            texts_info = texts_info if random.random() > self.cfg else [""]
        else:
            texts_info = texts

        if not isinstance(self.tokenizer, list):
            text_tokens_and_mask = self.tokenizer(
                texts_info,
                max_length=self.tokenizer.model_max_length,
                padding=self.padding,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            prompt_ids = text_tokens_and_mask["input_ids"]
            prompt_mask = text_tokens_and_mask["attention_mask"]
        else:
            prompt_ids, prompt_mask = [], []
            for tokenizer in self.tokenizer:
                text_tokens_and_mask = tokenizer(
                    texts_info,
                    max_length=tokenizer.model_max_length,
                    padding=self.padding,
                    truncation=True,
                    return_attention_mask=True,
                    add_special_tokens=True,
                    return_tensors="pt"
                )
                prompt_ids.append(text_tokens_and_mask["input_ids"])
                prompt_mask.append(text_tokens_and_mask["attention_mask"])
        return (prompt_ids, prompt_mask)

    @staticmethod
    def text_preprocessing(text, use_clean_caption=True, support_chinese=False, text_preprocess_methods=None):
        if text_preprocess_methods:
            if isinstance(text_preprocess_methods, list):
                for text_preprocess_method in text_preprocess_methods:
                    text = TextProcesser.text_preprocessing(text, text_preprocess_methods=text_preprocess_method)
            else:
                method_name = text_preprocess_methods["method"]
                param = text_preprocess_methods.get("param", None)
                method = getattr(TextProcesser, method_name, None)
                if method:
                    if param:
                        text = method(text, **param)
                    else:
                        text = method(text)
                else:
                    raise NotImplementedError(f"The text preprocessing method {method_name} is not implemented.")
        else:
            if use_clean_caption:
                text = TextProcesser.clean_caption(text, support_chinese=support_chinese)
            else:
                text = text.lower().strip()
        return text

    @staticmethod
    def basic_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
    
    @staticmethod
    def whitespace_clean(text):
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    @staticmethod
    def clean_caption(caption, support_chinese=False):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        if not support_chinese:
            caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            "-",
            caption,
        )

        # Uniform quotation marks
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip addresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(
            r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
        )

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(
            TextProcesser.bad_punct_regex, r" ", caption
        )  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = TextProcesser.basic_clean(caption)

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(
            r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
        )
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(
            r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
        )  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()


def get_seed_worker(seed):
    """Deterministic dataloader"""

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


class SingletonMeta(type):
    """
    This is a metaclass for creating singletons.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


def format_numel_str(numel: int) -> str:
    B = 1024 ** 3
    M = 1024 ** 2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def collate_fn_default(batch):
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        input_ids = torch.cat(input_ids, dim=-1)
        use_mask = True
    elif "mask" in batch[0] and isinstance(batch[0]["mask"], torch.Tensor):
        masks = [x.pop("mask") for x in batch]
        input_ids = [x.pop("input_ids") for x in batch]
        masks = torch.cat(masks, dim=0)
        input_ids = torch.cat(input_ids, dim=0)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["input_ids"] = input_ids

    return ret


def preprocess_multimodal(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
) -> Dict:
    """
    Process multimodal sources by handling image tokens.
    """
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    img_start_token = MODEL_CONSTANTS['llava']["IMG_START_TOKEN"]
    img_end_token = MODEL_CONSTANTS['llava']["IMG_END_TOKEN"]

    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if image_token in sentence["value"]:
                sentence["value"] = sentence["value"].replace(image_token, "").strip()
                sentence["value"] = image_token + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
            replace_token = image_token
            if mm_use_im_start_end:
                replace_token = img_start_token + replace_token + img_end_token
            sentence["value"] = sentence["value"].replace(image_token, replace_token)

    return sources


def preprocess_v1(
        sources,
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = True
) -> Dict:
    """
    Process sources for llava-v1 of the preprocessing pipeline.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    conv = get_conv_template("llava-v1")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
             for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = ignore_index
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer, image_token_index))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer, image_token_index)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = ignore_index

            cur_len += round_len
        target[cur_len:] = ignore_index

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = ignore_index
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
    )


def preprocess_plain(
        sources: Sequence[str],
        is_multimodal,
        mm_use_im_start_end,
        tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """
    Process plain text sources for preprocessing.
    """
    sources = preprocess_multimodal(sources, is_multimodal, mm_use_im_start_end)

    image_token_index = MODEL_CONSTANTS['llava']["IMAGE_TOKEN_INDEX"]
    image_token = MODEL_CONSTANTS['llava']["IMAGE_TOKEN"]
    ignore_index = MODEL_CONSTANTS['llava']["IGNORE_INDEX"]
    conv = get_conv_template("llava-plain")
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        source[0]["value"] = image_token
        conversation = source[0]["value"] + source[1]["value"] + conv.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, image_token_index=image_token_index, return_tensors="pt")
                 for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer, image_token_index=image_token_index))
        target[:tokenized_len] = ignore_index

    return dict(input_ids=input_ids[0], labels=targets[0])


def preprocess_internlm(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        num_image: int = 1
) -> Dict:
    """
    Process sources for internvl model preprocessing.
    """
    conv = get_conv_template(template_name)
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    conversations = get_formatted_conversations(sources, roles, conv)

    im_start_token = MODEL_CONSTANTS['internvl']["IMG_START_TOKEN"]
    im_context_token = MODEL_CONSTANTS['internvl']["IMG_CONTEXT_TOKEN"]
    im_end_token = MODEL_CONSTANTS['internvl']["IMG_END_TOKEN"]

    if not text_only:
        new_conversations = []
        for conversation in conversations:
            for i in range(num_image):
                image_tokens = f"{im_start_token}{im_context_token * num_image_token_list[i]}{im_end_token}"
                conversation = conversation.replace("<image>", image_tokens, 1)
            new_conversations.append(conversation)
        conversations = new_conversations

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding=False if group_by_length or use_packed_ds else "max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())  # 浦语里面 pad_token_id = eos_token_id
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID  # <s>
        parts = conversation.split(conv.roles[1])  # [UNUSED_TOKEN_146]assistant\n
        info = parts[0] + conv.roles[1]
        temp_len = len(tokenizer(info).input_ids) - 1  # 去除tokenizer的<s>
        target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
        cur_len = cur_len + temp_len

        for index in range(1, len(parts) - 1):
            info = parts[index]
            part1, part2 = info.split(conv.roles[0])
            temp_len = len(tokenizer(part1).input_ids) - 1
            cur_len = cur_len + temp_len
            part = conv.roles[0] + part2 + conv.roles[1]
            temp_len = len(tokenizer(part).input_ids) - 1
            target[cur_len: cur_len + temp_len] = IGNORE_TOKEN_ID
            cur_len = cur_len + temp_len
        last_info = parts[-1]
        temp_len = len(tokenizer(last_info).input_ids) - 1
        cur_len = cur_len + temp_len

        target[cur_len:] = IGNORE_TOKEN_ID
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.", flush=True)

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=input_ids.ne(tokenizer.pad_token_id)[0],
    )


def pad_to_multiple(sequence, multiple=1, pad_value=0):
    current_length = sequence.size(0)
    target_length = ((current_length + multiple - 1) // multiple) * multiple # 计算最近的N的倍数
    padding_length = target_length - current_length
    return F.pad(sequence, (0, padding_length), value=pad_value)


def preprocess_internvl2_5(
        template_name,
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        num_image_token_list: list,
        text_only: bool = False,
        group_by_length: bool = False,
        use_packed_ds: bool = False,
        ds_name: str = None,
        num_image: int = 1
) -> Dict:
    if len(sources) != 1:
        raise ValueError('process only the first conversations')
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]  # remove system prompt
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    if not text_only:
        IMG_START_TOKEN_ = MODEL_CONSTANTS[template_name]['IMG_START_TOKEN']
        IMG_CONTEXT_TOKEN_ = MODEL_CONSTANTS[template_name]['IMG_CONTEXT_TOKEN']
        IMG_END_TOKEN_ = MODEL_CONSTANTS[template_name]['IMG_END_TOKEN']
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for _ in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = f'{IMG_START_TOKEN_}{IMG_CONTEXT_TOKEN_ * num_image_token_list[current_image_idx]}{IMG_END_TOKEN_}'
                    conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        if current_image_idx != num_image:
            raise ValueError(f"{current_image_idx} != {num_image}")

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    for conversation in conversations:
        if conversation['from'] == 'human':
            batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
            roles.append('human')
        elif conversation['from'] == 'gpt':
            batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
            roles.append('gpt')
        else:
            raise NotImplementedError

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:  # for InternLM series
        batches[0] = tokenizer.bos_token + batches[0]

    # Tokenize conversations
    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:  # for InternLM series
        input_ids = [item[1:] for item in input_ids]

    final_input_ids, final_targets = [], []
    ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
    ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
    for role, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
        if role == 'system' or role == 'human':
            final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
        elif role == 'gpt':
            target = input_id.copy()
            target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
            target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
            final_targets.append(target)
        else:
            raise NotImplementedError
    input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
    targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

    if get_args().context_parallel_size > 1:
        # 如果开启CP，序列长度自动padding到CP的倍数
        cp_size = get_args().context_parallel_size
        input_ids = pad_to_multiple(input_ids, cp_size * 2, tokenizer.pad_token_id)
        targets = pad_to_multiple(targets, cp_size * 2, IGNORE_TOKEN_ID)

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)

    return dict(
        input_ids=input_ids[0],
        labels=targets[0],
        attention_mask=input_ids.ne(tokenizer.pad_token_id)[0],
    )


def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
    """
    Tokenize prompts with image tokens.
    """
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def get_formatted_conversations(sources, roles, conv):
    """
    Format conversations based on provided roles and conversation template.
    """
    # Apply prompt templates
    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            if role != conv.roles[j % 2]:
                raise ValueError(
                    f"Role mismatch at {sentence}, expected {conv.roles[j % 2]}, got {role}")
            sentence["value"] = sentence["value"].strip()
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    return conversations


def preprocess(
        template_name,
        sources,
        tokenizer,
        num_image_token_list,
        group_by_length,
        is_multimodal,
        mm_use_im_start_end,
        num_image: int = 1
):
    """
    Select and run the appropriate preprocessing function based on template name.
    """
    if template_name == "internlm2-chat":
        ret = preprocess_internlm(template_name, sources,
                                  tokenizer, num_image_token_list,
                                  group_by_length=group_by_length,
                                  num_image=num_image)
    elif template_name in ("internvl2_5", "internvit_qwen3"):
        ret = preprocess_internvl2_5(template_name, sources,
                                     tokenizer, num_image_token_list,
                                     group_by_length=group_by_length,
                                     num_image=num_image)
    elif template_name == "llava-v1":
        ret = preprocess_v1(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer,
            has_image=True)
    elif template_name == "llava-plain":
        ret = preprocess_plain(
            sources,
            is_multimodal,
            mm_use_im_start_end,
            tokenizer)
    else:
        raise ValueError("%s preprocessor is not implemented" % type(template_name))
    return ret


def build_iterations(train_dl=None, val_dl=None, test_dl=None, iterator_type="cyclic"):

    def _cyclic_iter(dl):
        while True:
            for x in dl:
                yield x
    
    def _get_iterator(dataloader, iter_type=iterator_type):
        """Return dataset iterator."""
        if iter_type == "single":
            return iter(dataloader)
        elif iter_type == "cyclic":
            return iter(_cyclic_iter(dataloader))
        else:
            raise NotImplementedError("unexpected iterator type")
    
    if train_dl is not None:
        train_data_iterator = _get_iterator(train_dl)
    else:
        train_data_iterator = None

    if val_dl is not None:
        valid_data_iterator = _get_iterator(val_dl)
    else:
        valid_data_iterator = None

    if test_dl is not None:
        test_data_iterator = _get_iterator(test_dl)
    else:
        test_data_iterator = None

    return train_data_iterator, valid_data_iterator, test_data_iterator


def get_value_from_args(key: str, default_value=None):
    """
    Get value from global args
    """
    try:
        config = get_args()
        for subkey in key.split("."):
            config = getattr(config, subkey)
        return config
    except AttributeError as e:
        if default_value is None:
            raise KeyError(f"Configuration key '{key}' not found, please check.") from e
        logger.info(f"Configuration key '{key}' not found, using default value: {default_value}.")
        return default_value


def cal_gradient_accumulation_size():
    args = get_args()
    world_size = torch.distributed.get_world_size()
    acc = int(args.global_batch_size / world_size / args.micro_batch_size * mpu.get_tensor_model_parallel_world_size()
                  * mpu.get_context_parallel_world_size() * mpu.get_pipeline_model_parallel_world_size())
    
    if getattr(args, "dist_train", False):
        from mindspeed.core.multi_modal.dist_train.dist_parallel_state import is_in_subworld
        from mindspeed.core.multi_modal.dist_train.dist_train_config import get_dist_model_config
        if is_in_subworld("vae"):
            dit_cfg = get_dist_model_config('dit')
            acc = int(
                args.global_batch_size / dit_cfg.world_size / args.micro_batch_size * dit_cfg.tensor_model_parallel_size
                * dit_cfg.context_parallel_size * dit_cfg.pipeline_model_parallel_size)
    return acc


def map_target_fps(
    fps: float,
    max_fps: float,
) -> Tuple[float, int]:
    """
    Map fps to a new fps that is less than max_fps.

    Args:
        fps (float): Original fps.
        max_fps (float): Maximum fps.

    Returns:
        tuple[float, int]: New fps and sampling interval.
    """
    if math.isnan(fps):
        return 0.0, 1
    if fps < max_fps:
        return fps, 1
    sampling_interval = math.ceil(fps / max_fps)
    new_fps = math.floor(fps / sampling_interval)
    return new_fps, sampling_interval


def check_none(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value) or np.isnan(value)
    return False

