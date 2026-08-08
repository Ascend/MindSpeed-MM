# --------------------------------------------------------
# Modified from huggingface diffusers repos
# This source code is licensed under the notice found in the root directory of this source tree.
# --------------------------------------------------------
# References: TextProcesser (diffusers deepfloyd_if pipeline)

"""Data reader/media processing utilities for generative-model (Wan2.2) FSDP2 datasets.
Moved out of ``data_utils/utils.py``; carries no megatron dependency."""

import os
import re
import html
import random
import math
import urllib.parse as ul
from collections import defaultdict
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import decord
except Exception as e:
    print(f"Failed to import decord module. The reason of decord unavailable is {e}")

import orjson
import ftfy
import torch
import numpy as np
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from einops import rearrange
from torchvision.datasets.folder import pil_loader

from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.transform_pipeline import get_transforms


VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
TENSOR_EXTENSIONS = (".pt", ".pth")


class DataFileReader:
    """get the data from different types of files such as csv/json/parquet"""

    def __init__(self, data_storage_mode="standard", **kwargs):
        """``data_storage_mode`` controls loading; ``reserved_keys`` filters keys from data.json; ``use_multiprocess`` enables parallel file processing (not recommended for fewer than 4 files)."""
        self.data_storage_mode = data_storage_mode
        self.reserved_keys = kwargs.get("reserved_keys", None)
        self.use_multiprocess = kwargs.get("use_multiprocess", False)

    def __call__(self, data_path, return_type="list"):
        if self.data_storage_mode == "standard":
            return self.get_datasamples(data_path, return_type=return_type)
        elif self.data_storage_mode == "combine" or self.data_storage_mode == "sorafeatured":
            redirect_keys = ["path"]
            return self.get_cap_list(data_path, redirect_keys)
        elif self.data_storage_mode == "vace":
            redirect_keys = ["video", "src_video", "src_video_mask", "src_ref_images"]
            return self.get_cap_list(data_path, redirect_keys)
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
            return orjson_load(data_path)
        elif data_path.endswith(".jsonl"):
            return orjson_load(data_path)
        elif data_path.endswith(".parquet"):
            data_out = pd.read_parquet(data_path)
            return data_out.to_dict("records")
        elif data_path.endswith(".txt"):
            with open(data_path, 'r') as f:
                data_out = f.readlines()
            data_out = [data.strip() for data in data_out]
            return data_out
        else:
            raise NotImplementedError(f"Unsupported file format: {data_path}")

    def get_cap_list(self, data_path, redirect_keys=None):
        with open(data_path, "r") as f:
            folder_anno = []
            for line_no, line in enumerate(f.readlines(), 1):
                if len(line.strip()) == 0:
                    continue
                fields = line.strip().split(",")
                if len(fields) != 2:
                    print(f"Skip malformed line {line_no} in {data_path}: expected 2 fields, got {len(fields)}")
                    continue
                folder_anno.append(fields)
        json_loader = JsonLoader([temp[1] for temp in folder_anno], use_multiprocess=self.use_multiprocess)
        for folder, anno in folder_anno:
            json_loader.set_process_func(anno, self._change_path, redirect_keys, folder)
        json_loader.set_process_func("all", self._remove_unused_keys, self.reserved_keys)
        content = json_loader.get_data()
        return content

    def _change_path(self, content, change_list, new_path):
        """Update file paths in specified keys to new base directory"""
        if change_list is None or len(change_list) == 0:
            return content
        for item in content:
            for key in change_list:
                if check_none(item.get(key)):
                    item[key] = None
                if item.get(key):
                    if isinstance(item[key], list):
                        new_sub = []
                        for file in item[key]:
                            new_sub.append(os.path.join(new_path, file))
                        item[key] = new_sub
                    else:
                        item[key] = os.path.join(new_path, item[key])
        return content

    def _remove_unused_keys(self, content, reserved_keys):
        """Filter dictionary items to keep only specified keys"""
        if reserved_keys is None or len(reserved_keys) == 0:
            return content
        new_contents = []
        for sub in content:
            new_contents.append({key: sub[key] for key in sub.keys() if key in reserved_keys})
        return new_contents


class JsonLoader:
    def __init__(self, json_path, use_multiprocess=False):
        """Initialize JsonLoader with JSON file paths and multiprocessing option"""
        self.json_path = json_path
        self.use_multiprocess = use_multiprocess
        self.json_contents = None
        self.process_funcs = {}

        self._check()
        self.json_path = [self.json_path] if isinstance(self.json_path, str) else self.json_path

    def _check(self):
        """Validate JSON file paths and check file existence"""
        if isinstance(self.json_path, str):
            if not os.path.exists(self.json_path):
                raise FileNotFoundError(f"{self.json_path} does not exist")
        elif isinstance(self.json_path, list):
            for path in self.json_path:
                if not isinstance(path, str):
                    raise TypeError("Unsupported data type")
                if not (path.endswith(".json") or path.endswith(".jsonl")):
                    raise TypeError("Unsupported file type")
                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} does not exist")
        else:
            raise TypeError("Unsupported data type")

    def set_process_func(self, file, process_func, *args, **kwargs):
        """Register data processing function for specified file"""
        if file == 'all':
            for _path in self.json_path:
                self.set_process_func(_path, process_func, *args, **kwargs)
        else:
            if file not in self.process_funcs:
                self.process_funcs[file] = []
            if all(fn["func"] != process_func for fn in self.process_funcs[file]):
                self.process_funcs[file].append({'func': process_func, 'args': args, 'kwargs': kwargs})

    def start_load(self):
        """Load JSON data using multiprocessing or single-process mode"""
        total_contents = []
        if self.use_multiprocess:
            total_contents = self._multiprocess_share_memory()
        else:
            for path in self.json_path:
                json_content = orjson_load(path)
                print(f"Building {path}...")
                if path in self.process_funcs:
                    for fn in self.process_funcs[path]:
                        json_content = fn["func"](json_content, *fn['args'], **fn['kwargs'])
                total_contents += json_content
        self.json_contents = total_contents

    def _multiprocess_share_memory(self):
        """Load JSON data using shared memory multiprocessing"""
        total_contents = []
        num_processes = len(self.json_path)
        shm_objects = []
        shm_size = []
        for path in self.json_path:
            size = int(os.path.getsize(path) * 1.2)
            shm = shared_memory.SharedMemory(create=True, size=size)
            shm_objects.append(shm)
            shm_size.append(size)
        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                future_to_task = {}
                for i in range(num_processes):
                    task = (self.json_path[i], shm_objects[i].name)
                    future = executor.submit(self._share_memory_process_func, *task)
                    future_to_task[future] = task
                for future in as_completed(future_to_task):
                    try:
                        shm_name = future.result()
                        existing_shm = shared_memory.SharedMemory(name=shm_name)
                        data_len = int.from_bytes(bytes(existing_shm.buf[:8]), 'big')
                        content = existing_shm.buf[8:8 + data_len]
                        content = bytes(content)
                        total_contents += orjson.loads(content)
                        existing_shm.close()
                    except Exception as error:
                        print(f"Process {future_to_task[future][1]} file failed when using multiprocess: {error}")
        finally:
            # Clean up shared memory to prevent resource leak
            for shm in shm_objects:
                try:
                    shm.close()
                    shm.unlink()
                except Exception as error:
                    print(f"Process {shm.name} file failed when clean shm: {error}")
        return total_contents

    def _share_memory_process_func(self, path, shm_name):
        """Child process function: load single file and write to shared memory"""
        json_content = orjson_load(path)
        print(f"Building {path}...")
        if path in self.process_funcs:
            for fn in self.process_funcs[path]:
                json_content = fn["func"](json_content, *fn["args"], **fn["kwargs"])
        modified_bytes = orjson.dumps(json_content)
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        existing_shm.buf[:8] = len(modified_bytes).to_bytes(8, "big")
        existing_shm.buf[8:len(modified_bytes) + 8] = modified_bytes
        existing_shm.close()
        return shm_name

    def get_data(self):
        """Get loaded JSON data, load if not already loaded"""
        if not self.json_contents:
            self.start_load()
        return self.json_contents


class DecordInit:
    """Using Decord (https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform Decord initialization, modifying ``results`` for the next transform."""
        reader = decord.VideoReader(
            filename, ctx=self.ctx, num_threads=self.num_threads
        )
        return reader

    def __repr__(self):
        repr_str = (
            f"{self.__class__.__name__}("
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
                        use_clean_caption=self.use_clean_caption,
                        support_chinese=self.support_chinese,
                        text_preprocess_methods=self.text_preprocess_methods
                    )
                    for text in texts
                ]
            else:
                texts_info = TextProcesser.text_preprocessing(
                    texts,
                    use_clean_caption=self.use_clean_caption,
                    support_chinese=self.support_chinese,
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


def check_none(value):
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return math.isnan(value) or np.isnan(value)
    return False


def orjson_load(data_path):
    if data_path.endswith(".json"):
        with open(data_path, 'rb') as file:
            content = orjson.loads(file.read())
    elif data_path.endswith(".jsonl"):
        content = []
        with open(data_path, 'rb') as file:
            for line in file:
                if line.strip():
                    content.append(orjson.loads(line))
    return content
