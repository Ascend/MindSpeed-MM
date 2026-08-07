from __future__ import annotations

import copy
import importlib
import logging
import os
import random
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS

from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    FILE_REJECTED_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    SCORE,
    SCORE_REJECTED,
    TEXT,
    VIDEO,
    VIDEO_MASK,
    VIDEO_REJECTED,
)
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.data_transform import (
    add_aesthetic_notice_image,
    add_aesthetic_notice_video,
)
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.utils import (
    DataFileReader,
    ImageProcesser,
    TextProcesser,
    VID_EXTENSIONS,
)
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.video_processor import VideoProcessor
from mindspeed_mm.fsdp.data.data_utils.generative_data_utils.video_reader import VideoReader
from mindspeed_mm.fsdp.utils.register import data_register


logger = logging.getLogger(__name__)

T2V_OUTPUT_DATA = {
    VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
}

_NON_VIDEO_PROCESS_KEYS = {
    "model_name_or_path",
    "use_fast_tokenizer",
    "split_special_tokens",
    "image_resolution",
    "video_resolution",
    "image_max_pixels",
    "image_min_pixels",
    "video_max_pixels",
    "video_min_pixels",
    "video_fps",
    "video_maxlen",
    "image_do_pan_and_scan",
    "crop_to_patches",
    "use_audio_in_video",
    "audio_sampling_rate",
    "trust_remote_code",
}


def _build_hf_tokenizer(config):
    """Build tokenizer without importing the legacy Megatron text_encoder package."""
    if isinstance(config, list):
        return [_build_hf_tokenizer(item) for item in config]
    if not isinstance(config, dict):
        config = config.to_dict()

    cfg = dict(config)
    backend = cfg.pop("hub_backend", "hf")
    if backend not in ("hf", "huggingface"):
        raise ValueError(f"Wan2.2 FSDP dataset only supports HF tokenizer backend, got {backend}.")
    tokenizer_name = cfg.pop("autotokenizer_name", "AutoTokenizer")
    pretrained_path = cfg.pop("from_pretrained", None)
    if pretrained_path is None:
        raise ValueError("tokenizer_config missing required key 'from_pretrained'.")
    cfg["pretrained_model_name_or_path"] = pretrained_path
    cfg.setdefault("local_files_only", True)
    cfg.setdefault("trust_remote_code", False)

    module = importlib.import_module("transformers")
    try:
        tokenizer_cls = getattr(module, tokenizer_name)
    except AttributeError as e:
        raise ValueError(f"transformers has no tokenizer class {tokenizer_name!r}.") from e
    return tokenizer_cls.from_pretrained(**cfg)


class MMBaseDataset(Dataset):
    """Base multimodal dataset providing basic parameters and methods; parameters come from ``dataset_param_dict`` in config."""

    def __init__(
        self,
        data_path: str = "",
        data_folder: str = "",
        return_type: str = "list",
        data_storage_mode: str = "standard",
        **kwargs,
    ):
        self.data_path = data_path
        self.data_folder = data_folder
        self.data_storage_mode = data_storage_mode
        self.get_data = DataFileReader(data_storage_mode=data_storage_mode, **kwargs)
        self.data_samples = self.get_data(self.data_path, return_type=return_type)

    def __len__(self):
        return len(self.data_samples)

    # must be reimplemented in the subclass
    def __getitem__(self, index):
        raise AssertionError("__getitem__() in dataset is required.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        elif ext.lower() in IMG_EXTENSIONS:
            return "image"
        else:
            raise NotImplementedError(f"Unsupported file format: {ext}")


@data_register.register("wan2_2")
class Wan2_2Dataset(MMBaseDataset):
    """Wan2.2 dataset adapter for the FSDP2 trainer."""

    def __init__(
        self,
        basic_param: dict,
        preprocess_param: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        text_preprocess_methods: Optional[Union[dict, List[dict]]] = None,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        tokenizer_config: Optional[Union[dict, List[dict]]] = None,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        dataset_param: Optional[dict] = None,
        **kwargs,
    ):
        if dataset_param is not None:
            if not isinstance(dataset_param, dict):
                dataset_param = dataset_param.to_dict() if hasattr(dataset_param, "to_dict") else {}
            use_text_processer = dataset_param.get("use_text_processer", use_text_processer)
            enable_text_preprocessing = dataset_param.get("enable_text_preprocessing", enable_text_preprocessing)
            text_preprocess_methods = dataset_param.get("text_preprocess_methods", text_preprocess_methods)
            use_clean_caption = dataset_param.get("use_clean_caption", use_clean_caption)
            support_chinese = dataset_param.get("support_chinese", support_chinese)
            tokenizer_config = dataset_param.get("tokenizer_config", tokenizer_config)

        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.enable_text_preprocessing = enable_text_preprocessing
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid
        if self.vid_img_fusion_by_splicing and self.use_img_num <= 0:
            raise ValueError("vid_img_fusion_by_splicing=True requires use_img_num > 0.")
        if self.use_text_processer and tokenizer_config is None:
            raise ValueError("use_text_processer=True requires tokenizer_config.")

        vid_img_process = dict(preprocess_param)
        self.cfg = vid_img_process.pop("cfg", 0.1)
        self.image_processer_type = vid_img_process.pop("image_processer_type", "image2video")
        self.use_aesthetic = vid_img_process.pop("use_aesthetic", False)
        self.video_reader_type = vid_img_process.pop("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.pop("image_reader_type", "torchvision")

        if self.video_reader_type in ("DecordVideo", "decord"):
            try:
                import decord  # noqa: F401
            except (ImportError, RuntimeError):
                self.video_reader_type = "TorchvisionVideo"
                logger.warning("decord is unavailable; falling back to TorchvisionVideo.")

        for key in _NON_VIDEO_PROCESS_KEYS:
            vid_img_process.pop(key, None)

        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcessor.create(**vid_img_process)

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.max_height = vid_img_process.get("max_height", 480)
        self.max_width = vid_img_process.get("max_width", 640)
        self.max_hxw = vid_img_process.get("max_hxw", None)
        self.min_hxw = vid_img_process.get("min_hxw", None)
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        transform_size = {
            "max_height": self.max_height,
            "max_width": self.max_width,
            "max_hxw": self.max_hxw,
            "min_hxw": self.min_hxw,
        }

        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            image_processer_type=self.image_processer_type,
            transform_size=transform_size,
        )

        if self.use_text_processer:
            self.tokenizer = _build_hf_tokenizer(tokenizer_config)
            self.text_processer = TextProcesser(
                tokenizer=self.tokenizer,
                enable_text_preprocessing=self.enable_text_preprocessing,
                text_preprocess_methods=text_preprocess_methods,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
            )

        self.data_samples = self.video_processer.select_valid_data(self.data_samples)

    def __getitem__(self, index, _retry_count=0):
        try:
            return self.getitem(index)
        except Exception as exc:
            if _retry_count >= 10:
                raise RuntimeError(f"Failed to load data after 10 retries. Last error: {exc}") from exc
            if self.data_storage_mode == "standard":
                path = self.data_samples[index][FILE_INFO]
                logger.warning("Failed to load sample %s: %s", path, exc)
            else:
                logger.warning("Failed to load sample: %s", exc)
            if self.__len__() <= 1:
                raise
            return self.__getitem__(np.random.randint(0, self.__len__()), _retry_count=_retry_count + 1)

    def getitem(self, index):
        examples = copy.deepcopy(T2V_OUTPUT_DATA)

        if self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            file_path, texts = sample[FILE_INFO], sample[CAPTIONS]
            if self.data_folder:
                file_path = os.path.join(self.data_folder, file_path)
        elif self.data_storage_mode == "combine":
            sample = self.data_samples[index]
            file_path = sample["path"]
            texts = sample["cap"]
        else:
            raise NotImplementedError(f"Unsupported data_storage_mode={self.data_storage_mode}.")

        file_type = self.get_type(file_path)
        if file_type == "image":
            video_value = self.image_processer(file_path)
        elif file_type == "video":
            vframes = self.video_reader(file_path)
            video_value = self.video_processer(vframes=vframes, **sample)
            examples["first_frame"] = video_value[:, 0, :, :]
            if self.vid_img_fusion_by_splicing:
                video_value = self.get_vid_img_fusion(video_value)
        examples[VIDEO] = video_value

        if isinstance(texts, (list, tuple)) and len(texts) > 1:
            texts = random.choice(texts)

        if self.use_aesthetic:
            aes = sample.get("aesthetic") or sample.get("aes")
            if aes is not None:
                if file_type == "video":
                    texts = add_aesthetic_notice_video(texts, aes)
                elif file_type == "image":
                    texts = add_aesthetic_notice_image(texts, aes)

        if self.use_text_processer:
            prompt_ids, prompt_mask = self.get_text_processer(texts)
            examples[PROMPT_IDS], examples[PROMPT_MASK] = prompt_ids, prompt_mask

        if FILE_REJECTED_INFO in sample.keys():
            rejected_video_path = os.path.join(self.data_folder, sample[FILE_REJECTED_INFO])
            rejected_file_type = self.get_type(rejected_video_path)
            if rejected_file_type == "image":
                rejected_video_value = self.image_processer(rejected_video_path)
            elif rejected_file_type == "video":
                rejected_vframes = self.video_reader(rejected_video_path)
                rejected_video_value = self.video_processer(vframes=rejected_vframes, **sample)
                if self.vid_img_fusion_by_splicing:
                    rejected_video_value = self.get_vid_img_fusion(rejected_video_value)
            examples[VIDEO_REJECTED] = rejected_video_value
            examples[SCORE] = sample[SCORE]
            examples[SCORE_REJECTED] = sample[SCORE_REJECTED]

        examples[FILE_INFO] = file_path
        return examples

    def get_vid_img_fusion(self, video_value):
        if self.use_img_num != 0 and self.use_img_from_vid:
            if self.num_frames < self.use_img_num:
                raise AssertionError("num_frames must be greater than or equal to use_img_num.")
            select_image_idx = np.linspace(0, self.num_frames - 1, self.use_img_num, dtype=int)
            images = video_value[:, select_image_idx]
            return torch.cat([video_value, images], dim=1)
        if self.use_img_num != 0 and not self.use_img_from_vid:
            raise NotImplementedError("Image fusion from external images is not supported.")
        raise NotImplementedError("Video-image fusion requires use_img_num > 0.")

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask = self.text_processer(texts)
        if isinstance(prompt_ids, torch.Tensor) and prompt_ids.dim() > 1 and prompt_ids.shape[0] == 1:
            prompt_ids = prompt_ids.squeeze(0)
        if isinstance(prompt_mask, torch.Tensor) and prompt_mask.dim() > 1 and prompt_mask.shape[0] == 1:
            prompt_mask = prompt_mask.squeeze(0)

        if self.vid_img_fusion_by_splicing and self.use_img_from_vid:
            if not isinstance(prompt_ids, list):
                prompt_ids = torch.stack([prompt_ids] * (1 + self.use_img_num))
                prompt_mask = torch.stack([prompt_mask] * (1 + self.use_img_num))
            else:
                prompt_ids = [
                    torch.stack([_prompt_ids] * (1 + self.use_img_num))
                    for _prompt_ids in prompt_ids
                ]
                prompt_mask = [
                    torch.stack([_prompt_mask] * (1 + self.use_img_num))
                    for _prompt_mask in prompt_mask
                ]
        if self.vid_img_fusion_by_splicing and not self.use_img_from_vid:
            raise NotImplementedError("Image fusion from external images is not supported.")
        return prompt_ids, prompt_mask

    def collate_fn(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {}
        for key in features[0].keys():
            values = [feature[key] for feature in features]
            if isinstance(values[0], torch.Tensor):
                if key == VIDEO:
                    batch["video"] = torch.stack(values, dim=0)
                elif key == PROMPT_IDS:
                    batch["prompt_ids"] = torch.stack(values, dim=0)
                elif key == PROMPT_MASK:
                    batch["prompt_mask"] = torch.stack(values, dim=0)
                elif key == VIDEO_MASK:
                    batch["video_mask"] = torch.stack(values, dim=0)
                else:
                    batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values
        return batch
