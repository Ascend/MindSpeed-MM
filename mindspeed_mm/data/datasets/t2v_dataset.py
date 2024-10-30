# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import random
from typing import Union
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

from mindspeed_mm.data.data_utils.constants import (
    CAPTIONS,
    FILE_INFO,
    PROMPT_IDS,
    PROMPT_MASK,
    PROMPT_MASK_2,
    PROMPT_IDS_2,
    TEXT,
    VIDEO,
    IMG_FPS
)
from mindspeed_mm.data.data_utils.utils import (
    VID_EXTENSIONS,
    DataSetProg,
    ImageProcesser,
    TextProcesser,
    VideoProcesser,
    VideoReader
)
from mindspeed_mm.data.datasets.mm_base_dataset import MMBaseDataset
from mindspeed_mm.models import Tokenizer
from mindspeed_mm.data.data_utils.data_transform import (
    MaskGenerator,
    add_aesthetic_notice_image,
    add_aesthetic_notice_video
)


T2VOutputData = {
    VIDEO: [],
    TEXT: [],
    PROMPT_IDS: [],
    PROMPT_MASK: [],
    PROMPT_IDS_2: [],
    PROMPT_MASK_2: [],
}


class T2VDataset(MMBaseDataset):
    """
    A mutilmodal dataset for text-to-video task based on MMBaseDataset

    Args: some parameters from dataset_param_dict in config.
        basic_param(dict): some basic parameters such as data_path, data_folder, etc.
        vid_img_process(dict): some data preprocessing parameters
        use_text_processer(bool): whether text preprocessing
        tokenizer_config(dict): the config of tokenizer
        use_feature_data(bool): use vae feature instead of raw video data or use text feature instead of raw text.
        vid_img_fusion_by_splicing(bool):  videos and images are fused by splicing
        use_img_num(int): the number of fused images
        use_img_from_vid(bool): sampling some images from video
    """

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        use_clean_caption: bool = True,
        support_chinese: bool = False,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        tokenizer_config_2: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.use_feature_data = use_feature_data
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.frame_interval = vid_img_process.get("frame_interval", 1)
        self.resolution = vid_img_process.get("resolution", (256, 256))

        self.max_height = vid_img_process.get("max_height", 480)
        self.max_width = vid_img_process.get("max_width", 640)
        self.max_hxw = vid_img_process.get("max_hxw", None)
        self.min_hxw = vid_img_process.get("min_hxw", None)
        self.train_fps = vid_img_process.get("train_fps", 24)
        self.speed_factor = vid_img_process.get("speed_factor", 1.0)
        self.drop_short_ratio = vid_img_process.get("drop_short_ratio", 1.0)
        self.cfg = vid_img_process.get("cfg", 0.1)
        self.image_processer_type = vid_img_process.get(
            "image_processer_type", "image2video"
        )
        self.hw_stride = vid_img_process.get("hw_stride", 32)
        self.ae_stride_t = vid_img_process.get("ae_stride_t", 32)
        self.force_resolution = vid_img_process.get("force_resolution", True)
        self.sp_size = vid_img_process.get("sp_size", 1)
        self.train_sp_batch_size = vid_img_process.get("train_sp_batch_size", 1)
        self.gradient_accumulation_size = vid_img_process.get("gradient_accumulation_size", 1)
        self.batch_size = vid_img_process.get("batch_size", 1)
        self.seed = vid_img_process.get("seed", 42)
        self.hw_aspect_thr = vid_img_process.get("hw_aspect_thr", 1.5)
        self.min_num_frames = vid_img_process.get("min_num_frames", 29)
        self.use_aesthetic = vid_img_process.get("use_aesthetic", False) 

        max_workers = vid_img_process.get("max_workers", 1)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.timeout = vid_img_process.get("timeout", 60) 
        
        if self.max_hxw is not None and self.min_hxw is None:
            self.min_hxw = self.max_hxw // 4
        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.get("image_reader_type", "torchvision")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)
        self.video_processer = VideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
            data_storage_mode=self.data_storage_mode,
            train_fps=self.train_fps,
            speed_factor=self.speed_factor,
            drop_short_ratio=self.drop_short_ratio,
            max_height=self.max_height,
            max_width=self.max_width,
            max_hxw=self.max_hxw,
            min_hxw=self.min_hxw,
            force_resolution=self.force_resolution,
            seed=self.seed,
            hw_stride=self.hw_stride,
            hw_aspect_thr=self.hw_aspect_thr,
            sp_size=self.sp_size,
            train_sp_batch_size=self.train_sp_batch_size,
            gradient_accumulation_size=self.gradient_accumulation_size,
            batch_size=self.batch_size,
            min_num_frames=self.min_num_frames
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
            image_processer_type=self.image_processer_type,
        )
        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.tokenizer_2 = None
            if tokenizer_config_2 is not None:
                self.tokenizer_2 = Tokenizer(tokenizer_config_2).get_tokenizer()
            self.text_processer = TextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                tokenizer_2=self.tokenizer_2,
                use_clean_caption=use_clean_caption,
                support_chinese=support_chinese,
                cfg=self.cfg,
            )

        if self.data_storage_mode == "combine":
            self.dataset_prog = DataSetProg()
            dataloader_num_workers = vid_img_process.get("dataloader_num_workers", 1)
            self.data_samples, self.sample_num_frames, self.sample_size = (
                self.video_processer.define_frame_index(self.data_samples)
            )
            self.lengths = self.sample_num_frames
            n_elements = len(self.data_samples)
            self.dataset_prog.set_cap_list(
                dataloader_num_workers, self.data_samples, n_elements
            )

    def __getitem__(self, index):
        try:
            future = self.executor.submit(self.getitem, index)
            data = future.result(timeout=self.timeout) 
            return data
        except Exception as e:
            if self.data_storage_mode == "standard":
                path = self.data_samples[index][FILE_INFO]
                print(f"Data {path}: the error is {e}")
            else:
                print(f"the error is {e}")
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))

    def __len__(self):
        return len(self.data_samples)

    def getitem(self, index):
        # init output data
        examples = T2VOutputData
        if self.data_storage_mode == "standard":
            sample = self.data_samples[index]
            if self.use_feature_data:
                raise NotImplementedError("Not support now.")
            else:
                path, texts = sample[FILE_INFO], sample[CAPTIONS]
                if self.data_folder:
                    path = os.path.join(self.data_folder, path)
                examples[TEXT] = texts
                video_value = (
                    self.get_vid_img_fusion(path)
                    if self.vid_img_fusion_by_splicing
                    else self.get_value_from_vid_or_img(path)
                )
                examples[VIDEO] = video_value
                if self.use_text_processer:
                    prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.get_text_processer(texts)# tokenizer, tokenizer_2
                    examples[PROMPT_IDS], examples[PROMPT_MASK] = (
                        prompt_ids,
                        prompt_mask,
                    )
        elif self.data_storage_mode == "combine":
            examples = self.get_merge_data(examples, index)
        else:
            raise NotImplementedError(
                f"Not support now: data_storage_mode={self.data_storage_mode}."
            )
        return examples

    def get_data_from_feature_data(self, sample):
        raise NotImplementedError("Not implemented.")

    def get_merge_data(self, examples, index):
        sample = self.dataset_prog.cap_list[index]
        file_path = sample["path"]
        if not os.path.exists(file_path):
            raise AssertionError(f"file {file_path} do not exist!")
        file_type = self.get_type(file_path)
        if file_type == "video":
            frame_indice = sample["sample_frame_index"]
            vframes, _, is_decord_read = self.video_reader(file_path)
            video = self.video_processer(
                vframes,
                is_decord_read=is_decord_read,
                predefine_num_frames=len(frame_indice),
            )
            examples[VIDEO] = video
        elif file_type == "image":
            image = self.image_processer(file_path)
            examples[VIDEO] = image

        text = sample["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]
        if self.use_aesthetic:
            if sample.get('aesthetic', None) is not None or sample.get('aes', None) is not None:
                aes = sample.get('aesthetic', None) or sample.get('aes', None)
                if file_type == "video":
                    text = [add_aesthetic_notice_video(text[0], aes)]
                elif file_type == "image":
                    text = [add_aesthetic_notice_image(text[0], aes)]
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.get_text_processer(text)# tokenizer, tokenizer_2
        examples[PROMPT_IDS], examples[PROMPT_MASK], examples[PROMPT_IDS_2], examples[PROMPT_MASK_2] = prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2
        return examples

    def get_value_from_vid_or_img(self, path):
        file_type = self.get_type(path)
        if file_type == "video":
            vframes, _, is_decord_read = self.video_reader(path)
            video_value = self.video_processer(vframes, is_decord_read)
        elif file_type == "image":
            video_value = self.image_processer(path)
        return video_value

    def get_vid_img_fusion(self, path):
        vframes, _, is_decord_read = self.video_reader(path)
        video_value = self.video_processer(vframes, is_decord_read)
        if self.use_img_num != 0 and self.use_img_from_vid:
            select_image_idx = np.linspace(
                0, self.num_frames - 1, self.use_img_num, dtype=int
            )
            if self.num_frames < self.use_img_num:
                raise AssertionError(
                    "The num_frames must be larger than the use_img_num."
                )
            images = video_value[:, select_image_idx]  # c, num_img, h, w
            video_value = torch.cat(
                [video_value, images], dim=1
            )  # c, num_frame+num_img, h, w
            return video_value
        elif self.use_img_num != 0 and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        else:
            raise NotImplementedError

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.text_processer(texts)# tokenizer, tokenizer_2
        if self.vid_img_fusion_by_splicing and self.use_img_from_vid:
            prompt_ids = torch.stack(
                [prompt_ids] * (1 + self.use_img_num)
            )  # 1+self.use_img_num, l
            prompt_mask = torch.stack(
                [prompt_mask] * (1 + self.use_img_num)
            )  # 1+self.use_img_num, l
        if self.vid_img_fusion_by_splicing and not self.use_img_from_vid:
            raise NotImplementedError("Not support now.")
        
        return (prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2)


class DynamicVideoTextDataset(MMBaseDataset):
    """
    A mutilmodal dataset for variable text-to-video task based on MMBaseDataset

    Args: some parameters from dataset_param_dict in config.
        basic_param(dict): some basic parameters such as data_path, data_folder, etc.
        vid_img_process(dict): some data preprocessing parameters
        use_text_processer(bool): whether text preprocessing
        tokenizer_config(dict): the config of tokenizer
        use_feature_data(bool): use vae feature instead of raw video data or use text feature instead of raw text.
        vid_img_fusion_by_splicing(bool):  videos and images are fused by splicing
        use_img_num(int): the number of fused images
        use_img_from_vid(bool): sampling some images from video
    """

    def __init__(
        self,
        basic_param: dict,
        vid_img_process: dict,
        use_text_processer: bool = False,
        enable_text_preprocessing: bool = True,
        use_clean_caption: bool = True,
        model_max_length: int = 120,
        tokenizer_config: Union[dict, None] = None,
        use_feature_data: bool = False,
        vid_img_fusion_by_splicing: bool = False,
        use_img_num: int = 0,
        use_img_from_vid: bool = True,
        dummy_text_feature=False,
        **kwargs,
    ):
        super().__init__(**basic_param)
        self.use_text_processer = use_text_processer
        self.use_feature_data = use_feature_data
        self.vid_img_fusion_by_splicing = vid_img_fusion_by_splicing
        self.use_img_num = use_img_num
        self.use_img_from_vid = use_img_from_vid

        self.num_frames = vid_img_process.get("num_frames", 16)
        self.frame_interval = vid_img_process.get("frame_interval", 1)
        self.resolution = vid_img_process.get("resolution", (256, 256))

        self.train_pipeline = vid_img_process.get("train_pipeline", None)
        self.video_reader_type = vid_img_process.get("video_reader_type", "torchvision")
        self.image_reader_type = vid_img_process.get("image_reader_type", "torchvision")
        self.video_reader = VideoReader(video_reader_type=self.video_reader_type)

        self.video_processer = VideoProcesser(
            num_frames=self.num_frames,
            frame_interval=self.frame_interval,
            train_pipeline=self.train_pipeline,
        )
        self.image_processer = ImageProcesser(
            num_frames=self.num_frames,
            train_pipeline=self.train_pipeline,
            image_reader_type=self.image_reader_type,
        )
        
        if "video_mask_ratios" in kwargs:
            self.video_mask_generator = MaskGenerator(kwargs["video_mask_ratios"])
        else:
            self.video_mask_generator = None

        if self.use_text_processer and tokenizer_config is not None:
            self.tokenizer = Tokenizer(tokenizer_config).get_tokenizer()
            self.text_processer = TextProcesser(
                model_max_length=model_max_length,
                tokenizer=self.tokenizer,
                use_clean_caption=use_clean_caption,
                enable_text_preprocessing=enable_text_preprocessing
            )

        self.data_samples["id"] = np.arange(len(self.data_samples))
        self.dummy_text_feature = dummy_text_feature
        self.get_text = "text" in self.data_samples.columns

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]

    def get_value_from_vid_or_img(self, num_frames, video_or_image_path, image_size):
        file_type = self.get_type(video_or_image_path)

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo, _ = self.video_reader(video_or_image_path)
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            video_fps = video_fps // self.frame_interval

            video = self.video_processer(vframes, num_frames=num_frames, frame_interval=self.frame_interval,
                                         image_size=image_size)  # T C H W
        else:
            # loading
            image = pil_loader(video_or_image_path)
            video_fps = IMG_FPS

            # transform
            image = self.image_processer(image)

            # repeat
            video = image.unsqueeze(0)

        return video, video_fps

    def __getitem__(self, index):
        index, num_frames, height, width = [int(val) for val in index.split("-")]
        sample = self.data_samples.iloc[index]
        video_or_image_path = sample["path"]
        if self.data_folder:
            video_or_image_path = os.path.join(self.data_folder, video_or_image_path)
            
        video, video_fps = self.get_value_from_vid_or_img(num_frames, video_or_image_path, image_size=(height, width))
        ar = height / width

        ret = {
            "video": video,
            "video_mask": None,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        
        if self.video_mask_generator is not None:
            ret["video_mask"] = self.video_mask_generator.get_mask(video)

        if self.get_text:
            prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.get_text_processer(sample["text"])# tokenizer, tokenizer_2
            ret["prompt_ids"] = prompt_ids
            ret["prompt_mask"] = prompt_mask

        if self.dummy_text_feature:
            text_len = 50
            ret["prompt_ids"] = torch.zeros((1, text_len, 1152))
            ret["prompt_mask"] = text_len

        return ret

    def get_text_processer(self, texts):
        prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2 = self.text_processer(texts)
        return prompt_ids, prompt_mask, prompt_ids_2, prompt_mask_2

