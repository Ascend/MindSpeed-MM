import math
from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer


class BaseTransform(ABC):
    def set_rng(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass

    @property
    def default_shape(self):
        raise NotImplementedError


def normalize_transform(mean, std):
    if mean is None and std is None:
        transform = None
    elif mean is None and std is not None:
        mean = [0.] * len(std)
        transform = transforms.Normalize(mean=mean, std=std)
    elif mean is not None and std is None:
        std = [1.] * len(mean)
        transform = transforms.Normalize(mean=mean, std=std)
    else:
        transform = transforms.Normalize(mean=mean, std=std)

    return transform


class BasicImageTransform(BaseTransform):
    def __init__(
            self,
            mean: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            std: Optional[Tuple[float, float, float]] = (0.5, 0.5, 0.5),
            normalize: bool = True
    ):
        self.mean = mean
        self.std = std

        transform_pipelines = [
            transforms.ToTensor()
        ]

        normalize = normalize_transform(mean, std) if normalize else torch.nn.Identity()
        if normalize is not None:
            transform_pipelines.append(normalize)

        self.transform = transforms.Compose(transform_pipelines)

    def __call__(self, x):
        x = self.transform(x)
        return x


class OCRDataset(Dataset):
    def __init__(
            self,
            data_path: str = None,
            tokenizer_path: str = None,
            image_size: int = 640,
            base_size: int = 1024,
            patch_size: int = 16,
            downsample_ratio: int = 4,
            cutoff_len: int = 2048,
            repeat_time: int = 5,
            trust_remote_code: bool = True
            ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)
        self.image_size = image_size
        self.base_size = base_size
        self.data = self.load_data(data_path)
        if repeat_time > 1:
            self.data = self.data * repeat_time

        self.valid_img_tokens = 0
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio
        self.image_token = '<image>'
        self.image_token_id = self.tokenizer.vocab.get(self.image_token)
        self.ignore_id = -100
        self.cutoff_len = cutoff_len

    def load_data(self, data_path):
        data_out = pd.read_json(data_path, lines=True)
        return data_out.to_dict("records")

    def process_one(self,
                    image_path: str = None,
                    prompt: str = None,
                    output: str = None):
        image = Image.open(image_path).convert("RGB")

        image_transform = BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True)

        image = image.resize((self.image_size, self.image_size))
        global_view = ImageOps.pad(image, (self.base_size, self.base_size),
                                   color=tuple(int(x * 255) for x in image_transform.mean))

        image_tensor = image_transform(global_view).to(torch.bfloat16)
        image_crop = torch.zeros((1, 3, self.base_size, self.base_size))
        images = torch.stack([image_crop, image_tensor.unsqueeze(0)])

        # no image crop now
        images_spatial_crop = torch.tensor([1, 1])

        num_queries = math.ceil((self.image_size // self.patch_size) / self.downsample_ratio)

        tokenized_image = ([self.image_token_id] * num_queries + [self.image_token_id]) * num_queries

        images_seq_mask, tokenized_str = [], []
        conversation = f"<|User|>: {prompt}<image>\n<|Assistant|>: {output}<|end|>"

        text_splits = conversation.split(self.image_token)
        # before image part
        tokenized_sep = self.tokenizer.encode(text_splits[0], add_special_tokens=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # image part
        tokenized_str += tokenized_image
        images_seq_mask += [True] * len(tokenized_image)

        # <|Assistant|> label part
        tokenized_sep = self.tokenizer.encode(text_splits[-1], add_special_tokens=False)
        tokenized_str += tokenized_sep
        images_seq_mask += [False] * len(tokenized_sep)

        # handle labels
        masked_tokenized_str = []
        for token_index in tokenized_str:
            if token_index != self.image_token_id:
                masked_tokenized_str.append(token_index)
            else:
                masked_tokenized_str.append(self.ignore_id)

        if not (len(tokenized_str) == len(images_seq_mask) == len(masked_tokenized_str)):
            raise AssertionError(f"tokenized_str's length {len(tokenized_str)}, "
                                 f"input_ids' length {len(masked_tokenized_str)}, "
                                 f"images_seq_mask's length {len(images_seq_mask)}, are not equal")

        input_ids = torch.LongTensor(tokenized_str)
        labels = torch.LongTensor(masked_tokenized_str)
        images_seq_mask = torch.tensor(images_seq_mask, dtype=torch.bool)

        labels[(input_ids < 0) | (input_ids == self.image_token_id)] = self.ignore_id
        input_ids[input_ids < 0] = self.tokenizer.pad_token_id

        if len(input_ids) > self.cutoff_len:
            input_ids = input_ids[:self.cutoff_len]
            labels = labels[:self.cutoff_len]
            images_seq_mask = images_seq_mask[:self.cutoff_len]

        return input_ids, labels, images_seq_mask, images_spatial_crop, images

    def __getitem__(self, idx):
        sample = self.data[idx]

        prompt = sample['conversations'][0]['content']
        output = sample['conversations'][1]['content']
        image_path = sample['conversations'][0]['images'][0] # for now, only one image

        input_ids, labels, images_seq_mask, images_spatial_crop, images = self.process_one(image_path, prompt, output)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "images": images,
            "images_seq_mask": images_seq_mask,
            "images_spatial_crop": images_spatial_crop
        }

    def __len__(self):
        return len(self.data)


class DataCollatorForDeepSeekOCR(object):
    def __init__(self, pad_id=2, **kwargs):
        self.pad_id = pad_id
        self.ignore_id = -100

    def __call__(self, sample_list):
        batched_input_ids = [sample['input_ids'] for sample in sample_list]
        batched_labels = [sample['labels'] for sample in sample_list]
        batched_images_seq_mask = [sample['images_seq_mask'] for sample in sample_list]

        batched_input_ids = pad_sequence(batched_input_ids, batch_first=True, padding_value=self.pad_id)
        batched_labels = pad_sequence(batched_labels, batch_first=True, padding_value=self.ignore_id)
        batched_images_seq_mask = pad_sequence(batched_images_seq_mask, batch_first=True, padding_value=0)
        batched_attention_mask = batched_input_ids != self.pad_id

        """padding images to max_patch_num"""
        max_n_patched = max(sample["images"].shape[0] for sample in sample_list)
        batched_images = []
        for sample in sample_list:
            images = sample["images"]
            n_pads = max_n_patched - images.shape[0]
            if n_pads > 0:
                pad_images = torch.zeros((n_pads, *images.shape[1:]), dtype=images.dtype)
                images = torch.cat([images, pad_images], dim=0)
            batched_images.append(images)
        batched_images = torch.stack(batched_images, dim=0)

        """padding images_spatial_crop to max_n_images"""
        max_n_images = max(sample["images_spatial_crop"].shape[0] for sample in sample_list)
        batched_images_spatial_crop = []
        for sample in sample_list:
            images_spatial_crop = sample["images_spatial_crop"]
            n_pads = max_n_images - sample["images_spatial_crop"].shape[0]
            if n_pads > 0:
                pad_images_spatial_crop = torch.full((n_pads, 2), 0, dtype=images_spatial_crop.dtype)
                images_spatial_crop = torch.cat([images_spatial_crop, pad_images_spatial_crop], dim=0)
            batched_images_spatial_crop.append(images_spatial_crop)
        batched_images_spatial_crop = torch.stack(batched_images_spatial_crop, dim=0)

        return {
            "input_ids": batched_input_ids,
            "labels": batched_labels,
            "attention_mask": batched_attention_mask,
            "images": batched_images,
            "images_seq_mask": batched_images_seq_mask,
            "images_spatial_crop": batched_images_spatial_crop
        }


if __name__ == "__main__":
    ocr_dataset = OCRDataset(
        "./data/output.jsonl",
        "./data/ckpt/deepseek-ai/DeepSeek-OCR",
        1024,
        640,
        1024
    )
    ocr_dataset.__getitem__(0)