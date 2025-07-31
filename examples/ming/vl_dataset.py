import os
import json
import tqdm

import torch
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq, AutoProcessor

from processing_bailingmm import USER_PREFIX, ASSISTANT_PREFIX
from bailingmm_utils import fetch_image


IMAGES_KEY = "images"
MESSAGES_TAG = "messages"
ROLE_TAG = "role"
CONTENT_TAG = "content"
USER_TAG = "user"
ASSISTANT_TAG = "assistant"
IGNORE_INDEX = -100
IMAGE_PLACEHOLDER = "<image>"


def find_medias(medias, dataset_dir):
    if medias is None:
        return None
    elif not isinstance(medias, list):
        medias = [medias]
    elif len(medias) == 0:
        return None
    else:
        medias = medias[:]
    for i, media in enumerate(medias):
        if os.path.isfile(os.path.join(dataset_dir, media)):
            medias[i] = os.path.join(dataset_dir, media)
        else:
            print(f"Media {media} does not exist in `media_dir`. Use original path.")
    return medias


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int):
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


class VLDataset(Dataset):

    def __init__(self, data_path, data_dir, processor_path=".", cutoff_len=2048, trust_remote_code=False):
        super().__init__()
        self.cutoff_len = cutoff_len
        self.processor = AutoProcessor.from_pretrained(
            processor_path, trust_remote_code=trust_remote_code)
        self.dataset = self.load_dataset(data_path, data_dir)

    def load_dataset(self, data_path, data_dir):
        with open(data_path, 'r', encoding='utf-8') as fr:
            raw_dataset = json.load(fr)
        dataset = []
        for example in tqdm.tqdm(raw_dataset):
            dataset.append(self.process_example(example, data_dir))
        return dataset

    def process_example(self, example, data_dir, mask_history=True):
        images = find_medias(example[IMAGES_KEY], data_dir)
        for i, image in enumerate(images):
            images[i] = fetch_image({"type": "image", "image": image})
        messages = example.get(MESSAGES_TAG, [])
        if messages is None or len(messages) < 2:
            raise ValueError("messages is invalid!")
        image_inputs = self.processor.image_processor(images=images, videos=None, return_tensors='pt')

        for image in images:
            image.close() # free image resource
        image_grid_thw = image_inputs["image_grid_thw"]

        encoded_messages = []
        for message in messages:
            prefix = USER_PREFIX if message[ROLE_TAG] == USER_TAG else ASSISTANT_PREFIX
            content = message[CONTENT_TAG]
            if IMAGE_PLACEHOLDER in content:
                special_token = "<IMAGE>"  # replace by a special token
                content = content.replace(IMAGE_PLACEHOLDER, special_token)
                content = self.processor._expand_image_tokens([content], image_grid_thw, special_token=special_token)[0]
            encoded_messages.append(self.processor.tokenizer.encode(prefix + content))
        message_pairs = [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]
        if mask_history:
            message_pairs = message_pairs[::-1]  # high priority for last turns
        total_length = 0
        input_ids = []
        labels = []
        for turn_idx, (source_ids, target_ids) in enumerate(message_pairs):
            if total_length >= self.cutoff_len:
                break
            source_len, target_len = infer_seqlen(len(source_ids), len(target_ids), self.cutoff_len - total_length)
            source_ids = source_ids[: source_len]
            target_ids = target_ids[: target_len]
            total_length += source_len + target_len

            source_label = source_ids
            if mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label
        return {
            "labels": labels,
            "input_ids": input_ids,
            "pixel_values": image_inputs["pixel_values"],
            "image_grid_thw": image_grid_thw
        }

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    @property
    def tokenizer(self):
        return self.processor.tokenizer


class MultiModalDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    def __call__(self, features):
        batch_images = []
        batch_thw = []
        for feature in features:
            images = feature.pop("pixel_values", None)
            thw = feature.pop("image_grid_thw", None)
            batch_images.append(images)
            batch_thw.append(thw)
        image_features = {
            "pixel_values": torch.cat(batch_images),
            "image_grid_thw": torch.cat(batch_thw)
        }
        features = super().__call__(features)
        features.update(image_features)
        return features