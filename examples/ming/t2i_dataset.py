import os
from PIL import Image

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor

from bailingmm_utils import process_ratio


def collate_fn(samples):
    images = torch.stack([sample["image"] for sample in samples])
    images = images.to(memory_format=torch.contiguous_format).float()
    prompt_ids = torch.stack([sample["prompt_ids"] for sample in samples])
    attn_mask = torch.stack([sample["attn_mask"] for sample in samples])
    return {"images": images, "batched_input_ids": prompt_ids, "batched_attn_mask": attn_mask}


class T2IDataset(Dataset):
    def __init__(
        self,
        args,
        gen_input_pixels=451584
    ):
        self.data_samples = pd.read_json(args.json_path, lines=True)
        self.image_folder = args.image_folder
        self.processor = AutoProcessor.from_pretrained(args.processor_path, trust_remote_code=args.trust_remote_code)
        self.processor.image_processor.max_pixels = gen_input_pixels
        self.processor.image_processor.min_pixels = gen_input_pixels

        closest_size, _ = process_ratio(ori_h=args.resolution[0], ori_w=args.resolution[1])
        resolution = [closest_size[0] * 1, closest_size[1] * 1]
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution) if args.center_crop else transforms.RandomCrop(resolution),
                transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, index):
        image_name, text = self.data_samples["image_name"][index], self.data_samples["text"][index]
        image_path = os.path.join(self.image_folder, image_name)
        prompt_ids, attn_mask = self._preprocess_text(text)
        image = self._preprocess_image(image_path)
        return {"image": image, "prompt_ids": prompt_ids, "attn_mask": attn_mask}


    def _preprocess_text(self, prompt):
        messages = [
            {
                "role": "HUMAN",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }
        ]

        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        prompt_inputs = self.processor(text=[prompt], return_tensors="pt")
        prompt_ids, attn_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        return prompt_ids, attn_mask

    def _preprocess_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()  # check image integrity
                img = Image.open(image_path)
                image = self.train_transforms(img.copy())
                return image
        except Exception as e:
            raise ValueError(f"Unable to open or parse image file: {image_path}") from e
