# Copyright 2022-2023 XProbe Inc.

from typing import Any, Dict
from PIL import Image

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers.generation.streamers import TextStreamer

from mindspeed_mm.models.text_encoder import Tokenizer
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.data.data_utils.conversation import get_conv_template
from mindspeed_mm.data.data_utils.video_reader import VideoReader
from mindspeed_mm.data.data_utils.multimodal_image_video_preprocess import dynamic_preprocess


def build_infer_transform(input_size):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
    else:
        start_idx, end_idx = first_idx, max_frame
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


class InternVLPipeline(GenerationMixin, InputsCheckMixin, MMEncoderMixin):
    def __init__(self, infer_config) -> None:
        self.infer_config = infer_config
        self.prepare_model(infer_config.device, infer_config.dtype)

        self.image_encoder = self.infer_model.image_encoder

        # prepare for generate
        self.device = infer_config.device
        self.dtype = infer_config.dtype
        self.infer_data_type = infer_config.infer_data_type
        self.model_config = infer_config.text_decoder
        self.template = infer_config.template
        self.text_decoder_model_id = getattr(infer_config.text_decoder, "model_id", None)
        self.generation_config = infer_config.generation_config

        self.main_input_name = "input_ids"
        self.image_size = self.infer_config.image_encoder.vision_encoder.image_size
        self.patch_size = self.infer_config.image_encoder.vision_encoder.patch_size
        self.downsample_ratio = self.infer_config.image_encoder.vision_encoder.downsample_ratio
        self.num_image_token = int((self.image_size // self.patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.num_segments = infer_config.num_segments

        self.model = self.infer_model.text_decoder.eval()
        self.vit_embeds = None

    def prepare_model(self, device, dtype):
        self.tokenizer = Tokenizer(self.infer_config.tokenizer).get_tokenizer()
        from pretrain_internvl import model_provider
        self.infer_model = model_provider()
        model_state_dict = torch.load(self.infer_config.from_pretrained, map_location="cpu")
        self.infer_model.load_state_dict(state_dict=model_state_dict["model"])
        self.infer_model.to(dtype=dtype, device=device).eval()

    def _prepare_images(self, image_path, input_size=448, max_num=12, upscale=False):
        image = Image.open(image_path).convert("RGB")
        if upscale:
            image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
        transform = build_infer_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def _prepare_video(self, video_path, bound=None, input_size=448, max_num=1, num_segments=32):
        video_reader = VideoReader(video_reader_type="DecordVideo")(video_path, layout="THWC", array_type="numpy")
        max_frame = video_reader.get_len() - 1
        fps = float(video_reader.get_video_fps())

        pixel_values_list, num_patches_list = [], []
        transform = build_infer_transform(input_size=input_size)
        frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
        for frame_index in frame_indices:
            img = Image.fromarray(video_reader.get_batch([frame_index])[0]).convert('RGB')
            img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
            pixel_values = [transform(tile) for tile in img]
            pixel_values = torch.stack(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
            pixel_values_list.append(pixel_values)
        pixel_values = torch.cat(pixel_values_list)
        return pixel_values, num_patches_list

    def _prepare_prompts(self, question, num_patches_list=None):
        if self.infer_data_type == "video":
            video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
            question = video_prefix + question
        else:
            question = "<image>\n" + question
        return question

    def prepare_inputs(self, prompt, input_path):
        if input_path:
            if isinstance(input_path, list):
                input_path = input_path[0]
            else:
                input_path = input_path
        else:
            input_path = self.infer_config.file_path

        if not prompt:
            prompt = self.infer_config.prompts
        if self.infer_data_type == "image":
            pixel_values = self._prepare_images(
                image_path=input_path,
                input_size=self.image_size
            )
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
            question = self._prepare_prompts(prompt)
        elif self.infer_data_type == "video":
            pixel_values, num_patches_list = self._prepare_video(
                video_path=input_path,
                input_size=self.image_size,
                num_segments=self.num_segments
            )
            question = self._prepare_prompts(prompt, num_patches_list)
        else:
            raise AssertionError(f"Inference data type must be image or video.")

        pixel_values = pixel_values.to(self.device).to(self.dtype)

        return pixel_values, question, num_patches_list

    def prepare_inputs_for_generation(
            self, input_ids, attention_mask=None, **kwargs
    ):
        B, S = input_ids.shape
        attention_mask = torch.ones(B, S).npu()
        attention_mask = self.infer_model._prepare_decoder_attention_mask(attention_mask)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        cur_input_embeds = self.model.embedding(input_ids, position_ids=position_ids)
        B, N, C = cur_input_embeds.shape
        cur_input_embeds = cur_input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        if selected.sum() == 0:
            raise ValueError("image special token must in input_ids")
        cur_input_embeds[selected] = self.vit_embeds.reshape(-1, C).to(cur_input_embeds.device)
        cur_input_embeds = cur_input_embeds.reshape(B, N, C)

        model_inputs = {
            "decoder_input": cur_input_embeds,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }
        return model_inputs

    def _update_model_kwargs_for_generation(self, model_kwargs: Dict[str, Any], model_inputs: Dict[str, Any]):
        # update position_ids
        if "position_ids" in model_kwargs:
            model_kwargs["position_ids"] = model_inputs["position_ids"]

        return model_kwargs

    @torch.no_grad()
    def _inference(
            self,
            input_ids,
            pixel_values=None,
            attention_mask=None,
            visual_features=None,
            return_ids=False,

    ) -> torch.LongTensor:
        if self.img_context_token_id is None:
            raise ValueError("img_context_token_id cannot be None")
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.image_encoder(pixel_values)
        if return_ids:
            streamer = None
        else:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.vit_embeds = vit_embeds
        outputs = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
            streamer=streamer)
        return outputs

    def __call__(self, prompt=None, input_path=None, return_ids=False):
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        pixel_values, question, num_patches_list = self.prepare_inputs(prompt=prompt, input_path=input_path)

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        template = get_conv_template(self.template)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].npu()
        attention_mask = model_inputs['attention_mask'].npu()
        generation_output = self._inference(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            return_ids=return_ids

        )

        if return_ids and generation_output is not None:
            answer_len = generation_output.shape[-1] - input_ids.shape[-1]
            response = self.tokenizer.batch_decode(generation_output[:, -answer_len:], skip_special_tokens=True)[0]
            response = response.split(template.sep)[0].strip()

            return response
        else:
            return None

    def evaluate(self, message, dataset=None):
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"

        if dataset in ['chartqa_test', 'mmmu_dev_val', 'mmmu_test']:
            self.max_num = 12
        elif dataset in ['docvqa_val', 'docvqa_test']:
            self.max_num = 18
        else:
            self.max_num = 6

        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt, image_idx = '', 1
            for x in message:
                if x['type'] == 'text':
                    prompt += x['value']
                elif x['type'] == 'image':
                    prompt += f'<Image-{image_idx}>'
                    image_idx += 1
            prompt = '\n'.join([f'Image-{i + 1}: <image>' for i in range(image_num)]) + '\n' + prompt

        question = prompt
        if image_num > 1:
            image_path = [x['value'] for x in message if x['type'] == 'image']
            num_patches_list = []
            pixel_values_list = []
            for image_idx, file_name in enumerate(image_path):
                upscale_flag = image_idx == 0 and dataset is not None and dataset == 'mmu_dev_val'
                curr_pixel_values = self._prepare_images(
                    file_name, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
                num_patches_list.append(curr_pixel_values.size(0))
                pixel_values_list.append(curr_pixel_values)
            pixel_values = torch.cat(pixel_values_list, dim=0)
        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            upscale_flag = dataset == 'mmu_dev_val'
            pixel_values = self._prepare_images(
                image_path, max_num=self.max_num, upscale=upscale_flag).to(self.device).to(torch.bfloat16)
            num_patches_list = [pixel_values.size(0)]
        else:
            pixel_values = None
            num_patches_list = []

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        template = get_conv_template(self.template)
        eos_token_id = self.tokenizer.convert_tokens_to_ids(template.sep)

        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = self.tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        self.init_input_ids = input_ids
        attention_mask = model_inputs['attention_mask'].to(self.device)

        self.generation_config.eos_token_id = eos_token_id

        if pixel_values is not None:
            vit_embeds = self.image_encoder(pixel_values)
        self.vit_embeds = vit_embeds
        generation_output = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=self.generation_config,
        )
        if generation_output is not None:
            answer_len = generation_output.shape[-1] - input_ids.shape[-1]
            response = self.tokenizer.batch_decode(generation_output[:, -answer_len:], skip_special_tokens=True)[0]
            response = response.split(template.sep)[0].strip()

            return response
        else:
            return None
