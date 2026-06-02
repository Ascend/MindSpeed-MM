from typing import Any, Dict

import torch
from PIL import Image
from transformers import StoppingCriteria
from transformers.generation.streamers import TextStreamer
from transformers import CLIPImageProcessor

from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.encode_mixin import MMEncoderMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.inputs_checks_mixin import InputsCheckMixin
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.data.data_utils.constants import MODEL_CONSTANTS
from mindspeed_mm.models.text_encoder import Tokenizer


class LlavaPipeline(GenerationMixin, InputsCheckMixin, MMEncoderMixin):

    def __init__(self, infer_config):

        self.infer_config = infer_config
        self.tokenizer = Tokenizer(infer_config.tokenizer).get_tokenizer()
        self.tokenizer.add_tokens([MODEL_CONSTANTS["llava"]["IMAGE_PATCH_TOKEN"]], special_tokens=True)
        self.image_processor = CLIPImageProcessor.from_pretrained(infer_config.image_processer_path, local_files_only=True)
        from pretrain_llava import model_provider
        self.vlmodel = model_provider()
        self.device = infer_config.device
        self.dtype = infer_config.dtype
        self.vlmodel.to(self.device, dtype=self.dtype)
        self.model = self.vlmodel.text_decoder
        self.model_config = self.vlmodel.config
        self.generation_config = infer_config.generation_config
        self.main_input_name = 'input_ids'

        self.system_prompt = "A chat between a curious human and an artificial intelligence assistant. " \
                             "The assistant gives helpful, detailed, and polite answers to the human's questions. "

    def __call__(self, prompt=None, images=None, input_ids=None, image_tensor=None, return_ids=False, device="npu",
                 stopping_criteria=None, dtype=torch.float16):

        if input_ids is None:
            if not prompt:
                prompt = self.infer_config.prompts
            prompt = self.format_prompt(prompt, mm_use_im_start_end=False)
            roles = [["Human", prompt], ["Assistant", None]]
            prompt = self.prompt_template(self.system_prompt, roles, sep="###")
            input_ids = self.tokenizer_image_token(prompt, MODEL_CONSTANTS["llava"]["IMAGE_TOKEN_INDEX"],
                                                   return_tensors='pt').unsqueeze(0).to(device)
        else:
            if prompt and (input_ids is not None):
                raise ValueError("Both 'prompt' and 'input_ids' cannot be set at the same time.")
            else:
                input_ids = input_ids

        if image_tensor is None:
            if not images:
                images = [self.infer_config.image_path]
            image = [Image.open(s).convert('RGB') for s in images]
            image_tensor = self.process_images(image, image_aspect_ratio="pad")
            if isinstance(image_tensor, list):
                image_tensor = [image.to(device, dtype=dtype) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(device, dtype=dtype)
        else:
            if images and (image_tensor is not None):
                raise ValueError("Both 'image' and 'image_tensor' cannot be set at the same time.")
            else:
                image_tensor = image_tensor

        if return_ids:
            streamer = None
        else:
            streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        attention_mask = torch.ones(input_ids.shape).bool().to(device)
        (inputs,
         position_ids,
         attention_mask,
         _,
         inputs_embeds,
         _) = self.vlmodel.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=None,
            images=image_tensor,
        )

        causal_attention_mask = torch.triu(
            torch.ones(inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[1],
                       device=inputs_embeds.device),
            diagonal=1
        ).bool()
        attention_mask = ~attention_mask
        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            inputs_embeds.shape[0], 1, inputs_embeds.shape[1], inputs_embeds.shape[1]
        )
        attention_mask = causal_attention_mask.masked_fill(expanded_attention_mask, True)

        inputs_embeds = inputs_embeds.transpose(0, 1)

        generation_output = self.generate(position_ids=position_ids,
                                          attention_mask=attention_mask,
                                          decoder_input=inputs_embeds,
                                          do_sample=True if self.generation_config.temperature > 0 else False,
                                          temperature=self.generation_config.temperature,
                                          max_new_tokens=self.generation_config.max_new_tokens,
                                          streamer=streamer,
                                          stopping_criteria=stopping_criteria,
                                          )

        if return_ids and generation_output is not None:
            response = self.tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0].strip()
            return response
        else:
            return None

    def evaluate(self, message, device="npu"):
        text, images = '', []
        for item in message:
            if item['type'] == 'text':
                text += item['value']
            elif item['type'] == 'image':
                text += ' <image> '
                images.append(item['value'])

        prompt = self.system_prompt + 'USER: ' + text + ' ASSISTANT: '
        input_ids = self.tokenizer_image_token(prompt).unsqueeze(0).to(device)
        stopping_criteria = KeywordsStoppingCriteria(['</s>'], self.tokenizer, input_ids)
        output = self(input_ids=input_ids, images=images, stopping_criteria=[stopping_criteria], return_ids=True,
                      device=device)
        if output is not None:
            return output
        else:
            return None

    def tokenizer_image_token(self, prompt, image_token_index=MODEL_CONSTANTS["llava"]["IMAGE_TOKEN_INDEX"],
                              return_tensors="pt"):
        prompt_chunks = [self.tokenizer(chunk).input_ids
                         for chunk in prompt.split(MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"])
        ]

        def insert_separator(x, sep):
            return [ele for sublist in zip(x, [sep] * len(x)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return torch.tensor(input_ids, dtype=torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids

    @staticmethod
    def format_prompt(prompt, mm_use_im_start_end):
        if mm_use_im_start_end:
            prompt = MODEL_CONSTANTS["llava"]["IMG_START_TOKEN"] + MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"] + \
                     MODEL_CONSTANTS["llava"]["IMG_END_TOKEN"] + '\n' + prompt
        else:
            prompt = MODEL_CONSTANTS["llava"]["IMAGE_TOKEN"] + '\n' + prompt
        return prompt

    @staticmethod
    def prompt_template(system: str, roles_prompts: list, sep: str):

        ret = system + sep
        for role, message in roles_prompts:
            if message:
                ret += role + ": " + message + sep
            else:
                ret += role + ":"
        return ret

    def prepare_inputs_for_generation(self, **kwargs):
        """
        Get the model output tokens and generate the model input kwargs.

        Args:
            **kwargs:

        Returns:

        """
        input_ids = kwargs.pop("input_ids")
        if input_ids.shape[-1] > 1:
            cur_inputs_embeds = self.model.embedding.word_embeddings(input_ids[-1][-1]).unsqueeze(0).unsqueeze(0)
            kwargs["input_ids"] = input_ids
            kwargs["attention_mask"] = self.generate_inverted_triangle_mask(kwargs["attention_mask"].shape[-1] + 1,
                                                                            cur_inputs_embeds.device).unsqueeze(0).unsqueeze(0)
            kwargs["decoder_input"] = torch.cat([kwargs["decoder_input"], cur_inputs_embeds], dim=0)
            kwargs["position_ids"] = None

        kwargs_dict = {"input_ids": input_ids,
                       "attention_mask": kwargs["attention_mask"],
                       "decoder_input": kwargs["decoder_input"],
                       "position_ids": None}

        return kwargs_dict

    def _update_model_kwargs_for_generation(self, model_kwargs: Dict[str, Any], model_inputs: Dict[str, Any]):
        model_kwargs["attention_mask"] = model_inputs["attention_mask"]
        model_kwargs["decoder_input"] = model_inputs["decoder_input"]

        return model_kwargs

    @staticmethod
    def generate_inverted_triangle_mask(size, device):
        """
        Generates a lower triangular mask with boolean values.
        :param size: The size of the mask (size x size).
        :return: The lower triangular mask (torch.BoolTensor).
        """
        # Generate an upper triangular index matrix
        indices = torch.arange(size).unsqueeze(0)  # shape: (1, size)

        # Create a boolean mask: the upper triangular part is True, the rest is False
        mask = indices > torch.arange(size).unsqueeze(1)  # shape: (size, size)

        return mask.to(device)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
