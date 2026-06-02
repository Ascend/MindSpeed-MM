from transformers import AutoProcessor
from mindspeed_mm.tasks.inference.pipeline.pipeline_mixin.generation_mixin import GenerationMixin
from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from mindspeed_mm.models.vision.vlm_attentionmask_for_llm import glm_position


class GlmPipeline(Qwen2VlPipeline, GenerationMixin):

    def __init__(self, infer_config):
        super().__init__(infer_config)
        self.image_processor = AutoProcessor.from_pretrained(infer_config.tokenizer.from_pretrained,
                                        local_files_only=True, use_fast=True)

    def get_rope_index(self, input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None):
        return glm_position(
            config=self.generation_config,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask
        )

    def __call__(self, prompt=None, images=None, videos=None, return_ids=False):
        return super().__call__(prompt, images, videos, return_ids, skip_special_tokens=False)

    def prepare_inputs(self, prompt=None, images=None, videos=None, messages=None):
        if not images and not messages and not videos:
            return None

        if not messages:
            content = []
            if images:
                content.append({"type": "image", "image": images})
            if videos:
                content.append({"type": "video", "video": videos})
            content.append({"type": "text", "text": prompt})
            messages = [[
                {
                    "role": "user",
                    "content": content,
                }
            ]]

        inputs = self.image_processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", padding=True)

        inputs = inputs.to(self.infer_config.device)
        return inputs
