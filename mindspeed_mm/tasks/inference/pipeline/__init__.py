__all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]

from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from mindspeed_mm.tasks.inference.pipeline.stepvideo_pipeline import StepVideoPipeline
from mindspeed_mm.tasks.inference.pipeline.wan_pipeline import WanPipeline
from mindspeed_mm.tasks.inference.pipeline.hunyuanvideo_15_pipeline import HunyuanVideo15Pipeline

sora_pipeline_dict = {
                      "WanPipeline": WanPipeline,
                      "StepVideoPipeline": StepVideoPipeline,
                      "HunyuanVideo15Pipeline": HunyuanVideo15Pipeline,
                      }

vlm_pipeline_dict = {
    "Qwen2VlPipeline": Qwen2VlPipeline,
}
