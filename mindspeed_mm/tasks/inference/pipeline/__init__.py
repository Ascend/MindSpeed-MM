__all__ = ["sora_pipeline_dict", "vlm_pipeline_dict"]

from mindspeed_mm.tasks.inference.pipeline.opensora_pipeline import OpenSoraPipeline
from mindspeed_mm.tasks.inference.pipeline.opensoraplan_pipeline import OpenSoraPlanPipeline
from mindspeed_mm.tasks.inference.pipeline.cogvideox_pipeline import CogVideoXPipeline
from mindspeed_mm.tasks.inference.pipeline.internvl_pipeline import InternVLPipeline
from mindspeed_mm.tasks.inference.pipeline.qihoo_pipeline import QihooPipeline
from mindspeed_mm.tasks.inference.pipeline.qwen2vl_pipeline import Qwen2VlPipeline
from mindspeed_mm.tasks.inference.pipeline.hunyuanvideo_pipeline import HunyuanVideoPipeline
from mindspeed_mm.tasks.inference.pipeline.wan_pipeline import WanPipeline
from mindspeed_mm.tasks.inference.pipeline.stepvideo_pipeline import StepVideoPipeline
from mindspeed_mm.tasks.inference.pipeline.glm_pipeline import GlmPipeline
from mindspeed_mm.tasks.inference.pipeline.videoalign_pipeline import VideoAlignPipeline

sora_pipeline_dict = {"OpenSoraPlanPipeline": OpenSoraPlanPipeline,
                      "OpenSoraPipeline": OpenSoraPipeline,
                      "QihooPipeline": QihooPipeline,
                      "CogVideoXPipeline": CogVideoXPipeline,
                      "HunyuanVideoPipeline": HunyuanVideoPipeline,
                      "WanPipeline": WanPipeline,
                      "StepVideoPipeline": StepVideoPipeline
}

vlm_pipeline_dict = {
    "InternVLPipeline": InternVLPipeline,
    "Qwen2VlPipeline": Qwen2VlPipeline,
    "GlmPipeline": GlmPipeline
}

