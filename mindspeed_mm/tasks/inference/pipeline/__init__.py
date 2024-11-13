from mindspeed_mm.tasks.inference.pipeline.opensora_pipeline import OpenSoraPipeline
from mindspeed_mm.tasks.inference.pipeline.opensoraplan_pipeline import OpenSoraPlanPipeline
from mindspeed_mm.tasks.inference.pipeline.cogvideox_pipeline import CogVideoXPipeline
from mindspeed_mm.tasks.inference.pipeline.internvl_pipeline import InternVLPipeline
from mindspeed_mm.tasks.inference.pipeline.llava_pipeline import LlavaPipeline

SoraPipeline_dict = {"OpenSoraPlanPipeline": OpenSoraPlanPipeline,
                     "OpenSoraPipeline": OpenSoraPipeline,
                     "CogVideoXPipeline": CogVideoXPipeline}

VlmPipeline_dict = {
    "InternVLPipeline": InternVLPipeline,
    "LlavaPipeline": LlavaPipeline
}

__all__ = ["SoraPipeline_dict"]
