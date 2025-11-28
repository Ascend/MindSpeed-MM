from mindspeed_mm.tasks.evaluation.eval_impl.impl_mmmu import MMMUEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_vqa import VQAEvalImpl
from mindspeed_mm.tasks.evaluation.eval_impl.impl_ai2d import AI2DEvalImpl

from mindspeed_mm.tasks.inference.pipeline import vlm_pipeline_dict, sora_pipeline_dict

eval_impl_dict = {"mmmu_dev_val": MMMUEvalImpl, "ai2d_test": AI2DEvalImpl, "chartqa_test": VQAEvalImpl,
                  "docvqa_val": VQAEvalImpl}

eval_pipeline_dict = {"internvl2_8b": vlm_pipeline_dict["InternVLPipeline"],
                      "qwen2_vl_7b": vlm_pipeline_dict["Qwen2VlPipeline"]
                      }
