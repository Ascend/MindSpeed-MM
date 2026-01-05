from .build_prompt_internvl import InternvlPromptTemplate
from .build_prompt_qwen2vl import Qwen2vlPromptTemplate

eval_model_prompt_dict = {
    "internvl2_8b": InternvlPromptTemplate,
    "qwen2_vl_7b": Qwen2vlPromptTemplate
    }