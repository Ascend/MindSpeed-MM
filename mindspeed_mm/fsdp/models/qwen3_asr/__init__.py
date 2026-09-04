import logging

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


logger = logging.getLogger(__name__)


try:
    AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
except ValueError:
    logger.debug("Qwen3-ASR AutoConfig is already registered.")

try:
    AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
except ValueError:
    logger.debug("Qwen3-ASR AutoModel is already registered.")

try:
    AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)
except ValueError:
    logger.debug("Qwen3-ASR AutoProcessor is already registered.")


__all__ = [
    "Qwen3ASRConfig",
    "Qwen3ASRForConditionalGeneration",
    "Qwen3ASRProcessor",
]
