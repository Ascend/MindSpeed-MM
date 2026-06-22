import logging

from transformers import AutoConfig, AutoModel, AutoProcessor

from .configuration_qwen3_asr import Qwen3ASRConfig
from .modeling_qwen3_asr import Qwen3ASRForConditionalGeneration
from .processing_qwen3_asr import Qwen3ASRProcessor


logger = logging.getLogger(__name__)


def _patch_qwen3_asr_optimizer_builder():
    import sys

    def patch_module_build_optimizer(module):
        build_optimizer = getattr(module, "build_optimizer", None)
        if build_optimizer is None or getattr(build_optimizer, "_qwen3_asr_patched", False):
            return

        def build_qwen3_asr_optimizer(*args, **kwargs):
            optimizer = build_optimizer(*args, **kwargs)
            model = kwargs.get("model")
            patch_optimizer = getattr(model, "_mindspeed_mm_patch_optimizer", None)
            if isinstance(model, Qwen3ASRForConditionalGeneration) and callable(patch_optimizer):
                return patch_optimizer(optimizer, kwargs)
            return optimizer

        build_qwen3_asr_optimizer._qwen3_asr_patched = True
        build_qwen3_asr_optimizer._qwen3_asr_original = build_optimizer
        module.build_optimizer = build_qwen3_asr_optimizer

    try:
        from mindspeed_mm.fsdp.train import trainer as trainer_module
    except ImportError:
        trainer_module = None

    if trainer_module is not None:
        patch_module_build_optimizer(trainer_module)

    main_module = sys.modules.get("__main__")
    if main_module is not None:
        patch_module_build_optimizer(main_module)


_patch_qwen3_asr_optimizer_builder()


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
