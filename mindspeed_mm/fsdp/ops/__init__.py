import logging

from mindspeed.fsdp.utils.log import print_rank
from .flash_attn.flash_attn import apply_transformers_attention_patch

logger = logging.getLogger(__name__)


def apply_ops_patch():
    apply_transformers_attention_patch()

    print_rank(logger.info, "✅ MindSpeed-MM ops patch applied.")