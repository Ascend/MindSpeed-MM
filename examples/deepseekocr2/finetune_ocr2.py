import sys
import os

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.optim import AdamW

from transformers import AutoModel

ocr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "deepseekocr")
if ocr_dir not in sys.path:
    sys.path.insert(0, ocr_dir)

from examples.deepseekocr.finetune_ocr import OCRTrainer, get_parser


class OCR2Trainer(OCRTrainer):
    def __init__(self, config):
        super().__init__(config)

    def build_model_and_optimizer(self, attn_implementation="eager"):
        self.model = AutoModel.from_pretrained(
            self.config.load,
            _attn_implementation=attn_implementation,
            trust_remote_code=True,
            use_safetensors=True
        ).to("cuda", dtype=torch.bfloat16)
        if self.config.freeze_tokenizer:
            self.model.model.sam_model.requires_grad_(False)
        if self.config.freeze_vis_encoder:
            self.model.model.qwen2_model.requires_grad_(False)
            self.model.model.projector.requires_grad_(False)
        if self.config.freeze_decoder:
            self.model.model.layers.requires_grad_(False)
            self.model.model.embed_tokens.requires_grad_(False)

        fsdp_kwargs = {}
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32
        )

        fully_shard(self.model.model.embed_tokens)
        for layer in self.model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        for layer in self.model.model.qwen2_model.model.model.layers:
            fully_shard(layer, **fsdp_kwargs)
        fully_shard(self.model.lm_head, **fsdp_kwargs)
        fully_shard(self.model, **fsdp_kwargs)

        if torch.distributed.get_rank() == 0:
            print(self.model)

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        num_warmup_steps = int(self.config.warmup_ratio * self.config.train_iters)
        self.scheduler = OCR2Trainer.get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.config.train_iters
        )


def get_ocr2_parser():
    parser = get_parser()

    # Training args
    parser.add_argument(
        '--freeze-tokenizer',
        action='store_true',
        default=False,
        help='Whether or not to freeze tokenizer weight in training.'
    )

    parser.add_argument(
        '--freeze-vis-encoder',
        action='store_true',
        default=False,
        help='Whether or not to freeze vision encoder weight in training.'
    )

    parser.add_argument(
        '--freeze-decoder',
        action='store_true',
        default=False,
        help='Whether or not to freeze decoder weight in training.'
    )
    return parser


def main():
    args = get_ocr2_parser().parse_args()

    OCR2Trainer.setup_distributed()
    ocr_trainer = OCR2Trainer(args)
    ocr_trainer.train()
    OCR2Trainer.cleanup_distributed()


if __name__ == "__main__":
    torch.npu.config.allow_internal_format = False
    main()
