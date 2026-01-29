from mindspeed_mm.fsdp.data.data_utils.func_utils.collator import MultiModalDataCollatorForSeq2Seq
from mindspeed_mm.fsdp.data.data_utils.func_utils.convert import load_tokenizer, IGNORE_INDEX
from mindspeed_mm.fsdp.data.data_utils.func_utils.model_args import ProcessorArguments
from mindspeed_mm.fsdp.data.data_utils.func_utils.template import get_template_and_fix_tokenizer


class DataCollatorForQwen2vl:
    def __init__(self, ignore_pad_token_for_loss: bool, dataset_param=None, **kwargs):
        process_args = ProcessorArguments(**dataset_param.preprocess_parameters.to_dict())
        tokenizer_module = load_tokenizer(process_args)
        tokenizer = tokenizer_module.get('tokenizer')
        template = get_template_and_fix_tokenizer(tokenizer, dataset_param.basic_parameters.template)
        self.data_collator = MultiModalDataCollatorForSeq2Seq(
            template=template,
            pad_to_multiple_of=8,  # for shift short attention
            label_pad_token_id=IGNORE_INDEX if ignore_pad_token_for_loss else tokenizer.pad_token_id,
            **tokenizer_module,
        )

    def __call__(self, *args, **kwargs):
        return self.data_collator(*args, **kwargs)

DATA_COLLATOR = {
    "qwen3vl": DataCollatorForQwen2vl,
}