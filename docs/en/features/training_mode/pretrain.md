# Pre-training

## Text-only Pre-training

### Applicable Backends

Text-only pre-training applies to both the pure FSDP2 backend and the Megatron-based backend. The two backends use different configuration items to select the pre-training data processing pipeline. Configure them according to the example of the corresponding backend below.

### Use Case

Pre-training is a core step in the development of language models. The goal is for the model to learn linguistic patterns and world knowledge from large-scale unlabeled corpora. The pre-training process focuses more on language modeling itself than on specific task execution. Taking GPT-class models as an example, they are a type of autoregressive language model whose core idea is to predict the next token based on historical context. Through repeated optimization of this predictive capability during pre-training, the model gradually learns how to understand context, maintain sentence coherence, and master higher-level linguistic structures, providing a general language representation capability for various downstream tasks.
Pre-training data typically consists of pure text format without task orientation, for example:

```json
{"text": "Today is a nice day, let's go hiking together."}
{"text": "Deep learning is changing the world."}
{"text": "The emergence of AI has driven the development of human society."}
```

### How to Use

#### Pure FSDP2 Backend

Configure the pre-training-related parameters in the `xx_config.yaml` file:

```yaml
### Data-related configuration
data:
  dataset_param:
    ...
    attr:
      formatting: alpaca
      prompt: text
    basic_parameters:
      stage: pretrain
      template: default
  dataloader_param:
    collate_param:
      collator_id: llm_pretrain
  ...
```

> [!NOTE]
>
> The FSDP2 backend selects the pre-training data processing pipeline through `basic_parameters.stage: pretrain`. When switching to pre-training, comment out or remove the original configurations under `attr` (such as the column mapping configuration used in the SFT scenario) and keep only the `formatting` and `prompt` configurations as shown in the example above, to avoid data alignment failures caused by residual configurations.

#### Megatron-based Backend

Configure the pre-training-related parameters in the `data.json` file:

```json
{
    "dataset_param": {
        ...
        "basic_parameters": {
            "template": "default",
        },
        "attr": {
            "formatting": "alpaca",
            "pretrain": true,
            "system": null,
            "images": null,
            "videos": null,
            "audios": null,
            "prompt": "text",
            "query": null,
            "response": null,
            "history": null
        }
    },
    "dataloader_param": {
        ...
        "collate_param": {
            "model_name": "llm_pretrain"
        },
        ...
    }
}
```

#### Parameter Description

The FSDP2 backend needs to configure `basic_parameters.stage`, `attr`, and `collate_param` as shown in the example above; the Megatron-based backend needs to configure `attr` and `collate_param` as shown in the example above. Modify other parameters according to the actual training task.

- `basic_parameters.packing`
  - Description: Splices multiple short text samples into a long sequence that conforms to the model's maximum length `cutoff_len`. For large-scale plain text pre-training, this parameter defaults to `true` to make full use of device memory and improve training efficiency.
  - Values:
    - `true`: Enables sample packing, which is the default value. This parameter can be omitted.
    - `false`: Disables sample packing.

- `cutoff_len`
  - Description: Maximum length of the training sequence.
  - Configuration location:
    - Pure FSDP2 backend: Corresponds to `cutoff_len` in `xx_config.yaml`.
    - Megatron-based backend: Corresponds to `SEQ_LEN` in `finetune_xx.sh`.

### Notes

The following notes apply to text-only pre-training (with no `images`, `videos`, or `audios` column configured in `attr`). When packing is enabled (the default), data preprocessing splices texts within each batch and splits them into fixed-length sequences according to `cutoff_len`. Ensure that the total number of tokens of valid texts in each preprocessing batch is not less than `cutoff_len`; otherwise, the batch cannot generate training samples. To resolve this, increase `preprocessing_batch_size`, increase the sample length, or reduce `cutoff_len`.

## Multimodal Pre-training Packing (FSDP2 Only)

The pure FSDP2 data layer supports expanding the image, video, or audio tokens of each multimodal document independently under `stage: pretrain`, and then packing by complete document. Different documents are not split. `cu_seqlens`, the reset position IDs, and the ignore label of the first token of a document are used to isolate the attention and linear attention state, the MRoPE positions, and the cross-document next-token loss, respectively.

Currently this capability is supported only for models configured with `dataset_type: huggingface`. The data layer selects the processing pipeline based on whether the `images`, `videos`, or `audios` column is configured in `attr`. If any media column is configured, the multimodal pipeline described in this section is used; if none is configured, the text-only pipeline described above is used and the behavior stays the same as before. Pre-training data can use the assistant-only document format, and the order in which media placeholders appear must match the corresponding media array. For example:

```json
{
  "messages": [{"role": "assistant", "content": "<image> is a red square."}],
  "images": ["/absolute/path/red_square.png"]
}
```

Taking Qwen3.5 FSDP2 as an example, configure the following:

```yaml
data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images  # The media column must be configured to use the multimodal pre-training pipeline.
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
    basic_parameters:
      stage: pretrain
      packing: true
      preprocess_on_fly: false
      cutoff_len: 4096
  dataloader_param:
    collate_param:
      collator_id: qwen3vl

training:
  micro_batch_size: 1
```

Note the following when using this feature:

1. Multimodal packing relies on offline batched preprocessing and cannot be used together with `preprocess_on_fly`.
2. Complete multimodal documents that exceed `cutoff_len` are discarded, and they are never split in the middle of media tokens or media features.
3. The current packed `cu_seqlens` pipeline requires right-side padding and `micro_batch_size` set to 1; a single packed row can still contain multiple original documents.
4. The model side must support consuming `cu_seqlens` to isolate the attention and other sequence states of different documents. For Qwen3.5, keep using `gdn_implementation: ascendc` and `causal_conv1d_implementation: triton` as in the example configuration.
5. The expanded image, video, and audio placeholder tokens (such as `<|image_pad|>`) do not participate in the loss and serve only as context; the `<|vision_start|>` and `<|vision_end|>` boundary tokens and the text that follows still participate in the loss. This rule applies to templates based on `Qwen2VLPlugin` (the Qwen2-VL, Qwen3-VL, GLM-4.1V, and Qwen Omni series).
6. Do not write media tokens (such as `<|vision_start|>` or `<|image_pad|>`) directly into the raw text of multimodal pre-training data. Use placeholders such as `<image>` and `<video>` instead. Documents that contain media tokens are discarded and a warning is printed. This check applies only to multimodal pre-training and does not affect SFT.
