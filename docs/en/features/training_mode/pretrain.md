# Pure Text Pre-training

## Applicable Backends

Plain text pre-training applies to both the pure FSDP2 backend and the Megatron-based backend. The two backends use different configuration items to select the pre-training data processing pipeline. Configure them according to the example of the corresponding backend below.

## Use Case

Pre-training is a core step in the development of language models. The goal is for the model to learn linguistic patterns and world knowledge from large-scale unlabeled corpora. The pre-training process focuses more on language modeling itself than on specific task execution. Taking GPT-class models as an example, they are a type of autoregressive language model whose core idea is to predict the next token based on historical context. Through repeated optimization of this predictive capability during pre-training, the model gradually learns how to understand context, maintain sentence coherence, and master higher-level linguistic structures, providing a general language representation capability for various downstream tasks.
Pre-training data typically consists of pure text format without task orientation, for example:

```json
{"text": "Today is a nice day, let's go hiking together."}
{"text": "Deep learning is changing the world."}
{"text": "The emergence of AI has driven the development of human society."}
```

## How to Use

### Pure FSDP2 Backend

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

### Megatron-based Backend

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

### Parameter Description

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

## Notes

When packing is enabled (the default), data preprocessing splices texts within each batch and splits them into fixed-length sequences according to `cutoff_len`. Ensure that the total number of tokens of valid texts in each preprocessing batch is not less than `cutoff_len`; otherwise, the batch cannot generate training samples. To resolve this, increase `preprocessing_batch_size`, increase the sample length, or reduce `cutoff_len`.
