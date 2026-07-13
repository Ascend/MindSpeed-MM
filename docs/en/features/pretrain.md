# Pure Text Pre-training

## Use Case

Pre-training is a core step in the development of language models. The goal is for the model to learn linguistic patterns and world knowledge from large-scale unlabeled corpora. The pre-training process focuses more on language modeling itself than on specific task execution. Taking GPT-class models as an example, they are a type of autoregressive language model whose core idea is to predict the next token based on historical context. Through repeated optimization of this predictive capability during pre-training, the model gradually learns how to understand context, maintain sentence coherence, and master higher-level linguistic structures, providing a general language representation capability for various downstream tasks.

Pre-training data typically consists of pure text format without task orientation, for example:

```json
{"text": "Today is a beautiful day. Let's go hiking together."}
{"text": "Deep learning is changing the world."}
{"text": "The emergence of AI has propelled the development of human society."}
```

## How to Use

1. Pure FSDP2 Backend
   In the `xx_config.yaml` file, configure the parameters related to pre-training:

   ```yaml
   Data-related configuration
   data:
     dataset_param:
       ...
       attr:
         formatting: alpaca
         pretrain: true
         prompt: text
       basic_parameters:
         template: default
     dataloader_param:
       collate_param:
         model_name: llm_pretrain
     ...
   ```

2. With Megatron Backend
   In the `data.json` file, configure the parameters related to pre-training:

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
    ...
   ```

### Parameter Description

1. The parameters under `attr` and `collate_param` must be replaced with the content shown in the example above. Values for other parameters should be modified accordingly.
2. `basic_parameters/packing` supports configuration. For large-scale pure text pre-training, the framework sets `packing` to `true` by default to fully utilize memory and improve training efficiency. If samples are not to be concatenated, configure as follows:

- `basic_parameters/packing`
  - Description: Concatenates multiple short text samples into a long sequence that conforms to the model's maximum length (`cutoff_len`).
  - Values:
    - `true` (default value): This parameter can be left unconfigured.
    - `false`: Manually specified.

## Notes

1. When packing is enabled (default), the total length of concatenated short texts must be no less than `cutoff_len`. Otherwise, an error will be raised.

- `cutoff_len`
  - Pure FSDP2 backend: corresponds to `cutoff_len` in `xx_config.yaml`
  - With Megatron backend: corresponds to `SEQ_LEN` in `finetune_xx.sh`.
