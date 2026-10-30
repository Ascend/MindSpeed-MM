# Data Configuration

## Applicable Backends

This document applies to the Mcore and Mcore-FSDP2 backends.

## Configuration Description

Data configuration can be performed in two ways.

Method 1: Configure data-related parameters in a YAML file. Take [examples/qwen3vl/qwen3vl_full_sft_30B.yaml](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/qwen3vl/qwen3vl_full_sft_30B.yaml) as an example:

```YAML
### Data-related configuration
data:
  dataset_param:
    dataset_type: huggingface
    # Dataset attributes
    attr:
      system: null
      images: images
      videos: null
      messages: messages
      role_tag: role
      content_tag: content
      user_tag: user
      assistant_tag: assistant
      observation_tag: null
      function_tag: null
      system_tag: null

    # Data preprocessing
    preprocess_parameters:
      model_name_or_path: *HF_MODEL_LOAD_PATH
      use_fast_tokenizer: true
      split_special_tokens: false
      image_max_pixels: 262144
      image_min_pixels: 1024
      video_max_pixels: 16384
      video_min_pixels: 0
      video_fps: 2.0
      video_maxlen: 64

    basic_parameters:
      template: qwen3_vl_nothink
      enable_thinking: false
      train_on_prompt: false
      mask_history: false
      tool_format: null
      dataset_dir: ./data
      dataset: *DATASET_PATH
      cache_dir: ./data/cache_dir
      overwrite_cache: false
      preprocessing_batch_size: 1000
      preprocessing_num_workers: 16
      max_samples: null

  # Data loading
  dataloader_param:
    pin_memory: true
    shuffle: true
    dataloader_mode: sampler
    drop_last: true
    sampler_type: BaseRandomBatchSampler
    collate_param:
      model_name: qwen3vl
      ignore_pad_token_for_loss: true
```

Method 2: Configure in a separate json file. Take [examples/qwen3vl/data_30.json](https://gitcode.com/cxiaolong/MindSpeed-MM_ulysses/blob/master/examples/qwen3vl/data_30B.json) as an example:

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",                                           // Dataset type. Parsed in the mindspeed_mm/data/build_mm_dataset function to obtain the corresponding Dataset class.
        "preprocess_parameters": {                                               // Preprocessing parameters passed to the specific Dataset class.
            "model_name_or_path": "./ckpt/hf_path/Qwen3-VL-30B-Instruct",  
            "use_fast_tokenizer": true,
            "split_special_tokens": false,
            "image_max_pixels": 262144,
            "image_min_pixels": 1024,
            "video_max_pixels": 16384,
            "video_min_pixels": 0,
            "video_fps": 2.0,
            "video_maxlen": 64
        },
        "basic_parameters": {                                                   // Basic parameters passed to the specific Dataset class.
            "template": "qwen3_vl_nothink",
            "dataset_dir": "./data",
            "dataset": "./data/mllm_format_llava_instruct_data.json",
            "cache_dir": "./data/cache_dir",
            "enable_thinking": false,
            "overwrite_cache": false,
            "train_on_prompt": false,
            "mask_history": false,
            "preprocessing_batch_size": 1000,
            "preprocessing_num_workers": 16,
            "max_samples": null,
            "tool_format": null
        },
        "attr": {                                                              // Parsed in get_qwen2vl_dataset.
            "system": null,
            "images": "images",
            "videos": null,
            "messages": "messages",
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": null,
            "function_tag": null,
            "system_tag": null
        }
    },
    "dataloader_param": {                                                      // DataLoader-related parameters. Parsed in the mindspeed_mm/data/build_mm_dataloader function to obtain the corresponding DataLoader class.
        "dataloader_mode": "sampler",
        "drop_last": true,
        "sampler_type": "BaseRandomBatchSampler",
        "collate_param": {
            "model_name": "qwen3vl",
            "ignore_pad_token_for_loss": true
        },
        "pin_memory": true,
        "shuffle": true
    }
}
```

The parameters configured in the json file above are the same as those configured in the yaml file; only the format differs. In the launch script, Method 1 requires the `config.yaml` file, and Method 2 requires the `data.json` file.
Note that understanding models support both methods above, while generation models currently support only Method 2.

Method 1:

```Shell
config_path=examples/qwen3vl/qwen3vl_full_sft_30B.yaml
torchrun $DISTRIBUTED_ARGS pretrain_transformers.py ${config_path} \
    --distributed-backend nccl
```

Method 2:

```Shell
......
MM_DATA="examples/qwen3vl/data_30.json"
MM_ARGS="
    --mm-data $MM_DATA \
    --mm-model $MM_MODEL \
    --mm-tool $MM_TOOL
"
......
logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs
torchrun $DISTRIBUTED_ARGS pretrain_transformers.py \
    $GPT_ARGS \
    $MM_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    2>&1 | tee logs/train_${logfile}.log
```
