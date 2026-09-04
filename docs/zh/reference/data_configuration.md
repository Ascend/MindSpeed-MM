# 数据配置

## 适用后端

该文档适用于Mcore以及Mcore-FSDP2后端。

## 配置说明

数据配置有两种方式可以进行配置，

方式一：在yaml中进行data相关参数配置，以[examples/qwen3vl/qwen3vl_full_sft_30B.yaml](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/qwen3vl/qwen3vl_full_sft_30B.yaml)为例：

```YAML
### 数据相关配置
data:
  dataset_param:
    dataset_type: huggingface
    #数据集属性
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

    # 数据预处理
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

  # 数据加载
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

方式二：在单独的json文件中进行配置，以[examples/qwen3vl/data_30.json](https://gitcode.com/cxiaolong/MindSpeed-MM_ulysses/blob/master/examples/qwen3vl/data_30B.json)为例：

```json
{
    "dataset_param": {
        "dataset_type": "huggingface",                                           // 数据集类型，在 mindspeed_mm/data/build_mm_dataset 函数中解析，拿到对应DataSet类
        "preprocess_parameters": {                                               // 预处理参数，是具体DataSet的入参
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
        "basic_parameters": {                                                   // 基础参数，是具体DataSet的入参
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
        "attr": {                                                              // 在 get_qwen2vl_dataset 中解析
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
    "dataloader_param": {                                                      // DataLoader相关参数， 在 mindspeed_mm/data/build_mm_dataloader 函数中解析，拿到对应DataLoader类
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

上述json文件中配置的参数与yaml文件中配置的参数是一致的，只是格式不同。在拉起脚本中，方式一需要指定config.yaml文件，方式二需要指定data.json文件。
注意:理解类型模型支持上述两种方式，生成模型暂时支持方式二。
方式一:

```Shell
config_path=examples/qwen3vl/qwen3vl_full_sft_30B.yaml
torchrun $DISTRIBUTED_ARGS pretrain_transformers.py ${config_path} \
    --distributed-backend nccl
```

方式二:

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
