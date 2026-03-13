# 纯文本预训练

## 使用场景

预训练（Pretraining）是语言模型发展的核心步骤，目标是让模型通过大规模无标签语料学习语言规律与世界知识。预训练过程更关注语言建模本身，而非具体任务执行。以GPT类模型为例，它是一种典型的自回归语言模型，其核心思想是基于历史上下文预测下一个标记。预训练的过程就是通过反复优化这种预测能力，模型逐渐学会如何理解语境、保持句子连贯性，并掌握更高层次的语言结构，为多种下游任务提供通用的语言表示能力。  
预训练数据通常为纯文本格式，无任务导向，例如：

```json
{"text": "今天是个好天气，我们一起去爬山。"}
{"text": "深度学习正在改变世界。"}
{"text": "AI的出现推动了人类社会的发展。"}
```

## 使用方法

1.纯fsdp2后端
在xx_config.yaml文件中,配置预训练相关的参数
```yaml
### 数据相关配置
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

2.含megatron后端
在data.json文件中，配置预训练相关的参数
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

### 参数说明

1.`attr`和`collate_param`下的参数需要全部替换成上述示例的内容，其他参数的值做对应修改
2.`basic_parameters/packing`支持配置，对于纯文本大规模预训练，框架将`packing`默认设为true，以充分利用显存、提升训练效率。如果样本不拼接，则按如下方式进行配置：

- **`basic_parameters/packing`**
  - 描述：多个短文本样本拼接成一个符合模型最大长度（cutoff_len）的长序列
  - 取值：
    - `true`：默认值，可以不配置该参数
    - `false`：手动指定

## 注意事项
1.packing开启（默认）的场景下，需要保证`max_samples`个短文本拼接成长序列的总长度不小于cutoff_len，否则会报错

- **`cutoff_len`**
  - 纯fsdp2后端：对应xx_config.yaml中的`cutoff_len`
  - 含megatron后端：对应finetune_xx.sh中的`SEQ_LEN`