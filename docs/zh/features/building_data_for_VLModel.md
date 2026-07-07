# 针对VL模型的数据构造

当前数据处理方式参考[LLaMAFactory](https://github.com/hiyouga/LLaMAFactory)仓库实现。

## 1. 适用范围

本文所述数据构造方式通用于仓库内大多数 VL 模型（如 [Qwen3.6](../../../examples/qwen3_6)、[Qwen3.5](../../../examples/qwen3_5)、[Qwen3VL](../../../examples/qwen3vl)、[Qwen2.5VL](../../../examples/qwen2.5vl)、[GLM4.5V](../../../examples/glm4.5v)、[Kimi-K2.5](../../../examples/kimik2_5)、[Step3-VL](../../../examples/step3_vl) 等）。各模型若有特殊数据要求（如视频、音频等），以其 README 为准。

<a id="real-data"></a>

## 2. 使用真实数据集

### 2.1. 数据集下载(以coco2017数据集为例)

(1)用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到本地路径，如`./data/COCO2017`。
  > [!NOTE]
  >
  > 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

(2)获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至本地，如`./data/`路径下。

下载得到的是原始格式数据，采用本地多模态 ShareGPT 风格的字段（示例见 2.3），使用前需参考 2.2 转换为训练实际读取的目标格式。

### 2.2. 数据格式转换

运行数据转换脚本，得到格式转换后的描述文件：

```shell
python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
    --coco_path ./data/COCO2017 \
    --llava_json_path ./data/llava_instruct_150k.json \
    --output_json_path ./data/mllm_format_llava_instruct_data.json
```

并在训练开始前修改`xxx_config.yaml`中data配置：

```yaml
### 数据相关配置
data:
  dataset_param:
    basic_parameters:
      # 将该字段修改为COCO2017所在路径
      dataset_dir: ./data/COCO2017
      # 将该字段修改为格式转换后json路径
      dataset: &DATASET_PATH ./data/mllm_format_llava_instruct_data.json
      # 该参数用于限制只读取`max_samples`条数据，可用于快速验证功能，null即为全部数据
      max_samples: null
```

对于其他格式的原始数据集，可以参考`mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py`自行设计数据格式转换脚本，转换后的数据目标格式如下：

```json
# 图文数据
[
  {
    "messages":[
      {
        "content": "<image>source1",
        "role": "user"
      },
      {
        "content": "target1",
        "role": "assistant"
      },
      {
        "content": "<image>source2",
        "role": "user"
      },
      {
        "content": "target2",
        "role": "assistant"
      }
    ],
    "images": [
      "demo_image_1.jpg", "demo_image_2.jpg"
    ]
  },
  ...
]
```

```json
# 视频文本数据
[
  {
    "messages":[
      {
        "content": "<video>source1",
        "role": "user"
      },
      {
        "content": "target1",
        "role": "assistant"
      }
    ],
    "videos": [
      "demo_video.mp4"
    ]
  },
  ...
]
```

### 2.3. 多数据集以及纯文本或有图无图混合数据（以LLaVA原始数据格式为例）

当前支持读取多个以`,`（注意不要加空格）分隔的数据集

例如`xxx_config.yaml`：

```yaml
### 数据相关配置
data:
  dataset_param:
    basic_parameters:
      dataset_dir: ./data/COCO2017  # 将该字段修改为COCO2017所在路径
      dataset: &DATASET_PATH ./data/mllm_format_llava_instruct_data1.json,./data/mllm_format_llava_instruct_data2.json  # 将该字段修改为格式转换后json路径
```

现在本框架已经支持纯文本/混合数据（有图像和无图像数据混合训练）。

> **注意：以下示例为转换前的源格式（LLaVA 原始风格），并非训练直接读取的格式。** 准备好后仍需参照上一章节转换为目标格式后再用于训练。

在数据构造时，对于包含图片的数据，需要保留`image`这个键值。
各字段含义如下：

- 图片字段：`image`
- 对话字段：`conversations`
- 角色字段：`from`
- 内容字段：`value`

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

在数据构造时，对于纯文本数据，可以去除`image`这个键值。

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

<a id="mock-data"></a>

## 3. 使用虚构数据进行功能/性能测试

使用真实数据集进行训练时，通常因为样本间序列长度不一，每一步迭代的时间会有所波动，且真实数据通常较大，有一定的下载和使用成本，因此在指定数据分辨率、序列长度的功能和性能测试场景，使用虚构数据可以更快的满足测试效果。

当前仓库提供了一种构造指定配置图文数据的方法，虚构数据生成脚本使用指令如下：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SAVE_DIR=./data/mocked_vl_data/
mkdir -p $SAVE_DIR
# 下方命令会生成包括512条样本的数据集，每条样本拥有10张1024*1024大小的图片以及16384的文本长度，--tokenizer_path需要指定当前待测模型的原始权重本地路径
python mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py \
    --tokenizer_path /home/weights/Qwen3.5-35B-A3B/ \
    --pic_width 1024 \
    --pic_height 1024 \
    --num_pics 10 \
    --text_length 16384 \
    --num_samples 512 \
    --save_dir $SAVE_DIR
```

并在训练开始前修改`xxx_config.yaml`中data配置：

```yaml
### 数据相关配置
data:
  dataset_param:
    basic_parameters:
      # 该参数指定模型训练的核心语言模块接受的最大序列长度，超出该配置的部分将被截断，建议构造数据是手动计算图文序列长度占比及总长度，尽可能与cutoff_len数值接近，否则会有截断图片占位符无法正常训练的风险
      cutoff_len: 16384
      # 将该字段修改构造数据的保存路径
      dataset_dir: ./data/mocked_vl_data
      # 将该字段修改为构造数据的json路径
      dataset: &DATASET_PATH ./data/mocked_vl_data/mock_data_pic_num_10_textlen_16384.json
      # 该参数用于限制只读取`max_samples`条数据，可用于快速验证功能，null即为全部数据
      max_samples: null
```
