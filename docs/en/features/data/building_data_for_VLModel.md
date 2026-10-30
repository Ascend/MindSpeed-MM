# Data Construction for VL Models

The current data processing approach is implemented with reference to the [LLaMAFactory](https://github.com/hiyouga/LLaMAFactory) repository.

## 1. Scope of Application

The data construction method described in this document applies to most VL models in the repository, such as [Qwen3.6](../../../../examples/qwen3_6), [Qwen3.5](../../../../examples/qwen3_5), [Qwen3-VL](../../../../examples/qwen3vl), [GLM4.5V](../../../../examples/glm4.5v), [Kimi-K2.5](../../../../examples/kimik2_5), and [Step3-VL](../../../../examples/step3_vl). If a model has special data requirements (such as video or audio), its README shall prevail.

<a id="real-data"></a>

## 2. Using Real Datasets

### 2.1. Dataset Download (COCO2017 as an Example)

(1) Download the [COCO2017](https://cocodataset.org/#download) dataset yourself and extract it to a local path, such as `./data/coco/COCO2017`.
  > [!NOTE]
  >
  > If resources cannot be accessed smoothly from the Hugging Face community, it is recommended to download them from ModelScope. Pay attention to the correctness and security of the files to be downloaded.

(2) Obtain the description file of the image dataset ([LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)) and download it locally, for example, to the `./data/coco/` path.

The downloaded data is in its original format, which uses the local multimodal ShareGPT-style fields (see "Section 2.3" for an example). Before use, refer to "Section 2.2" to convert it into the target format actually read during training.

### 2.2. Data Format Conversion

Run the data conversion script to obtain the description file after format conversion:

```shell
python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py \
    --coco_path ./data/coco/COCO2017 \
    --llava_json_path ./data/coco/llava_instruct_150k.json \
    --output_json_path ./data/coco/mllm_format_llava_instruct_data.json
```

The directory structure after running the data conversion script is as follows:

```text
./data/
└── coco/                                  # Parent directory of the COCO dataset (stores the annotation JSON)
    ├── COCO2017/                          # COCO image dataset (both --coco_path and dataset_dir point here)
    │   ├── train2017/
    │   │   ├── 000000000001.jpg
    │   │   └── ...
    │   └── val2017/
    ├── llava_instruct_150k.json           # Original LLaVA annotation file
    └── mllm_format_llava_instruct_data.json  # Converted annotation file (finally used for training)
```

Before starting training, modify the data configuration in `xxx_config.yaml`:

```yaml
### Data-related configuration
data:
  dataset_param:
    basic_parameters:
      # Change this field to the COCO2017 path (that is, the parent directory of train2017/val2017); at runtime, this path is concatenated with the relative image paths in the annotation JSON
      dataset_dir: ./data/coco/COCO2017
      # Change this field to the JSON path after format conversion
      dataset: &DATASET_PATH ./data/coco/mllm_format_llava_instruct_data.json
      # This parameter limits reading to only `max_samples` entries, which can be used for quick functional verification; null means all data
      max_samples: null
```

For raw datasets in other formats, refer to `mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py` to design a data format conversion script as needed. The target format of the converted data is as follows:

```json
# Image-text data
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
# Video-text data
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

### 2.3. Multiple Datasets and Mixed Data (Text-Only or With/Without Images — Using LLaVA Original Data Format as an Example)

Currently, reading multiple datasets separated by `,` (do not add spaces) is supported.

For example, in `xxx_config.yaml`:

```yaml
### Data-related configuration
data:
  dataset_param:
    basic_parameters:
      dataset_dir: ./data/coco/COCO2017  # Change this field to the COCO2017 path (that is, the parent directory of train2017/val2017)
      dataset: &DATASET_PATH ./data/coco/mllm_format_llava_instruct_data1.json,./data/coco/mllm_format_llava_instruct_data2.json  # Change this field to the JSON path after format conversion
```

The framework now supports text-only/mixed data (mixed training with and without images).

> **Note: The following example shows the source format before conversion (LLaVA original style), not the format directly read during training.** After preparation, convert it to the target format as described in the previous section before using it for training.

When constructing data, the `image` key must be retained for samples containing images.
The meaning of each field is as follows:

- Image field: `image`
- Conversation field: `conversations`
- Role field: `from`
- Content field: `value`

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

During data construction, the `image` key can be removed for text-only data.

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

## 3. Using Mock Data for Functional/Performance Testing

When training with real datasets, the iteration time usually fluctuates across steps because sequence lengths vary between samples, and real data is usually large and incurs certain download and usage costs. Therefore, in functional and performance test scenarios with a specified data resolution and sequence length, using mock data can meet the test objectives faster.

The repository provides a method for constructing mock data with configurable image-text composition. The mock data generation script can be invoked as follows:

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
SAVE_DIR=./data/fake/mocked_vl_data/
mkdir -p $SAVE_DIR
# The following command generates a dataset containing 512 samples, where each sample has 10 images of 1024*1024 size and a text length of 16384. --tokenizer_path needs to specify the local path of the original weights of the model under test
python mindspeed_mm/fsdp/tools/data_tool/generate_mock_data_for_vlmodel.py \
    --tokenizer_path /home/weights/Qwen3.5-35B-A3B/ \
    --pic_width 1024 \
    --pic_height 1024 \
    --num_pics 10 \
    --text_length 16384 \
    --num_samples 512 \
    --save_dir $SAVE_DIR
```

Before starting training, modify the data configuration in `xxx_config.yaml`:

```yaml
### Data-related configuration
data:
  dataset_param:
    basic_parameters:
      # This parameter specifies the maximum sequence length accepted by the core language module during model training; any part exceeding this configuration is truncated. When constructing data, it is recommended to manually calculate the image-text sequence length ratio and the total length and keep them as close to the cutoff_len value as possible; otherwise, there is a risk that the image placeholders are truncated and cannot be trained properly
      cutoff_len: 16384
      # Change this field to the save path of the constructed data
      dataset_dir: ./data/fake/mocked_vl_data
      # Change this field to the JSON path of the constructed data
      dataset: &DATASET_PATH ./data/fake/mocked_vl_data/mock_data_pic_num_10_textlen_16384.json
      # This parameter limits reading to only `max_samples` entries, which can be used for quick functional verification; null means all data
      max_samples: null
```
