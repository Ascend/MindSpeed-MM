# 快速入门：Qwen2.5-VL模型微调和Wan2.1模型预训练

MindSpeed MM同时支持多模态生成和多模态理解模型，下面分别以Qwen2.5-VL（理解模型）和Wan2.1（生成模型）两个典型模型为例，介绍MindSpeed MM的使用方法，引导开发者快速上手预置模型在昇腾NPU上的高效运行。

## 多模态理解模型

本章节以Qwen2.5-VL-3B为例，指导用户在单机场景下如何完成多模态理解模型的微调。

### 环境准备
  
1. 基于PyTorch框架和Python3.10完成模型训练环境的安装，具体请参见[MindSpeed MM安装指导](../install_guide.md)。
2. 在`MindSpeed-MM`下创建以下目录用于存储日志、数据及权重文件。

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

### 权重下载及转换

1. 权重下载

   从Hugging Face下载对应的模型权重[Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/tree/main)。
    
2. 权重文件保存

   创建`ckpt/hf_path/Qwen2.5-VL-3B-Instruct`目录并将下载的模型权重保存到该目录下。
   
3. 权重转换

    MindSpeed MM修改了部分原始网络的结构名称，可使用`mm-convert`工具对原始预训练权重进行转换。执行如下命令运行工具：

    ```bash
    # Qwen2.5-VL-3B
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[36]] \
    --cfg.parallel_config.vit_pp_layers [[32]] \
    --cfg.parallel_config.tp_size 1
    ```

    **表 1** 权重转换工具参数解析

    |参数|说明|是否必选|默认值|
    |-|-|-|-|
    |Qwen2_5_VLConverter|Qwen2.5-VL模型转换工具|是|/|
    |hf_to_mm|Hugging Face模型转换MindSpeed MM模型权重|是|/|
    |mm_dir|转换后保存目录|是|/|
    |hf_dir|Hugging Face权重目录|是|/|
    |llm_pp_layers|llm在每个卡上切分的层数，注意要和examples/qwen2.5vl/model_3b.json中配置的pipeline_num_layers一致|否|36|
    |vit_pp_layers|vit在每个卡上切分的层数，注意要和examples/qwen2.5vl/model_3b.json中配置的pipeline_num_layers一致|否|32|
    |tp_size|TP并行数量，注意要和微调启动脚本中的配置一致|否|1|

    > [!NOTE]  
    > 由于Qwen2_5_VL和Qwen2_VL在权重转换逻辑上保持一致，更多工具详情可参见[权重转换命令行工具](../features/mm_convert.md)。

### 数据预处理

1. 数据集下载
   
   以COCO2017数据集为例，创建`data/COCO2017`目录后下载并解压[COCO2017](https://cocodataset.org/#download)数据集。
       
2. 获取数据集描述文件

   从Hugging Face下载图片数据集的描述文件[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)，保存至./data/路径下。

3. 数据集预处理

    执行如下数据转换脚本：

    ```python
    python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py
    ```

    转换后参考数据目录结构如下：

    ```bash
    $playground
    ├── data
        ├── COCO2017
            ├── train2017

        ├── llava_instruct_150k.json
        ├── mllm_format_llava_instruct_data.json
        ...
    ```

    > [!NOTE]  
    > 由于Qwen2_5_VL和Qwen2_VL在数据转换逻辑上保持一致，因此采用了Qwen2_VL下的数据转换脚本来满足当前需求。

### 启动微调

1. 数据目录配置

    在`examples/qwen2.5vl/data_3b.json`中完成数据集路径的配置，配置示例如下：

    ```json
        {
            "dataset_param": {
                "dataset_type": "huggingface",
                "preprocess_parameters": {
                    "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-3B-Instruct",
                    ...
                },
                "basic_parameters": {
                    ...
                    "dataset_dir": "./data",
                    "dataset": "./data/mllm_format_llava_instruct_data.json",
                    "cache_dir": "./data/cache_dir",
                    ...
                },
                ...
            },
            ...
        }
    ```

    **表 2** 参数配置解析

    |参数|说明|取值|
    |-|-|-|
    |model_name_or_path|权重|"./ckpt/hf_path/Qwen2.5-VL-3B-Instruct"，与[权重下载及转换](#权重下载及转换)中的`hf_config.hf_dir`一致。|
    |dataset_dir|数据集目录|"./data"|
    |dataset|数据集|"./data/mllm_format_llava_instruct_data.json"|

    > [!CAUTION]   
    > 为了避免写入同一个文件导致冲突的问题，在多机上不要配置同一个挂载目录(`cache_dir`)。

2. 编辑微调示例脚本

    ```shell
    vi examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```

3. 模型保存加载及日志信息配置

    完成模型加载、保存路径以及保存间隔`--save-interval`的配置，配置示例如下：

    ```bash
    ...
    # 加载路径
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
    # 保存路径
    SAVE_PATH="save_dir"
    ...
    GPT_ARGS="
        ...
        --no-load-optim \  # 不加载优化器状态，若需加载请移除
        --no-load-rng \  # 不加载随机数状态，若需加载请移除
        --no-save-optim \  # 不保存优化器状态，若需保存请移除
        --no-save-rng \  # 不保存随机数状态，若需保存请移除
        ...
    "
    ...
    OUTPUT_ARGS="
        --log-interval 1 \  # 日志间隔
        --save-interval 5000 \  # 保存间隔
        ...
        --log-tps \  # 增加此参数可使能在训练中打印每步语言模块的平均序列长度，并在训练结束后计算每秒吞吐tokens量。
    "
    ```

    **表 3** 参数配置解析

    |参数|说明|取值|
    |-|-|-|
    |LOAD_PATH|加载路径|ckpt/mm_path/Qwen2.5-VL-3B-Instruct|
    |SAVE_PATH|保存路径|save_dir|
    |`--log-interval`|日志间隔|1|
    |`--save-interval`|保存间隔|5000|
    |`--no-load-optim`|不加载优化器状态，若需加载请移除|/|
    |`--no-load-rng`|不加载随机数状态，若需加载请移除|/|
    |`--no-save-optim`|不加载随机数状态，若需加载请移除|/|
    |`--no-save-rng`|不保存随机数状态，若需保存请移除|/|

    > [!NOTE]   
    > 由于分布式优化器保存文件较大，导致耗时较长，请谨慎设置保存间隔。

4. 模型运行参数配置

    完成模型运行参数的配置，配置示例如下：

    ```bash
    # 根据实际情况修改 ascend-toolkit 路径
    source /usr/local/Ascend/cann/set_env.sh
    NPUS_PER_NODE=8          # 使用单节点的8卡NPU              
    MASTER_ADDR=localhost    # 单机使用本节点ip
    MASTER_PORT=29501        # 本节点端口号为29501
    NNODES=1                 # 根据参与节点数量配置，单机为1
    NODE_RANK=0              # 单机RANK为0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
    ```

5. 启动微调

    保存微调脚本后，启动微调任务，命令如下：

    ```shell
    bash examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh
    ```

### 后续处理

MindSpeed MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回Hugging Face格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

以下是mm2hf的转换示例：

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-3B-Instruct" \
--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-3B-Instruct" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-3B-Instruct" \
--cfg.parallel_config.llm_pp_layers [36] \
--cfg.parallel_config.vit_pp_layers [32] \
--cfg.parallel_config.tp_size 1
```

**表 4** mm2hf参数

|参数|含义|是否必选|默认值|
|:----|:----|:----|:----|
|Qwen2_5_VLConverter|Qwen2.5-VL模型转换工具|是|/|
|mm_to_hf|MindSpeed MM模型转换Hugging Face模型权重|是|/|
|save_hf_dir|mm微调后转换回hf模型格式的目录|是|/|
|mm_dir|微调后保存的权重目录|是|/|
|hf_dir|Hugging Face权重目录|是|/|
|llm_pp_layers|llm在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致|否|36|
|vit_pp_layers|vit在每个卡上切分的层数，注意要和微调时model.json中配置的pipeline_num_layers一致|否|32|
|tp_size|TP并行数量，注意要和微调启动脚本中的配置一致|否|1|

如果需要用转换后模型进行训练的话，同步修改`examples/qwen2.5vl/finetune_qwen2_5_vl_3b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-3B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-3B-Instruct"
```

## 多模态生成模型

本章节以Wan2.1-T2V-1.3B为例，指导用户在单机场景下如何完成多模态生成模型的预训练。

### 环境准备

1. 基于PyTorch框架和Python3.10完成模型训练环境的安装，具体请参见[MindSpeed MM安装指导](../install_guide.md)。

2. 安装其它依赖：

    ```bash
    # 源码安装Diffusers
    pip install diffusers==0.33.1
    ```

3. Decord搭建

    - X86版安装

        ```bash
        pip install decord==0.6.0
        ```

    - ARM版安装

        `apt`方式安装请参考[decord](https://github.com/dmlc/decord)

        `yum`方式安装请参考脚本[build_manylinux2010.sh](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh)

### 权重下载及转换

1. 权重下载

   从Hugging Face下载对应的模型权重[Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tree/main)。

2. 权重文件保存

   在MindSpeed-MM下创建`weights/Wan2.1-T2V-1.3B-Diffusers/`，并将下载的模型权重保存到该目录下。

3. 权重转换

    需要对下载后的Wan2.1模型权重`transformer`部分进行权重转换，运行权重转换工具：

    ```shell
    mm-convert WanConverter hf_to_mm \
    --cfg.source_path ./weights/Wan2.1-T2V-1.3B-Diffusers/transformer/ \
    --cfg.target_path ./weights/Wan2.1-T2V-1.3B-Diffusers/transformer_mm/
    ```

    **表 5** 权重转换工具参数解析

    | 参数 |说明 |
    | :-- | :--- | 
    |WanConverter|Wan2.1模型转换工具|
    |hf_to_mm|Hugging Face模型转换MindSpeed MM模型权重|
    | source_path | 原始权重路径|
    | target_path | 转换或切分后权重保存路径 | 

### 数据预处理

在`MindSpeed-MM`下创建`dataset`目录，随后在`dataset`下创建目录`videos`和文件`data.json`，并将需要处理的视频保存在`videos`中。数据集中所有的视频-文本对信息保存在`data.json`中。

具体目录结构示例如下：

```bash
dataset
├──data.json
├──videos
│  ├──video0001.mp4
│  ├──video0002.mp4
```

视频-文本对信息示例如下：

```json
[
    {
        "path": "videos/video0001.mp4",
        "cap": "Video discrimination1.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
    {
        "path": "videos/video0002.mp4",
        "cap": "Video discrimination2.",
        "num_frames": 81,
        "fps": 24,
        "resolution": {
            "height": 480,
            "width": 832
        }
    },
]
```

|参数|说明|默认值|
|-|-|-|
|path|视频存放路径|/|
|cap|视频描述|根据用户实际情况配置|
|num_frames|最大的帧数|81|
|fps|视频帧数|根据用户实际情况配置|
|height|视频高度|根据用户实际情况配置|
|width|视频宽度|根据用户实际情况配置|

### 特征提取

1. 配置data.txt

    修改`examples/wan2.1/feature_extract/data.txt`文件，其中每一行表示一个数据集，第一个参数表示数据文件夹的路径，第二个参数表示`data.json`文件的路径，用`,`分隔。作如下修改：

    ```text
    ./dataset,./dataset/data.json
    ```

2. 配置data.json

    修改`examples/wan2.1/feature_extract/data.json`文件，根据实际情况配置如下参数：
    - `num_frames`：表示最大帧数，默认为81，超过则随机选取其中的`num_frames`帧。
    - `max_height`：表示最大高度，默认为480，超过则centercrop到最大分辨率。
    - `max_width`：表示最大宽度，默认为832，超过则centercrop到最大分辨率。
    - `from_pretrained`：表示tokenizer权重所对应路径，默认为"weights/Wan2.1-T2V-1.3B-Diffusers/tokenizer"。

    ```json
    "preprocess_parameters": {
        ......
        "num_frames": 81,
        "max_height": 480,
        "max_width": 832,
        ......
    "tokenizer_config": 
    {
        ......
        "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/tokenizer",
        ......
    }
    }
    ```

3. 配置model_t2v.json

    修改`examples/wan2.1/feature_extract/model_t2v.json`文件，其中`from_pretrained`为下载的权重所对应路径，包括vae和text_encoder。根据实际情况修改参数：

    ```json
    {
        "ae": {
            ......
            "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/vae",
            ......
        },
        "text_encoder": {
            ......
            "from_pretrained": "weights/Wan2.1-T2V-1.3B-Diffusers/text_encoder"
        }
    }
    ```

4. 配置tools.json

    修改`mindspeed_mm/tools/tools.json`，其中`sorafeature`的`save_path`为提取后的特征保存路径：

    ```json
        "sorafeature":{
        "save_path": "./sora_features"
    }
    ```
   
5. 配置特征提取脚本

    修改`examples/wan2.1/feature_extract/feature_extraction.sh`中的`NPUS_PER_NODE`，默认参数为1，请修改为实际使用卡数。 

6. 启动特征提取

    ```bash
    bash examples/wan2.1/feature_extract/feature_extraction.sh
    ```

### 启动训练

1. 参数配置检查

    确认完成下表中所有配置文件的修改字段修改。

    **表 7**  配置文件修改字段表

    | 配置文件   |      字段       | 修改说明      |
    | --- | :---: | :--- |
    | examples/wan2.1/1.3b/t2v/data.txt    | 文件内容  | 提取后的特征保存路径 |
    | examples/wan2.1/1.3b/t2v/feature_data.json   |   from_pretrained   | 修改为下载的权重所对应路径，与[权重下载及转换](#权重下载及转换)中的保持一致|
    | examples/wan2.1/1.3b/t2v/pretrain.sh |    NPUS_PER_NODE    | 每个节点的卡数                                      |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |       NNODES        | 节点数量                                            |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |      LOAD_PATH      | 权重转换后的预训练权重路径，与[权重下载及转换](#权重下载及转换)中的保持一致                         |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |      SAVE_PATH      | 训练过程中保存的权重路径                            |
    | examples/wan2.1/1.3b/t2v/pretrain.sh |         CP          | 训练时的CP size（建议根据训练时设定的分辨率调整）   |

2. 启动训练
    feature_data.json中修改tokenizer权重路径

    ```bash
    bash examples/wan2.1/1.3b/t2v/pretrain.sh
    ```

### 后续处理

如需转回Hugging Face格式，需运行权重转换脚本：

```shell
mm-convert WanConverter mm_to_hf \
--cfg.source_path <path for your saved weight/> \
--cfg.target_path ./converted_weights/Wan2.1-T2V-1.3B-Diffusers/transformer/
--cfg.hf_dir weights/Wan2.1-T2V-1.3B-Diffusers/transformer/
```

>[!CAUTION]  
>如基于Layer Zero进行训练，则需首先转换回初始权重后，再进行以上操作。

## 参考

多模态理解模型更多细节请参考《[Qwen2_5_VL 使用指南](../../../examples/qwen2.5vl/README.md)》。

多模态生成模型更多细节请参考《[Wan2.1 使用指南](../../../examples/wan2.1/README.md)》。
