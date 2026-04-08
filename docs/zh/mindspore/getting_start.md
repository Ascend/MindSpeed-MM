# 快速入门：Qwen2.5-VL-7B模型微调

以MindSpore AI套件为后端的MindSpeed MM同时支持了部分多模态生成和多模态理解模型。下面介绍典型模型Qwen2.5-VL在MindSpore后端下的使用方法，引导开发者快速上手预置模型在MindSpore + 昇腾NPU上的高效运行。

## 多模态理解模型

以Qwen2.5-VL-7B模型为例，介绍多模态理解模型的高效运行方式。

### 环境准备

1. 基于MindSpore框架和Python3.10完成模型训练环境的安装，具体请参见[MindSpeed MM安装指导](../install_guide.md)。
2. 在`MindSpeed-MM`下创建以下目录用于存储日志、数据及权重文件：

    ```bash
    mkdir logs
    mkdir data
    mkdir ckpt
    ```

### 权重下载及转换

1. 权重下载

    从Hugging Face库下载对应的模型权重[Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)。

2. 权重文件保存

   创建`ckpt/hf_path/Qwen2.5-VL-7B-Instruct`目录并将下载的模型权重保存到该目录下

3. 权重转换

    MindSpeed MM修改了部分原始网络的结构名称，使用`mm-convert`工具对原始预训练权重进行转换。

    以下是将Hugging Face权重转为MindSpeed MM权重的转换示例：

    ```bash
    # 7b
    mm-convert  Qwen2_5_VLConverter hf_to_mm \
    --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
    --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
    --cfg.parallel_config.llm_pp_layers [[1,10,10,7]] \
    --cfg.parallel_config.vit_pp_layers [[32,0,0,0]] \
    --cfg.parallel_config.tp_size 1
    ```

    **表 1** 权重转换工具参数表

    |参数|说明|是否必选|默认值|
    |-|-|-|-|
    |mm_dir|转换后保存目录|是|/|
    |hf_dir|Hugging Face权重目录|是|/|
    |llm_pp_layers|llm在每个卡上切分的层数，注意要和examples/qwen2.5vl/model_7b.json中配置的pipeline_num_layers一致|否|[1,10,10,7]|
    |vit_pp_layers|vit在每个卡上切分的层数，注意要和examples/qwen2.5vl/model_7b.json中配置的pipeline_num_layers一致|否|[32,0,0,0]|
    |tp_size|TP并行数量，注意要和微调启动脚本中的配置一致|否|1|

    > [!NOTE]  
    > 由于Qwen2_5_VL和Qwen2_VL在权重转换逻辑上保持一致，详情参考[权重转换命令行工具](../features/mm_convert.md)。

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

   在`examples/qwen2.5vl/data_7b.json`中完成数据集路径的配置，配置示例如下：

    ```json
    {
        "dataset_param": {
            "dataset_type": "huggingface",
            "preprocess_parameters": {
                "model_name_or_path": "./ckpt/hf_path/Qwen2.5-VL-7B-Instruct",
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
    |model_name_or_path|权重|"./ckpt/hf_path/Qwen2.5-VL-7B-Instruct"，与[权重下载及转换](#权重下载及转换)中的`hf_config.hf_dir`一致。|
    |dataset_dir|数据集目录|"./data"|
    |dataset|数据集|"./data/mllm_format_llava_instruct_data.json"|

    > [!CAUTION]   
    > `cache_dir`在多机上不要配置同一个挂载目录避免写入同一个文件导致冲突。

2. 编辑微调示例脚本

    ```shell
    vi examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
    ```

3. 模型保存加载及日志信息配置

    ```bash
    ...
    # 加载路径
    LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
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

    若需要加载指定迭代次数的权重、优化器等状态，需将加载路径`LOAD_PATH`设置为保存文件夹路径`LOAD_PATH="save_dir"`，并修改`latest_checkpointed_iteration.txt`文件内容为指定迭代次数。

    ```bash
    $save_dir
    ├── latest_checkpointed_iteration.txt
    ├── ...
    ```

    **表 3** 参数配置解析

    |参数|说明|取值|
    |-|-|-|
    |LOAD_PATH|加载路径|/|
    |SAVE_PATH|保存路径|/|
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
    NPUS_PER_NODE=8         # 使用单节点的8卡NPU  
    MASTER_ADDR=localhost   # 单机使用本节点ip
    MASTER_PORT=29501       # 本节点端口号为29501
    NNODES=1                # 根据参与节点数量配置，单机为1
    NODE_RANK=0             # 单机RANK为0
    WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
        ```

5. 启动微调

    保存微调脚本后，启动微调任务，命令如下：

    ```shell
    bash examples/qwen2.5vl/finetune_qwen2_5_vl_7b.sh
    ```

### 后续处理

MindSpeed MM修改了部分原始网络的结构名称，在微调后，如果需要将权重转回Hugging Face格式，可使用`mm-convert`权重转换工具对微调后的权重进行转换，将权重名称修改为与原始网络一致。

以下是mm2hf的转换示例：

```bash
mm-convert  Qwen2_5_VLConverter mm_to_hf \
--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
--cfg.parallel_config.llm_pp_layers [1,10,10,7] \
--cfg.parallel_config.vit_pp_layers [32,0,0,0] \
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

如果需要使用转换的模型进行训练，同步修改`examples/mindspore/qwen2.5vl/finetune_qwen2_5_vl_7b.sh`中的`LOAD_PATH`参数，该路径为转换后或者切分后的权重目录，注意与原始权重 `ckpt/hf_path/Qwen2.5-VL-7B-Instruct`进行区分。

```shell
LOAD_PATH="ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
```
