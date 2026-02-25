# Diffsynth-Studio

<p align="left">
</p>

- [Qwen Image Edit](#jump1)
  - [模型介绍](#模型介绍)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# Qwen Image Edit

## 模型介绍

Qwen Image Edit 是基于 Qwen Image 基础模型扩展的图像编辑功能，通过引入输入图像编辑机制实现对现有图像的修改能力。Qwen Image Edit 继承了 Qwen Image 的完整Dit架构 QwenImageDit 和其他核心组件 QwenImageTextEncoder、QwenImageVAE，使用 Qwen2VLProcessor 处理包含图像和文本的编辑指令，采用特殊的提示模板，将编辑图像和文本指令联合编码。Qwen Image 系列的核心结构创新在于采用 MSRoPE 多模态位置编码解决文本与图像位置混淆问题；功能上以卓越的多语言文本渲染（尤其中文）和精准图像编辑为特色，同时具备强大的通用图像生成能力。

## 版本说明

### 参考实现

  ```shell
  url=https://github.com/modelscope/Diffsynth-Studio
  commit_id=084bc2fc78422fd15b37f7a8db02ad924eaf2917
  ```

### 变更记录

2025.11.18: 首次发布Qwen Image Edit

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

1. 软件与驱动安装

    ```bash
    # 创建并激活python环境，安装 torch 和 torch_npu，请参考上述安装指南

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 Qwen Image Edit 项目 [GitHub参考实现](https://github.com/modelscope/Diffsynth-Studio) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/modelscope/DiffSynth-Studio.git
    cd DiffSynth-Studio
    git checkout 084bc2f
    bash ../MindSpeed-MM/examples/diffsynth/qwen_image_edit/replace_npu_patch.sh
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    pip install -r requirements.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【下载权重】

    用户可访问modelscope官网自行下载完整模型库[Qwen Image Edit 模型权重](https://modelscope.cn/models/Qwen/Qwen-Image-Edit) 

2. 【准备微调数据集】

    用户可自行获取数据集[UltraEdit 数据集](https://huggingface.co/datasets/BleachNick/UltraEdit)，该数据集规模庞大，可仅下载一个parquet文件。parquet格式数据需要转换为下文规定的格式。用户也可以自行构建数据集，构建后的数据集应符合以下几点要求：

    - 数据集文件夹应按照以下规范形式存放

      ```shell
      dataset_name
      ├── metadata_edit.csv
      └── images
            ├── edited_00010000.jpg
            ├── source_00010000.jpg
            └── ......
      ```
    
      > **说明：**
      >`dataset_name` 代表数据集名称。
      >`metadata_edit.csv` 是结构化数据索引/配置文件，使用表格的形式记录数据的路径、标签、元数据、配对关系等信息。
      >`images` 是存放图像的文件夹，所有图像均存放在此文件夹中。
      >

    - `metadata_edit.csv` 对图像编辑任务所需的一对输入图像进行信息记录的示例如下：

      | image | edit_image | prompt | sample_id |
      | :----: | :----: | :----: | :----: |
      | images/source_00010000.jpg | images/edited_00010000.jpg | transform the bird into a butterfly | 10000 |
      | images/source_00010001.jpg | images/edited_00010001.jpg | change the cat's fur color to orange | 10001 |


      > **说明：**
      >`image` 是编辑前的图像，图像命名打头为`source_`，后跟图像序号；`edit_image` 则是编辑后的图像，图像命名打头为`edited_`，后跟图像序号。
      >`prompt` 是图像编辑指令。
      >`sample_id` 是图像序号，一对`image`和`edit_image`使用共同的图像序号。
      >

3. 【配置 Lora 微调脚本】

   下载好权重和数据集之后，即可根据实际存放路径修改模型微调shell脚本`train_qwen_image_edit_lora.sh`

   - 根据权重路径修改`transformer_path`、`text_encoder_path`、`model_paths`、`tokenizer_path`、`processor_path`
      `model_paths`存放transformer、text_encoder、vae三个模型组件的权重路径，对于使用多个分片文件保存的权重，需要使用`[]`将其囊括并使用`,`和换行符相分隔，修改后的`model_paths`参数示例如下：
      
      ```shell
      transformer_path="Qwen/Qwen-Image-Edit/transformer"
      text_encoder_path="Qwen/Qwen-Image-Edit/text_encoder"
      model_paths='[
          [
              "'"${transformer_path}"'/diffusion_pytorch_model-00001-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00002-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00003-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00004-of-00005.safetensors",
              "'"${transformer_path}"'/diffusion_pytorch_model-00005-of-00005.safetensors"
          ],
          [
              "'"${text_encoder_path}"'/model-00001-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00002-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00003-of-00004.safetensors",
              "'"${text_encoder_path}"'/model-00004-of-00004.safetensors"
          ],
          "/path/Qwen-Image-Edit/vae/diffusion_pytorch_model.safetensors"
      ]'
      ```
    
      `tokenizer_path`存放tokenizer地址，`processor_path`存放processor地址：

      ```shell
      tokenizer_path="Qwen/Qwen-Image-Edit/tokenizer"
      processor_path="Qwen/Qwen-Image-Edit/processor"
      ```

    - 根据数据集路径修改`dataset_base_path`、`dataset_metadata_path`
      `dataset_base_path`存放所构建数据集的`dataset_name`，`dataset_metadata_path`则存放csv文件地址：

      ```shell
      dataset_base_path="/path/dataset"
      dataset_metadata_path="/path/dataset/metadata_edit.csv"
      ```

    - 根据分布式训练配置修改accelerate配置
      `accelerate_config.yaml`默认设置单机8卡进行zerostage2切分策略训练，如需更改lora训练的配置，请在`./examples/qwen_image/model_training/lora/accelerate_config.yaml`中进行修改，同时将该文件的相对路径放置在`config_file`参数后。

    - 根据数据集规模、`dataset_repeat`、`num_epochs`控制train steps：
      `dataset_repeat`指数据集的样本重复次数、`num_epochs`则指训练epoch数。`dataset_repeat`应设置得较小，以免训练时显存溢出。Qwen Image Edit的训练步数由以下公式求得：

      ```shell
      training_steps = len(dataloader) × num_epochs
      ```


4. 【修改代码文件】

    1. 根据NPU特性使用patch
    
      打开`train.py`文件

        ```shell
        vim ./examples/qwen_image/model_training/train.py # 进入Python文件
        ```

        - 在import栏（第4行之前插入代码）

        ```python
        # 添加代码
        from diffsynth.models.qwen_image_edit_patch import apply_patches
        apply_patches()

        from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig # 原代码
        ```

    2. 【Optional】多机运行

        修改config文件

        ```bash
        vim ./examples/qwen_image/model_training/lora/accelerate_config.yaml
        ```

        将文件中的`deepspeed_multinode_launcher`, `main_process_ip`, 以及`main_process_port`消除注释而进行使用，主节点IP在组成多机的所有机器中保持一致，根据实际使用需要修改`num_machines`、`num_processes`。所有机器的CANN版本应保持一致。

        ```shell
            zero_stage: 2
            deepspeed_multinode_launcher: standard
          main_process_ip: localhost  # 主节点IP
          main_process_port: 6000     # 主节点port
          machine_rank: 0             # 当前机器的rank
          num_machines: 1             # 总共的机器数
          num_processes: 8            # 总共的卡数
        ```

5. 【启动 Qwen Image Edit微调脚本】

    本任务主要提供train_qwen_image_edit_lora微调脚本，支持多卡训练。

    启动 Qwen Image Edit lora微调脚本

    ```shell
    bash examples/qwen_image/model_training/lora/train_qwen_image_edit_lora.sh  
    ```

### 微调性能

#### 吞吐

Qwen Image Edit 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Qwen Image Edit-LoRA微调  |  20.59  |     8      | bf16 | 2.7.1 | ✔ |
| 竞品A | 8p | Qwen Image Edit-LoRA微调  | 17.47  |     8      | bf16 | 2.7.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

【Qwen Image Edit模型推理】

  ```shell
  vim ./examples/qwen_image/model_inference/inference_qwen_image_edit_bf16.py # 进入运行推理的Python文件
  ```

  1. 修改路径

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = "${transformer_path}/diffusion_pytorch_model*.safetensors"

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = "${text_encoder_path}/model*.safetensors"

    vae_file = "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"

    tokenizer_file = "Qwen/Qwen-Image/tokenizer"

    processor_file = "Qwen/Qwen-Image/processor"
    ```

    `transformer_files`和`text_encoder_files`若为分片保存的多文件权重，需要使用`[]`将其囊括并使用`,`和换行符相分隔：

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = [
              "${transformer_path}/diffusion_pytorch_model-00001-of-00005.safetensors",
              "${transformer_path}/diffusion_pytorch_model-00002-of-00005.safetensors",
              "${transformer_path}/diffusion_pytorch_model-00003-of-00005.safetensors",
              "${transformer_path}/diffusion_pytorch_model-00004-of-00005.safetensors",
              "${transformer_path}/diffusion_pytorch_model-00005-of-00005.safetensors"
          ]

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = [
              "${text_encoder_path}/model-00001-of-00004.safetensors",
              "${text_encoder_path}/model-00002-of-00004.safetensors",
              "${text_encoder_path}/model-00003-of-00004.safetensors",
              "${text_encoder_path}/model-00004-of-00004.safetensors"
          ]
    ```

  2. 创建推理结果路径
      ```shell
      mkdir -p inference
      ```

  3. 运行代码

      ```shell
      source /usr/local/Ascend/cann/set_env.sh
      python examples/qwen_image/model_inference/inference_qwen_image_edit_bf16.py
      ```

【lora微调Qwen Image Edit模型推理】

  ```shell
  vim ./examples/qwen_image/model_inference/inference_qwen_image_edit_lora_bf16.py
  ```

  1. 修改路径

    ```python
    transformer_path = "Qwen/Qwen-Image-Edit/transformer"
    transformer_files = "${transformer_path}/diffusion_pytorch_model*.safetensors"

    text_encoder_path = "Qwen/Qwen-Image-Edit/text_encoder"
    text_encoder_files = "${text_encoder_path}/model*.safetensors"

    vae_file = "Qwen/Qwen-Image/vae/diffusion_pytorch_model.safetensors"

    tokenizer_file = "Qwen/Qwen-Image/tokenizer"

    processor_file = "Qwen/Qwen-Image/processor"

    lora_path = "Qwen-Image-LoRA/model.safetensors"
    ```

    `lora_path`是存放 lora 权重的绝对路径，其他权重同上

  2. 创建推理结果路径
      ```shell
      mkdir -p inference
      ```

  3. 运行代码

      ```shell
      source /usr/local/Ascend/cann/set_env.sh
      python examples/qwen_image/model_inference/inference_qwen_image_edit_lora_bf16.py
      ```
  
<a id="jump3"></a>

### 环境变量声明

| 环境变量                          | 描述                                                                 | 取值说明                                                                                                               |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                                |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志   |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                                           |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                                          |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                             |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                            |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征 |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                        |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                     |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                        |
| `OMP_NUM_THREADS`             | 设置执行期间使用的线程数    |      需要配置为整数                                                  |


## 引用

### 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/public_address_statement.md)
