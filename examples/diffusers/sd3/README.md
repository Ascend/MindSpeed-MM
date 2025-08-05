# Diffusers

<p align="left">
</p>

- [SD3/SD3.5](#jump1)
  - [模型介绍](#模型介绍)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
    - [性能](#性能)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# Stable Diffusion 3 & Stable Diffusion 3.5

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=5f724735437d91ed05304da478f3b2022fe3f6fb
  ```

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

1. 软件与驱动安装


    ```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    # 建议从原仓编译安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 SD3/SD3.5 [GitHub参考实现](https://github.com/huggingface/diffusers) 或 [适配昇腾AI处理器的实现](https://gitee.com/ascend/ModelZoo-PyTorch.git) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout 5f724735437d91ed05304da478f3b2022fe3f6fb
    cp -r ../MindSpeed-MM/examples/diffusers/sd3 ./sd3
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_sd3.txt #修改版本：torchvision==0.16.0, torch==2.7.1, accelerate==0.33.0, 添加deepspeed==0.15.2
    pip install -r examples/dreambooth/requirements_sd3.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    用户需自行获取并解压[pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    vim sd3/finetune_sd3_dreambooth_deepspeed_**16.sh
    vim sd3/finetune_sd3_dreambooth_fp16.sh
    ```

    ```shell
    dataset_name="pokemon-blip-captions" # 数据集 路径
    ```

   - pokemon-blip-captions数据集格式如下:

    ```shell
    pokemon-blip-captions
    ├── dataset_infos.json
    ├── README.MD
    └── data
          └── train-001.parquet
    ```

    - 只包含图片的训练数据集，如非deepspeed脚本使用训练数据集dog:[下载地址](https://huggingface.co/datasets/diffusers/dog-example)，在shell启动脚本中将`input_dir`参数设置为本地数据集绝对路径>

    ```shell
    input_dir="dog" # 数据集路径
    ```

    ```shell
    dog
    ├── alvan-nee-*****.jpeg
    ├── alvan-nee-*****.jpeg
    ```

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >

2. 【配置 SD3/SD3.5 微调脚本】

    【SD3】
    用户可访问huggingface官网自行下载[sd3-medium模型](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main) `model_name`模型

    ```bash
    export model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径
    ```

    【SD3.5】
    用户可访问huggingface官网自行下载[sd3.5-large模型](https://huggingface.co/stabilityai/stable-diffusion-3.5-large/tree/main) `model_name`模型

    ```bash
    export model_name="stabilityai/stable-diffusion-3.5-large" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径SD3与SD3.5为相同脚本

    ```shell
    scripts_path="./sd3" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径 （此为sd3）
    dataset_name="pokemon-blip-captions" 
    batch_size=4
    num_processors=8 # 卡数（为计算FPS使用，yaml文件里需同步修改）
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

    数据集选择：如果选择默认[原仓数据集](https://huggingface.co/datasets/diffusers/dog-example),需修改两处`dataset_name`为`input_dir`：

    ```shell
    input_dir="dog"

    # accelerator 修改 --dataset_name=#dataset_name
    --instance_data_dir=$input_dir
    ```

    | 数据集 | 路径设置 | accelerate 设置 |
    |:----------:|:----------:|:----------:|
    | dog | input_dir="dog" | --instance_data_dir=$input_dir; --instance_prompt="A photo of sks dog" |
    | pokemon | dataset_name="pokemon-blip-captions" | --dataset_name=$dataset_name --caption_column="text"; --instance_prompt="A photo of pokemon" |

    修改`fp16_accelerate_config.yaml`的`deepspeed_config_file`的路径:

    ```shell
    vim sd3/fp16_accelerate_config.yaml
    # 修改：
    deepspeed_config_file: ./sd3/deepspeed_fp16.json # deepspeed JSON文件路径
    ```

3. 【Optional】Ubuntu系统需在`train_dreambooth_sd3.py`1705行附近 与 `train_dreambooth_lora_sd3.py`1861行附近 添加 `accelerator.print("")`

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    # 或
    vim examples/dreambooth/train_dreambooth_lora_sd3.py
    ```

    如下：

    ```python
    if global_step >= args.max_train_steps: # 原代码
      break
    accelerator.print("") # 添加
    ```

4. 【如需保存checkpointing请修改代码】

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    ```

    - 在文件上方的import栏增加`DistributedType`在`from accelerate import Accelerator`后 （30行附近）
    - 在`if accelerator.is_main_process`后增加 `or accelerator.distributed_type == DistributedType.DEEPSPEED`（dreambooth在1681行附近），并在`if args.checkpoints_total_limit is not None`后增加`and accelerator.is_main_process`

    ```python
    from accelerate import Accelerator, DistributedType
    # from accelerate import Accelerator # 原代码
    from accelerate.logging import get_logger # 原代码
     
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
    # if accelerator.is_main_process: # 原代码 1681/1833行附近
      if global_step % args.checkpointing_steps == 0:  # 原代码 不进行修改
        if args.checkpoints_total_limit is not None and accelerator.is_main_process: # 添加
    ```

5. 【修改文件】

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    # 或
    vim examples/dreambooth/train_dreambooth_lora_sd3.py
    ```

    在log_validation里修改`pipeline = pipeline.to(accelerator.device)`，`train_dreambooth_sd3.py`在174行附近`train_dreambooth_lora_sd3.py`在198行附近

    ```python
    # 修改pipeline为：
    pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
    # pipeline = pipeline.to(accelerator.device) # 原代码
    ```

6. 【启动 SD3 微调脚本】

    本任务主要提供**混精fp16**和**混精bf16**dreambooth和dreambooth+lora的**8卡**训练脚本，使用与不使用**deepspeed**分布式训练。

    ```shell
    bash sd3/finetune_sd3_dreambooth_deepspeed_**16.sh #使用deepspeed,dreambooth微调 
    bash sd3/finetune_sd3_dreambooth_lora_deepspeed_fp16.sh #使用deepspeed,dreambooth微调 (sd3.5)
    bash sd3/finetune_sd3_dreambooth_fp16.sh #无使用deepspeed,dreambooth微调
    bash sd3/finetune_sd3_dreambooth_lora_fp16.sh #无使用deepspeed,dreambooth+lora微调 (sd3)
    ```

### 性能

#### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Resolution | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调  |   16.09 |     4      | bf16 | 1024 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调  |  16.01 |     4      | bf16 | 1024 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调 |  15.16 |     4      | fp16 | 1024 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调 |   15.53 |     4      | fp16 | 1024 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Dreambooth-全参微调 | 3.11  | 1 | fp16 | 1024 | 2.1 | ✘ |
| 竞品A | 8p | Dreambooth-全参微调 | 3.71 | 1 | fp16 | 1024 | 2.1 | ✘ |
| Atlas 900 A2 PODc |8p | DreamBooth-LoRA | 108.8 | 8 | fp16 | 1024 | 2.1 | ✘ |
| 竞品A | 8p | DreamBooth-LoRA | 110.69 | 8 | fp16 | 1024 | 2.1 | ✘ |

SD3.5 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Resolution | Torch_Version | deepspeed | gradient checkpointing |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调  |   26.24 |     8      | bf16 | 512 | 2.1 | ✔ | ✔ |
| 竞品A | 8p | Dreambooth-全参微调  |  28.33 |     8      | bf16 | 512 | 2.1 | ✔ | ✔ |
| Atlas 900 A2 PODc | 8p | Dreambooth-Lora |  47.93 |     8      | fp16 | 512 | 2.1 | ✔ | ✘ |
| 竞品A | 8p | Dreambooth-Lora |   47.95 |     8      | fp16 | 512 | 2.1 | ✔ | ✘ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

 【运行推理的脚本】

  图生图推理脚本需先准备图片：[下载地址](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png)
  修改推理脚本中预训练模型路径以及图生图推理脚本中的本地图片加载路径
  调用推理脚本

  ```shell
  cd sd3/ # # 进入sd3目录
  ```

【SD3/SD3.5模型推理】

```shell
vim infer_sd3_text2img.py # 进入运行T2I推理的Python文件
# 或
vim infer_sd3_img2img.py # 进入运行I2I推理的Python文件
```

  1. 修改路径

      ```python
      MODEL_PATH = "stabilityai/stable-diffusion-3.5-large"  # 路径可选择sd3/sd3.5模型权重 或 Dreambooth 微调后输出模型
      DTYPE = torch.float16 # 可选择混精模式
      ```

  2. 运行代码

      ```shell
      python infer_sd3_text2img.py  # 单卡推理，文生图
      python infer_sd3_img2img.py   # 单卡推理，图生图
      ```

  【lora微调SD3模型推理】

  ```shell
  vim infer_sd3_text2img_lora.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "stabilityai/stable-diffusion-3.5-large"  # 路径可选择sd3/sd3.5模型权重 或 Dreambooth 微调后输出模型
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
      ```

  2. 运行代码

      ```shell
      python infer_sd3_text2img_lora.py
      ```

  【分布式推理】

  ```shell
  vim infer_sd3_text2img_distrib.py
  ```

- 修改模型权重路径 model_path为模型权重路径或微调后的权重路径
- 如lora微调 可将lora_weights修改为Lora权重路径

  ```python
  model_path = "stabilityai/stable-diffusion-3.5-large"  # 模型权重/微调权重路径
  lora_weights = "/pytorch_lora_weights.safetensors"  # Lora权重路径
  ```

- 启动分布式推理脚本
  
  - 因使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

  ```shell
  accelerate launch --num_processes=4 infer_sd3_text2img_distrib.py # 单机四卡进行分布式推理
  ```

## 使用基线数据集进行评估

## 环境变量声明
ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
ASCEND_GLOBAL_EVENT_ENABLE： 设置应用类日志是否开启Event日志，0：关闭Event日志，1：开启Event日志  
TASK_QUEUE_ENABLE： 用于控制开启task_queue算子下发队列优化的等级，0：关闭，1：开启Level 1优化，2：开启Level 2优化  
COMBINED_ENABLE： 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景  
HCCL_WHITELIST_DISABLE： 配置在使用HCCL时是否开启通信白名单，0：开启白名单，1：关闭白名单  
CPU_AFFINITY_CONF： 控制CPU端算子任务的处理器亲和性，即设定任务绑核，设置0或未设置：表示不启用绑核功能， 1：表示开启粗粒度绑核， 2：表示开启细粒度绑核  
HCCL_CONNECT_TIMEOUT:  用于限制不同设备之间socket建链过程的超时等待时间，需要配置为整数，取值范围[120,7200]，默认值为120，单位s  
ACLNN_CACHE_LIMIT： 配置单算子执行API在Host侧缓存的算子信息条目个数  
TOKENIZERS_PARALLELISM： 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为  
PYTORCH_NPU_ALLOC_CONF： 控制缓存分配器行为  
OMP_NUM_THREADS： 设置执行期间使用的线程数

## 引用

### 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/public_address_statement.md)
