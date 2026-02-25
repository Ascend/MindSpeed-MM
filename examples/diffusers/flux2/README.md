# Diffusers

<p align="left">
</p>

- [FLUX2](#jump1)
  - [模型介绍](#模型介绍)
  - [环境搭建](#环境搭建)
  - [微调T2I](#jump2)
    - [准备工作](#jump3)
    - [性能](#jump4)
  - [微调Img2Img](#jump5)
    - [准备工作](#jump6)
    - [性能](#jump7)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [推理T2I](#jump8)
    - [推理I2I](#jump9)
    - [性能](#jump10)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# FLUX2

## 模型介绍

[FLUX.2 dev](https://blackforestlabs.ai/announcing-black-forest-labs/) 是一种基于Rectified Flow Transformers (矫正流) 的生成模型。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=29a930a
  ```

## 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

1. 软件与驱动安装

    ```bash
    # 安装 torch 和 torch_npu，参考上述安装指南进行安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd MindSpeed-MM
    ```

3. 模型搭建

    3.1 【下载 FLUX2 [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装依赖,进入代码路径】

    ```shell
    bash examples/diffusers/flux2/install.sh
    cd ../diffusers/examples/dreambooth
    ```

<a id="jump2"></a>

## 微调T2I

<a id="jump3"></a>

### 准备工作

1. 【准备微调数据集】

    - 用户需自行获取并解压[3d-icon](https://huggingface.co/datasets/linoyts/3d_icon)数据集，并在`finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh`脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    打开脚本：

    ```shell
    vim finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

    ```shell
    dataset_name="linoyts/3d_icon" # 数据集 路径
    ```

   - 3d_icon数据集格式如下:

    ```shell
    3d_icon
    ├── metadata.jsonl
    ├── README.MD
    ├── gitattributes
    ├── 00.jpg
    ├── 01.jpg
    ├── ...jpg
    └── 22.jpg
    ```

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >

    - 如用自己的微调数据集，需在shell脚本中修改`dataset_name`：

    ```shell
    dataset_name="/path/customized_datasets" # 数据集路径
    ```

    在shell脚本`accelerate launch`目录下（58行左右）将修改 `dataset_name=$dataset_name`，并将`instance_prompt`改为与自己数据集所匹配的prompt，`caption_column`修改为数据集匹配名称，如用3dicon数据集，则无需修改:

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_lora_flux2.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --caption_column="prompt" \
      --instance_prompt="a prompt that is suitable for your own dataset" \
    ```

2. 【配置 FLUX2 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.2-dev模型](https://huggingface.co/black-forest-labs/FLUX.2-dev) `model_name`模型

    获取对应的微调模型后，在启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    打开脚本：

    ```shell
    vim finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

    ```shell
    model_name="black-forest-labs/FLUX.2-dev" # 预训练模型路径
    batch_size=1
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    ```shell
    vim train_dreambooth_lora_flux2.py # 进入Python文件
    ```

    1. 【Optional】Ubuntu系统需在1879行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    2. 【Optional】如机器未联网，需对save_model_card进行修改：
        将save_model_card删除或者放到args.push_to_hub目录下：

        ```python
        elif args.bnb_quantization_config_path:
            quant_training = "BitsandBytes" # 原代码
        if args.push_to_hub:
            save_model_card(
                (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
                quant_training=quant_training,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # 原代码

        ```

    3. 【Optional】多机运行

        修改config文件

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        将文件中的`main_process_ip`, 以及`main_process_port`消除注释而进行使用。

        ```shell
          main_process_ip: localhost  # 主节点IP
          main_process_port: 6000     # 主节点port
          machine_rank: 0             # 当前机器的rank
          num_machines: 1             # 总共的机器数
          num_processes: 8            # 总共的卡数
        ```

4. 【启动 Flux2 T2I 微调脚本】

    本任务主要提供dreambooth_lora_flux2_t2i微调脚本，支持多卡训练。

    启动Flux2 T2I dreambooth_lora微调脚本

    ```shell
    bash finetune_t2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

<a id="jump4"></a>

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | Resolution | AMP_Type | Torch_Version | FSDP2 |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-全参微调  |  1.28  | 1 | 1024 | bf16 | 2.7.1 | ✔ |
| 竞品A | 8p | Flux-全参微调  |  1.24 | 1 | 1024 | bf16 | 2.7.1 | ✔ |

<a id="jump5"></a>

## 微调Img2Img

<a id="jump6"></a>

### 准备工作

1. 【准备微调数据集】

    - 用户需自行获取并解压[kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    vim finetune_i2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

    ```shell
    dataset_name="kontext-community/relighting" # 数据集路径
    ```

   - kontext-community/relighting数据集格式如下:

    ```shell
    relighting
    ├── .gitattributes
    ├── README.md
    └── data
          └── train-00000-of-00001.parquet
    ```

2. 【配置 FLUX2 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.2-dev模型](https://huggingface.co/black-forest-labs/FLUX.2-dev) `model_name`模型

    ```bash
    export model_name="black-forest-labs/FLUX.2-dev" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="black-forest-labs/FLUX.2-dev" # 预训练模型路径
    batch_size=1
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    ```shell
    vim train_dreambooth_lora_flux2_img2img.py # 进入Python文件
    ```

    1. 【Optional】Ubuntu系统需在1796行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    2. 【Optional】如机器未联网，需对save_model_card进行修改：
        将save_model_card删除或者放到args.push_to_hub目录下：

        ```python
        validation_prompt = args.validation_prompt if args.validation_prompt else args.final_validation_prompt # 原代码
        if args.push_to_hub:
            save_model_card(
                (args.hub_model_id or Path(args.output_dir).name) if not args.push_to_hub else repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                instance_prompt=args.instance_prompt,
                validation_prompt=validation_prompt,
                repo_folder=args.output_dir,
                fp8_training=args.do_fp8_training,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # 原代码

        ```

    3. 【Optional】多机运行

        修改config文件

        ```bash
        vim bf16_accelerate_config.yaml

        ```

        将文件中的`main_process_ip`, 以及`main_process_port`消除注释而进行使用。

        ```shell
          main_process_ip: localhost  # 主节点IP
          main_process_port: 6000     # 主节点port
          machine_rank: 0             # 当前机器的rank
          num_machines: 1             # 总共的机器数
          num_processes: 8            # 总共的卡数
        ```

4. 【启动 Flux2 Img2Img 微调脚本】

    本任务主要提供dreambooth_lora_flux2_i2i微调脚本，支持多卡训练。

    启动Flux2 I2I dreambooth_lora微调脚本

    ```shell
    bash finetune_i2i_flux2_dreambooth_lora_fsdp_bf16.sh
    ```

<a id="jump7"></a>

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | Resolution | AMP_Type | Torch_Version | FSDP2 |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-全参微调  |  0.61  | 1 | 1024 | bf16 | 2.7.1 | ✔ |
| 竞品A | 8p | Flux-全参微调  |  0.6 | 1 | 1024 | bf16 | 2.7.1 | ✔ |

## 推理

### 环境搭建及运行

  **同[环境搭建](#环境搭建)对应章节**

<a id="jump8"></a>

### 推理T2I

进入运行T2I推理任務的Python文件

```shell
vim infer_flux2_text2img.py 
```

1. 修改路径

    ```python
    MODEL_PATH = "black-forest-labs/FLUX.2-dev"  # FLUX模型路径
    ```

    若使用lora微调FLUX2模型推理，需修改LORA_WEIGHTS参数：

    ```python
    LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
    ```

2. 运行代码

    - 因需要使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

    ```shell
    accelerate launch --num_processes=4 infer_flux2_text2img.py # 单机四卡进行分布式推理
    ```

<a id="jump9"></a>

### 推理I2I

进入运行I2I推理任務的Python文件

```shell
vim infer_flux2_img2img.py 
```

1. 修改路径

    ```python
    MODEL_PATH = "black-forest-labs/FLUX.2-dev"  # FLUX模型路径
    IMAGE = "./infer_result/flux2.fsdp_ulysses4.png"  # 需要编辑的图片路径
    PROMPT = "Change the crab to a dog"  # 编辑任务所需要的prompt
    ```

    若使用lora微调FLUX2模型推理，需修改LORA_WEIGHTS参数：

    ```python
    LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
    ```

2. 运行代码

    - 因需要使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

    ```shell
    accelerate launch --num_processes=4 infer_flux2_img2img.py # 单机四卡进行分布式推理
    ```

<a id="jump10"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  文生图  | 1.14 | bf16 | 2.7.1 |
| 竞品A | 8p |  文生图  | 1.05 | bf16 | 2.7.1 |
| Atlas 900 A2 PODc |8p |  图生图  | 1.14 | bf16 | 2.7.1 |
| 竞品A | 8p |  图生图  | 1.04 | bf16 | 2.7.1 |

## 环境变量声明

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
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复 <br> `2`: 开启多流复用特性用                                                               |

## 引用

### 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/public_address_statement.md)
