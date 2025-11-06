# Diffusers

<p align="left">
</p>

- [FLUX-Kontext](#jump1)
  - [模型介绍](#模型介绍)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
    - [性能](#性能)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [性能](#jump3)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# FLUX-Kontext

## 模型介绍

[FLUX.1-Kontext-dev](https://bfl.ai/models/flux-kontext) 是基于FLUX，当前先进的上下文图像生成与编辑技术的生成模型，它可以结合文本与图像，实现精确、连贯的生成效果。

## 版本说明

### 参考实现

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=c222570a9b47901266fecf34222f540870c3bb1b
  ```

### 变更记录

2025.09.08：首次发布Flux-Kontext

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

1. 软件与驱动安装

    ```bash
    # 安装 torch 和 torch_npu，参考上述安装指南进行安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 FLUX-Kontext [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout c222570
    cp -r ../MindSpeed-MM/examples/diffusers/flux-kontext/* ./examples/dreambooth
    cp ../MindSpeed-MM/sources/images/flux_cat.png ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/mm_requirements_kontext.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    - 用户需自行获取并解压[kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    ```shell
    dataset_name="kontext-community/relighting" # 数据集路径
    ```

   - kontext-community/relighting数据集格式如下:

    ```shell
    relighting
    ├── .gitattributes
    ├── README.MD
    └── data
          └── train-00000-of-00001.parquet
    ```

2. 【配置 FLUX-Kontext 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.1-Kontext-dev模型](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) `model_name`模型

    ```bash
    export model_name="black-forest-labs/FLUX.1-Kontext-dev" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="black-forest-labs/FLUX.1-Kontext-dev" # 预训练模型路径
    batch_size=2
    max_train_steps=5000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    1. 打开`train_dreambooth_lora_flux_kontext.py`文件

        ```shell
        cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
        vim train_dreambooth_lora_flux_kontext.py # 进入Python文件
        ```

        - 在import栏`if is_wandb_available():`上方（71行附近添加代码）

        ```python
        # 添加代码到train_dreambooth_lora_flux_kontext.py 71行附近
        from transformer_patches import apply_patches
        apply_patches()

        if is_wandb_available(): # 原代码
          import wandb
        ```

        - 在train_dataloader前修改batch_sample,将`数据集drop_last修改为False`在1645行附近

        ```python
        # 修改drop_last为False：
        batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=False)
        # batch_sampler = BucketBatchSampler(train_dataset, batch_size=args.train_batch_size, drop_last=True) # 原代码
        ```

    2. 【Optional】Ubuntu系统需在1701行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    3. 【Optional】多机运行

        修改config文件

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        将文件中的`deepspeed_multinode_launcher`, `main_process_ip`, 以及`main_process_port`消除注释而进行使用。

        ```shell
            zero_stage: 2
          #  deepspeed_multinode_launcher: standard
          # main_process_ip: localhost  # 主节点IP
          # main_process_port: 6000     # 主节点port
          machine_rank: 0             # 当前机器的rank
          num_machines: 1             # 总共的机器数
          num_processes: 8            # 总共的卡数
        ```

        如运行双机：
        - 将两台机器的yaml文件的main_process_ip与main_process_port设置成一样的主节点与port
        - 一台节点`machine_rank: 0`，另一台`machine_rank: 1`
        - 两台机器均设置`num_machines: 2`，`num_processes: 16`

4. 【启动 FLUX-Kontext LoRA微调脚本】

    本任务主要提供train_dreambooth_lora_flux_kontext微调脚本，支持多卡训练。

    启动微调脚本

    ```shell
    bash finetune_kontext_dreambooth_lora_deepspeed_bf16.sh 
    ```

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Resolution | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-Kontext LoRA微调  |  1.97  |     2      | bf16 | 1024 | ✔ |
| 竞品A | 8p | Flux-Kontext LoRA微调  |  2.00 |     2      | bf16 | 1024 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【FLUX-Kontext模型推理】

```shell
vim infer_kontext_text2img_bf16.py # 进入运行推理的Python文件
```

  1. 修改路径

      ```python
      MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"  # FLUX模型路径
      ```

  2. 运行代码

      ```shell
      python infer_kontext_text2img_bf16.py
      ```

  【lora微调FLUX-Kontext模型推理】

  ```shell
  vim infer_kontext_text2img_lora_bf16.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "black-forest-labs/FLUX.1-Kontext-dev"  # Flux 模型路径
      LORA_WEIGHTS = "./logs/pytorch_lora_weights.safetensors"  # LoRA权重路径
      ```

  2. 运行代码

      ```shell
      python infer_kontext_text2img_lora_bf16.py
      ```

<a id="jump3"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  LoRA文生图  | 1.04 | bf16 | 2.7.1 |
| 竞品A | 8p |  LoRA文生图  | 1.04 | bf16 | 2.7.1 |

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

## 引用

### 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/public_address_statement.md)
