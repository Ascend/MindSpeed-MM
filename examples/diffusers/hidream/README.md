# Diffusers

<p align="left">
</p>

- [HiDream](#jump1)
  - [模型介绍](#模型介绍)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
    - [微调性能](#微调性能)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [推理性能](#jump3)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# HiDream-I1

## 模型介绍

HiDream-I1是一个开源的图像生成模型，它使用了动态MoE架构的双流解耦稀疏DiT，第一阶段通过两个独立的编码器处理图像和文本，随后采用动态MoE的单流稀疏DiT进行处理。

## 版本说明

### 参考实现

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=d72184eba358b883d7186a0a96dedd8118fcb72a
  ```

### 变更记录

2025.06.25：首次发布HiDream-I1

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
    pip install torch-2.7.1-cp310-cp310m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.7.1*-cp310-cp310m-linux_aarch64.whl
    
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

    3.1 【下载 HiDream [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout d72184e
    cp -r ../MindSpeed-MM/examples/diffusers/hidream/* ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/mm_requirements_hidream.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    - 用户需自行获取并解压[3d-icon](https://huggingface.co/datasets/linoyts/3d_icon)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

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
      ./train_dreambooth_lora_hidream.py \
      --pretrained_model_name_or_path=$model_name  \
      --pretrained_tokenizer_4_name_or_path=$pretrained4_path \
      --pretrained_text_encoder_4_name_or_path=$pretrained4_path \
      --dataset_name=$dataset_name \
      --caption_column="prompt" \
      --instance_prompt="a prompt that is suitable for your own dataset" \
    ```

2. 【配置 Lora 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[HiDream-I1-Full模型](https://huggingface.co/HiDream-ai/HiDream-I1-Full) `model_name`模型,与[Llama-3.1-8B-Instruct模型](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) `pretrained4_path`模型

    ```shell
    model_name="HiDream-ai/HiDream-I1-Dev" # 预训练模型路径
    pretrained4_path="meta-llama/Meta-Llama-3.1-8B-Instruct" # Forth pretrained path
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="HiDream-ai/HiDream-I1-Full"
    pretrained4_path="meta-llama/Meta-Llama-3.1-8B-Instruct" 
    dataset_name="linoyts/3d_icon"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=512
    gradient_accumulation_steps=1
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    1. 打开`train_dreambooth_lora_hidream.py`文件

        ```shell
        cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
        vim train_dreambooth_lora_hidream.py # 进入Python文件
        ```

        - 在import栏/`if is_wandb_available():`上方（71行附近添加代码）

        ```python
        # 添加代码到train_dreambooth_lora_hidream.py 71行附近
        from transformer_patches import apply_patches
        apply_patches()

        if is_wandb_available(): # 原代码
          import wandb
        ```

    2. 【Optional】Ubuntu系统需在1701行附近 添加 `accelerator.print("")`，如不添加终端显示可能不会同步更新。

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    3. 【Optional】如机器未联网，需对save_model_card进行修改：
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
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            ) # 原代码

        ```

    4. 【Optional】多机运行

        修改config文件

        ```bash
        vim bf16_accelerate_config.yaml
        ```

        将文件中的`deepspeed_multinode_launcher`, `main_process_ip`, 以及`main_process_port`消除注释而进行使用。

        ```shell
            zero_stage: 2
            deepspeed_multinode_launcher: standard
          main_process_ip: localhost  # 主节点IP
          main_process_port: 6000     # 主节点port
          machine_rank: 0             # 当前机器的rank
          num_machines: 1             # 总共的机器数
          num_processes: 8            # 总共的卡数
        ```

4. 【启动 HiDream 微调脚本】

    本任务主要提供dreambooth_lora_hidream微调脚本，支持多卡训练。

    启动HiDream dreambooth_lora微调脚本

    ```shell
    bash finetune_hidream_dreambooth_lora_deepspeed_bf16.sh
    ```

### 微调性能

#### 吞吐

HiDream 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | HiDream-全参微调  |  18.37  |     8      | bf16 | 2.6 | ✔ |
| 竞品A | 8p | HiDream-全参微调  | 19.61  |     8      | bf16 | 2.6 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【Hidream模型推理】

对PROMPTS及推理时所需参数进行修改

```shell
vim prompt_utils.py 
```

对推理文件所需权重路径进行修改

```shell
vim infer_hidream_text2img_bf16.py # 进入运行推理的Python文件
```

- 修改路径

  ```python
  MODEL_PATH = "HiDream-ai/HiDream-I1-Full"  # Model path for HiDream
  FORTH_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # pretrained model path for tokenizer & text encoder
  ```

运行代码

```shell
python infer_hidream_text2img_bf16.py
```

【lora微调Hidream模型推理】

对PROMPTS及推理时所需参数进行修改

```shell
vim prompt_utils.py 
```

对推理文件所需权重路径进行修改

```shell
vim infer_hidream_text2img_lora_bf16.py
```

- 修改路径

  ```python
  MODEL_PATH = "HiDream-ai/HiDream-I1-Full"  # Model path for HiDream
  FORTH_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # pretrained model path for tokenizer & text encoder
  OUTPUT_PATH = "./infer_result"  # Output path
  ```

运行代码

```shell
python infer_hidream_text2img_lora_bf16.py
```
  
<a id="jump3"></a>

### 推理性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  文生图  | 1.27 | bf16 | 2.6 |
| 竞品A | 8p |  文生图  | 1.88 | bf16 | 2.6 |

### 环境变量声明

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
