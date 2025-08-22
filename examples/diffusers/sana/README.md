# Diffusers

<p align="left">
</p>

- [SANA](#jump1)
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

# SANA

## 模型介绍

Sana是由NVIDIA、麻省理工学院和清华大学共同推出的文生图模型，通过使用深度压缩自编码器、Linear DiT、与Decoder Only的小型语言模型，能高效的生成高达4096x4096分辨率的高清图像。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=cd0a4a82cf8625b96e2889afee2fce5811b35c05
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

    3.1 【下载 Sana [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout cd0a4a82cf8625b96e2889afee2fce5811b35c05
    cp -r ../MindSpeed-MM/examples/diffusers/sana/* ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_sana.txt #修改版本：torch==2.7.1, accelerate==0.33.0, transformers==4.47.1 添加deepspeed==0.15.2
    pip install -r examples/dreambooth/requirements_sana.txt # 安装对应依赖
    ```

<a id="jump2"></a>

## 微调

1. 【准备微调数据集】

    - 用户需自行获取并解压[pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

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

    > **说明：**
    >该数据集的训练过程脚本只作为一种参考示例。
    >

    - 如用自己的微调数据集，需在shell脚本中修改`dataset_name`：

    ```shell
    dataset_name="/path/customized_datasets" # 数据集路径
    ```

    在shell脚本`accelerate launch`目录下（70行左右）将修改 `dataset_name=$dataset_name`，并将`instance_prompt`与`validation_prompt`改为与自己数据集所匹配的prompt:

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_lora_sana.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --instance_prompt="a prompt that is suitable for your own dataset" \
      --validation_prompt="a validation prompt based on your own dataset" \
    ```

2. 【配置 Lora 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[Sana 4K模型](https://huggingface.co/Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers) `model_name`模型

    ```shell
    export model_name="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers" # 预训练模型路径
    dataset_name="pokemon-blip-captions"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=1024
    gradient_accumulation_steps=1
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=8 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    1. 打开`train_dreambooth_lora_sana.py`文件

        ```shell
        cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
        vim train_dreambooth_lora_sana.py # 进入Python文件
        ```

        - 在import栏/`if is_wandb_available():`上方（69行附近添加代码）

        ```python
        # 添加代码到train_dreambooth_sana.py 69行附近
        from patch_sana import create_save_model_hook, create_load_model_hook

        if is_wandb_available(): # 原代码
          import wandb
        ```

        在986行附近修改vae的dtype为BF16

        ```python
        vae.to(dtype=torch.bfloat16)
        # vae.to(dtype=torch.float32) # 原码
        transformer.to(accelerator.device, dtype=weight_dtype) # 原码
        # because Gemma2 is particularly suited for bfloat16.
        text_encoder.to(dtype=torch.bfloat16) # 原码
        ```

    2. 【Optional】Ubuntu系统需在1701行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    3. 【Optional】模型checkpoint saving保存

        ```shell
        --checkpointing_steps=5001 \ # 修改5001步为所需要步数
        ```

        【如需保存checkpointing请修改代码】

        ```shell
        vim examples/dreambooth/train_dreambooth_lora_sana.py #（1788行附近）
        ```

        - 在文件上方的import栏增加`DistributedType`在`from accelerate import Accelerator`后 （31行附近）
        - 在`if accelerator.is_main_process`后增加 `or accelerator.distributed_type == DistributedType.DEEPSPEED` (1431行附近)，并在`if args.checkpoints_total_limit is not None`后增加`and accelerator.is_main_process`

        ```python
        from accelerate import Accelerator, DistributedType
        # from accelerate import Accelerator # 原代码

        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        # if accelerator.is_main_process: # 原代码
          if global_step % args.checkpointing_steps == 0:  # 原代码 不进行修改
            if args.checkpoints_total_limit is not None and accelerator.is_main_process: # 添加
        ```

        Lora任务需调用patch任务进行权重保存：
        在`train_dreambooth_lora_sana.py`文件中找到代码`accelerator.register_save_state_pre_hook(save_model_hook)`进行修改(1088行附近)，复制粘贴以下代码：

        ```python
        # 添加
        save_Model_Hook = create_save_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
        )
        load_Model_Hook = create_load_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
              args=args,
        )
        accelerator.register_save_state_pre_hook(save_Model_Hook) # 修改
        accelerator.register_load_state_pre_hook(load_Model_Hook) # 修改
        # accelerator.register_save_state_pre_hook(save_model_hook) # 原代码
        # accelerator.register_load_state_pre_hook(load_model_hook) # 原代码
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

4. 【启动 SANA 微调脚本】

    本任务主要提供sana_dreambooth_lora微调脚本，支持多卡训练。

    启动SANA dreambooth_lora微调脚本

    ```shell
    bash finetune_sana_dreambooth_lora_deepspeed_bf16.sh
    ```

### 性能

#### 吞吐

SANA 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Sana-全参微调  |  28.7  |     8      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Sana-全参微调  | 32.8  |     8      | bf16 | 2.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【SANA模型推理】

```shell
vim infer_sana_text2img_bf16.py # 进入运行推理的Python文件
```

  1. 修改路径

      ```python
      MODEL_PATH = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"  # SANA模型路径
      ```

  2. 运行代码

      ```shell
      python infer_sana_text2img_bf16.py
      ```

  【lora微调SANA模型推理】

  ```shell
  vim infer_sana_text2img_lora_bf16.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "./SANA"  # SANA 模型路径
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
      ```

  2. 运行代码

      ```shell
      python infer_sana_text2img_lora_bf16.py
      ```
  
  【分布式推理】

  ```shell
  vim infer_sana_text2img_distrib.py
  ```

- 修改模型权重路径 model_path为模型权重路径或微调后的权重路径
- 如lora微调 可将lora_weights修改为Lora权重路径

    ```python
    model_path = "Efficient-Large-Model/Sana_1600M_4Kpx_BF16_diffusers"  # 模型权重/微调权重路径
    lora_weights = "/pytorch_lora_weights.safetensors"  # Lora权重路径
    ```

- 启动分布式推理脚本

  - 因使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

  ```shell
  accelerate launch --num_processes=4 infer_sana_text2img_distrib.py # 单机四卡进行分布式推理
  ```

<a id="jump3"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  文生图  | 0.84 | bf16 | 2.1 |
| 竞品A | 8p |  文生图  | 1.04 | bf16 | 2.1 |

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
