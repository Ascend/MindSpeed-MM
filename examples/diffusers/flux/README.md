# Diffusers

<p align="left">
</p>

- [FLUX](#jump1)
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

# FLUX

## 模型介绍

[FLUX.1 dev](https://blackforestlabs.ai/announcing-black-forest-labs/) 是一种基于Rectified Flow Transformers (矫正流) 的生成模型。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id= a98a839de75f1ad82d8d200c3bc2e4ff89929081
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

    3.1 【下载 FLUX [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout a98a839de75f1ad82d8d200c3bc2e4ff89929081
    cp -r ../MindSpeed-MM/examples/diffusers/flux/* ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_flux.txt #修改版本： torch==2.7.1, accelerate==1.10.0, transformers==4.47.1, 添加deepspeed==0.17.2
    pip install -r examples/dreambooth/requirements_flux.txt # 安装对应依赖
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

    - 只包含图片的训练数据集，如deepspeed脚本使用训练数据集dog:[下载地址](https://huggingface.co/datasets/diffusers/dog-example)，并将dog文件夹转移到`examples/dreambooth/`目录下

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

    - 如用自己的微调数据集，需在shell脚本中将`input_dir`修改为`dataset_name`：

    ```shell
    dataset_name="/path/customized_datasets" # 数据集路径
    ```

    在shell脚本`accelerate launch`目录下（70行左右）将`instance_data_dir=$instance_dir \` 修改为 `dataset_name=$dataset_name`，并将`instance_prompt`与`validation_prompt`改为与自己数据集所匹配的prompt:

    ```shell
    # Example
    accelerate launch --config_file ${config_file} \
      ./train_dreambooth_flux.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --instance_prompt="a prompt that is suitable for your own dataset" \
      --validation_prompt="a validation prompt based on your own dataset" \
    ```

2. 【配置 FLUX 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[FLUX.1-dev模型](https://huggingface.co/black-forest-labs/FLUX.1-dev) `model_name`模型

    ```bash
    export model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="black-forest-labs/FLUX.1-dev" # 预训练模型路径
    batch_size=16
    max_train_steps=5000
    mixed_precision="bf16" # 混精
    resolution=256
    config_file="pretrain_${mixed_precision}_accelerate_config.yaml"

    # accelerate launch --config_file ${config_file} \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    1. 在 `src/diffusers/models/embeddings.py` 文件里，在 `class FluxPosEmbed(nn.Module):` 下的 **第813行左右** 找到代码： `freqs_dtype = torch.float32 if is_mps else torch.float64` 进行修改, 请修改为：`freqs_dtype = torch.float32`

        ```shell
        # 修改为freqs_dtype = torch.float32
        vim src/diffusers/models/embeddings.py
        ```

        ```python
        freqs_dtype = torch.float32 # 813行附近
        # freqs_dtype = torch.float32 if is_mps else torch.float64 # 原代码
        ```

    2. 打开`train_dreambooth_flux.py`或`train_dreambooth_lora_flux_advanced.py`文件

        ```shell
        cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
        vim train_dreambooth_flux.py # 进入Python文件
        # 如是flux lora，需先进入advanced_diffusion_training目录
        vim ../advanced_diffusion_training/train_dreambooth_lora_flux_advanced.py # 进入Python文件
        ```

        - 在import栏`if is_wandb_available():`上方（62行附近添加代码）

        ```python
        # 添加代码到train_dreambooth_flux.py 62行附近
        from patch_flux import TorchPatcher, config_gc, create_save_model_hook
        TorchPatcher.apply_patch()
        config_gc()

        if is_wandb_available(): # 原代码
          import wandb
        ```

        - 在log_validation里修改`pipeline = pipeline.to(accelerator.device)`，`train_dreambooth_flux.py`在171行附近

        ```python
        # 修改pipeline为：
        pipeline = pipeline.to(accelerator.device, dtype=torch_dtype)
        # pipeline = pipeline.to(accelerator.device) # 原代码
        ```

    3. 【Optional】Ubuntu系统需在1701行附近 添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps: # 原代码
          break
        accelerator.print("") # 添加
        ```

    4. 【Optional】模型checkpoint saving保存

        【因模型较大 如不需要`checkpointing_steps`，请设置其大于`max_train_steps`, 避免开启】

        ```shell
        --checkpointing_steps=50000 \ # 修改50000步为所需要步数
        ```

        【如需保存checkpointing请修改代码】

        ```shell
        vim train_dreambooth_flux.py #（1669行附近）
        vim ../advanced_diffusion_training/train_dreambooth_lora_flux_advanced.py #（2322行附近）
        ```

        - 在文件上方的import栏增加`DistributedType`在`from accelerate import Acceleratore`后 （30行附近）
        - 在`if accelerator.is_main_process`后增加 `or accelerator.distributed_type == DistributedType.DEEPSPEED` (1669/2322行附近)，并在`if args.checkpoints_total_limit is not None`后增加`and accelerator.is_main_process`

        ```python
        from accelerate import Accelerator, DistributedType
        # from accelerate import Accelerator # 原代码

        if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
        # if accelerator.is_main_process: # 原代码
          if global_step % args.checkpointing_steps == 0:  # 原代码 不进行修改
            if args.checkpoints_total_limit is not None and accelerator.is_main_process: # 添加
        ```

        Lora任务需调用patch任务进行权重保存：
        在`train_dreambooth_lora_flux_advanced.py`文件中找到代码`accelerator.register_save_state_pre_hook(save_model_hook)`进行修改(1712行附近)，复制粘贴以下代码：

        ```python
        # 添加
        save_Model_Hook = create_save_model_hook(
              accelerator=accelerator,
              unwrap_model=unwrap_model,
              transformer=transformer,
              text_encoder_one=text_encoder_one,
              args=args,
              weight_dtype=weight_dtype
        )
        accelerator.register_save_state_pre_hook(save_Model_Hook) # 修改
        # accelerator.register_save_state_pre_hook(save_model_hook) # 原代码
        accelerator.register_load_state_pre_hook(load_model_hook) # 原代码 不修改
        ```

        更改shell脚本：

        ```shell
        export HCCL_CONNECT_TIMEOUT=1200 # 大幅调高HCCL_CONNECT_TIMEOUT (如5000)
        export HCCL_EXEC_TIMEOUT=17000
        --checkpointing_steps=50000 \ # 修改50000步为所需要步数
        ```

    5. 【Optional】多机运行

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

4. 【启动 FLUX 微调脚本】

    本任务主要提供flux_dreambooth与flux_dreambooth_lora微调脚本，支持多卡训练。

    启动FLUX dreambooth微调脚本

    ```shell
    bash finetune_flux_dreambooth_deepspeed_bf16.sh 
    ```

    启动FLUX dreambooth_lora微调脚本

    ```shell
    bash finetune_flux_dreambooth_lora_deepspeed_bf16.sh
    ```

### 性能

#### 吞吐

FLUX 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Flux-全参微调  |  55.23  |     16      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Flux-全参微调  |  53.65 |     16      | bf16 | 2.1 | ✔ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【FLUX模型推理】

```shell
vim infer_flux_text2img_bf16.py # 进入运行推理的Python文件
```

  1. 修改路径

      ```python
      MODEL_PATH = "/black-forest-labs/FLUX.1-dev"  # FLUX模型路径
      ```

  2. 运行代码

      ```shell
      python infer_flux_text2img_bf16.py
      ```

  【DREAMBOOTH微调FLUX模型推理】

  ```shell
  vim infer_flux_text2img_dreambooth_bf16.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "./output_FLUX_dreambooth"  # Dreambooth微调保存模型路径
      ```

  2. 运行代码

      ```shell
      python infer_flux_text2img_dreambooth_bf16.py
      ```

  【lora微调FLUX模型推理】

  ```shell
  vim infer_flux_text2img_lora_bf16.py
  ```

  1. 修改路径

      ```python
      MODEL_PATH = "./FLUX"  # Flux 模型路径
      LORA_WEIGHTS = "./output/pytorch_lora_weights.safetensors"  # LoRA权重路径
      ```

  2. 运行代码

      ```shell
      python infer_flux_text2img_lora_bf16.py
      ```
  
  【分布式推理】

  ```shell
  vim infer_flux_text2img_distrib.py
  ```

- 修改模型权重路径 model_path为模型权重路径或微调后的权重路径
- 如lora微调 可将lora_weights修改为Lora权重路径

    ```python
    model_path = "/black-forest-labs/FLUX.1-dev"  # 模型权重/微调权重路径
    lora_weights = "/pytorch_lora_weights.safetensors"  # Lora权重路径
    ```

- 启动分布式推理脚本

  - 因使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

  ```shell
  accelerate launch --num_processes=4 infer_flux_text2img_distrib.py # 单机四卡进行分布式推理
  ```

<a id="jump3"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version |
|:---:|:---:|:----------:|:-----:|:---:|:---:|
| Atlas 900 A2 PODc |8p |  文生图  | 1.16 | bf16 | 2.1 |
| 竞品A | 8p |  文生图  | 1.82 | bf16 | 2.1 |
| Atlas 900 A2 PODc |8p |  文生图微调  | 1.12 | bf16 | 2.1 |
| 竞品A | 8p |  文生图微调  | 1.82 | bf16 | 2.1 |

## 环境变量声明

ASCEND_SLOG_PRINT_TO_STDOUT： 是否开启日志打印， 0：关闭日志打屏，1：开启日志打屏  
ASCEND_GLOBAL_LOG_LEVEL： 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志。0：对应DEBUG级别，1：对应INFO级别，2：对应WARNING级别，3：对应ERROR级别，4：对应NULL级别，不输出日志  
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
