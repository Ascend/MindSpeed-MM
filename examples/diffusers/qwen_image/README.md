# Diffusers

<p align="left">
</p>

- [Qwen Image](#jump1)
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

# Qwen Image

## 模型介绍

Qwen Image是基于 MMDiT 扩散骨干与 Qwen2.5-VL 文本编码器构建的多模态图像生成模型，其核心结构创新在于采用 MSRoPE 多模态位置编码解决文本与图像位置混淆问题；功能上以卓越的多语言文本渲染（尤其中文）和精准图像编辑为特色，同时具备抢的通用图像生成能力。

## 版本说明

### 参考实现

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=7a2b78bf0f788d311cc96b61e660a8e13e3b1e63
  ```

### 变更记录

2025.09.08：首次发布Qwen Image

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

1. 软件与驱动安装

    ```bash
    # 创建并激活python环境，安装 torch 和 torch_npu，请参考上述安装指南

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 Qwen Image [GitHub参考实现](https://github.com/huggingface/diffusers) 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git
    cd diffusers
    git checkout 7a2b78b
    cp -r ../MindSpeed-MM/examples/diffusers/qwen_image/* ./examples/dreambooth
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    pip install -r examples/dreambooth/requirements_qwen_image.txt # 安装对应依赖
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

    在shell脚本`accelerate launch`目录下（40行左右）将修改 `dataset_name=$dataset_name`，并将`instance_prompt`改为与自己数据集所匹配的prompt，`caption_column`修改为数据集匹配名称:

    ```shell
    # Example
    accelerate launch --config_file $config_file \
      --num_process $num_processors \
      ./train_dreambooth_lora_qwen_image.py \
      --pretrained_model_name_or_path=$model_name  \
      --dataset_name=$dataset_name \
      --caption_column="text" \
      --instance_prompt="a photo of pokemon" \
    ```

2. 【配置 Lora 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[Qwen Image模型](https://huggingface.co/Qwen/Qwen-Image) `model_name`模型

    ```shell
    model_name="Qwen/Qwen-Image" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动微调脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    model_name="Qwen/Qwen-Image"
    dataset_name="pokemon-blip-captions"
    batch_size=8
    num_processors=8
    max_train_steps=5000
    mixed_precision="bf16"
    resolution=512
    gradient_accumulation_steps=1
    config_file="bf16_accelerate_config.yaml"

    # accelerate launch --config_file $config_file \ 目录下
    --dataloader_num_workers=0 \ # 请基于系统配置与数据大小进行调整num workers
    ```

3. 【修改代码文件】

    1. 【Optional】如机器未联网，需对save_model_card进行修改：
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

    2. 【Optional】多机运行

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

4. 【启动 Qwen Image 微调脚本】

    本任务主要提供dreambooth_lora_qwen_image微调脚本，支持多卡训练。

    启动Qwen Image dreambooth_lora微调脚本

    ```shell
    bash finetune_qwen_image_dreambooth_lora_deepspeed_bf16.sh
    ```

## 推理

### 环境搭建及运行

  **同微调对应章节**

```shell
cd examples/dreambooth/ # 从diffusers目录进入dreambooth目录
```

【DREAMBOOTH微调Qwen Image模型推理】

```shell
vim infer_qwen_image_text2img_bf16.py # 进入运行推理的Python文件
```

  1. 修改路径

      ```python
      model_name="Qwen/Qwen-Image" # Qwen Image pretrained model
      output_path = "./infer_result"  # Inference result output folder
      ```

  2. 运行代码

      ```shell
      python infer_qwen_image_text2img_bf16.py
      ```

【lora微调FLUX模型推理】

  ```shell
  vim infer_qwen_image_text2img_lora_bf16.py
  ```

  1. 修改路径

      ```python
      model_name="Qwen/Qwen-Image" # Qwen Image pretrained model
      lora_path = "qwen_image_lora" # Folder containing trained lora weights
      output_path = "./infer_result"  # Inference result output folder
      ```

  2. 运行代码

      ```shell
      python infer_qwen_image_text2img_lora_bf16.py
      ```
  
<a id="jump3"></a>

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

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/public_address_statement.md)
