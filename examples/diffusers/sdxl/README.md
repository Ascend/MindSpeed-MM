# Diffusers

<p align="left">
</p>

- [SDXL](#jump1)
  - [模型介绍](#模型介绍)
  - [预训练](#预训练)
    - [环境搭建](#环境搭建)
    - [预训练](#jump2)
    - [性能](#性能)
  - [微调](#微调)
    - [环境搭建](#jump3)
    - [微调](#jump3.1)
    - [性能](#jump3.2)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
    - [性能](#jump4)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# Stable Diffusion XL

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```
  url=https://github.com/huggingface/diffusers
  commit_id=5956b68a6927126daffc2c5a6d1a9a189defe288
  ```

## 预训练

### 环境搭建

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

1. 软件与驱动安装

    ```bash
    # python3.10
    conda create -n test python=3.10
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl
    
    # apex for Ascend 参考 https://gitcode.com/Ascend/apex
    # 建议从原仓编译安装

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/cann/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 SDXL [GitHub参考实现](https://github.com/huggingface/diffusers)】或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    cd diffusers
    git checkout 5956b68a6927126daffc2c5a6d1a9a189defe288
    cp -r ../MindSpeed-MM/examples/diffusers/sdxl ./sdxl
    ```

    【主要代码路径】

    ```shell
    code_path=examples/text_to_image/
    ```

    3.2【安装 `{任务pretrain/train}_sdxl_deepspeed_{混精fp16/bf16}.sh`】

    转移 `collect_dataset.py` 与 `pretrain_model.py` 与 `train_text_to_image_sdxl_pretrain.py` 与 `patch_sdxl.py` 到 `examples/text_to_image/` 路径

    ```shell
    # Example: 需要修改.py名字进行四次任务
    cp ./sdxl/train_text_to_image_sdxl_pretrain.py ./examples/text_to_image/
    ```

    3.3【安装其余依赖库】

    ```shell
    pip install -e .
    vim examples/text_to_image/requirements_sdxl.txt #修改版本：torch==2.7.1
    pip install -r examples/text_to_image/requirements_sdxl.txt # 安装diffusers原仓对应依赖
    pip install -r sdxl/requirements_sdxl_extra.txt #安装sdxl对应依赖
    ```

<a id="jump2"></a>

### 预训练

1. 【准备预训练数据集】

    用户需自行获取并解压laion_sx数据集（目前数据集暂已下架，可选其他数据集）与[pokemon-blip-captions](https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions/tree/main)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径

    修改`pretrain_sdxl_deepspeed_**16.sh`的dataset_name为`laion_sx`的绝对路径

    ```shell
    vim sdxl/pretrain_sdxl_deepspeed_**16.sh
    ```

    修改`train_sdxl_deepspeed_**16.sh`的dataset_name为`pokemon-blip-captions`的绝对路径

    ```shell
    vim sdxl/train_sdxl_deepspeed_**16.sh
    ```

    laion_sx数据集格式如下:

    ```shell
    laion_sx数据集格式如下
    ├── 000000000.jpg
    ├── 000000000.json
    ├── 000000000.txt
    ```

    pokemon-blip-captions数据集格式如下:

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
  
2. 【配置 SDXL 预训练脚本与预训练模型】

    联网情况下，预训练模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) `model_name`模型与[sdxl-vae模型](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) `vae_name`

    ```bash
    export model_name="stabilityai/stable-diffusion-xl-base-1.0" # 预训练模型路径
    export vae_name="madebyollin/sdxl-vae-fp16-fix" # vae模型路径
    ```

    获取对应的预训练模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径

    ```bash
    scripts_path="./sdxl" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-xl-base-1.0" # 预训练模型路径
    vae_name="madebyollin/sdxl-vae-fp16-fix" # vae模型路径
    dataset_name="laion_sx" # 数据集路径
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/pretrain_${mixed_precision}_accelerate_config.yaml"

    # accelerate launch *** \ 目录下
    --dataloader_num_workers=8 \ # 请基于系统配置与数据大小进行调整
    ```

    修改bash文件中`accelerate`配置下`train_text_to_image_sdxl_pretrain.py`的路径（默认路径在diffusers/sdxl/）

    ```bash
    accelerate launch --config_file ${config_file} \
      ${scripts_path}/train_text_to_image_sdxl_pretrain.py \  #如模型根目录为sdxl则无需修改
    ```

    修改`pretrain_fp16_accelerate_config.yaml`的`deepspeed_config_file`的路径:

    ```bash
    deepspeed_config_file: ./sdxl/deepspeed_fp16.json # deepspeed JSON文件路径
    ```

    修改`examples/text_to_image/train_text_to_image_sdxl.py`文件

    ```bash
    vim examples/text_to_image/train_text_to_image_sdxl.py
    ```

    1. 在文件58行修改修改version

        ```python
        # 讲minimum version从0.31.0修改为0.30.0
        check_min_version("0.30.0")
        ```

    2. 在文件59行添加代码

        ```python
        from patch_sdxl import TorchPatcher, config_gc
        TorchPatcher.apply_patch()
        config_gc()
        ```

    <a id="jump2.1"></a>

    3. 【Optional】Ubuntu系统需在文件1216行附近添加 `accelerator.print("")`

        ```python
        if global_step >= args.max_train_steps:
          break
        accelerator.print("")
        ```

    4. 【FPS打印方式请参考train_text_to_image_sdxl_pretrain.py】

3. 【启动 SDXL 预训练脚本】

    本任务主要提供**混精fp16**和**混精bf16**两种**8卡**训练脚本，默认使用**deepspeed**分布式训练。

    **pretrain**模型主要来承担第二阶段的文生图的训练
    **train**模型主要来承担第一阶段的文生图的训练功能

    ```shell
    bash sdxl/pretrain_sdxl_deepspeed_**16.sh
    bash sdxl/train_sdxl_deepspeed_**16.sh
    ```

### 性能

#### 吞吐

SDXL 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| 竞品A | 8p | SDXL_train_bf16  |  30.65 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_bf16  | 29.92 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_train_fp16 |  30.23 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_train_fp16 | 28.51 |     4      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_pretrain_bf16  |  21.14 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_bf16  | 19.79 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | SDXL_pretrain_fp16 |  20.77 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | SDXL_pretrain_fp16 | 19.67 |     4      | fp16 | 2.1 | ✔ |

## 微调

<a id="jump3"></a>

### 环境搭建

#### LORA微调-数据集

   > **说明：**
   > 环境搭建同预训练。数据集同预训练的`pokemon-blip-captions`，请参考预训练章节。
   >

  ```shell
  sdxl/finetune_sdxl_lora_deepspeed_fp16.sh
  ```

#### Controlnet微调-数据集

   用户需自行获取[fill50k](https://huggingface.co/datasets/fusing/fill50k)数据集，并在以下启动shell脚本中将`dataset_name`参数设置为本地数据集的绝对路径，以及需要修改里面fill50k.py文件

   ```shell
   sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh
   ```

   deepspeed版本需改成0.14.4版本

   参考如下修改controlnet/train_controlnet_sdxl.py, 追加trust_remote_code=True

   ```shell
   dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
            trust_remote_code=True
          )
   ```

   > **注意：**
   >需要修改数据集下面的fill50k.py文件中的57到59行，修改示例如下:
>
   > ```python
   > metadata_path = "数据集路径/fill50k/train.jsonl"
   > images_dir = "数据集路径/fill50k"
   > conditioning_images_dir = "数据集路径/fill50k"
   >```
>
   fill50k数据集格式如下:

   ```
   fill50k
   ├── images
   ├── conditioning_images
   ├── train.jsonl
   └── fill50k.py
   ```

   > **说明：**
   >该数据集的训练过程脚本只作为一种参考示例。

#### 全参微调-数据集

   > **说明：**
   > 数据集同Lora微调，请参考Lora章节。
   >

#### 获取预训练模型

   获取[sdxl-base模型](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) `model_name`模型与[sdxl-vae模型](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix) `vae_name`。
  
   获取对应的预训练模型后，在`Controlnet微调`shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径，将`vae_name`参数设置为本地`vae`模型绝对路径。
  
   ```shell
   sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh
   ```

   `Lora微调`与`全参微调`shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径

   ```shell
   sdxl/finetune_sdxl_deepspeed_fp16.sh
   sdxl/finetune_sdxl_lora_deepspeed_fp16.sh
   ```

   > **说明：**
   > 预训练模型同预训练，请参考预训练章节。
   >

<a id="jump3.1"></a>

### 微调

   【Optional】如是Ubuntu系统需在 `examples/text_to_image/train_text_to_image_lora_sdxl.py` 与 `examples/controlnet/train_controlnet_sdxl.py` 添加 `accelerator.print("")`：[参考](#jump2.1)

   > **注意**
   > train_text_to_image_lora_sdxl 在1235行附近添加; train_controlnet_sdxl 在1307行附近添加
   >

  【Lora断点推理权重保存】

   如需保存checkpointing steps中的Lora_weights，须在代码上方（同sdxl预训练中的patch修改）添加如下：

   ```python
  from patch_sdxl import save_Lora_Weights
  ```
  
  并在train_text_to_image_lora_sdxl.py的1227行附近，`accelerator.save_state(save_path)`下方添加`save_Lora_Weights(unwrap_model, unet, text_encoder_one, text_encoder_two, args.train_text_encoder, save_path)`,如下：

  ```python
  accelerator.save_state(save_path)
  save_Lora_Weights(unwrap_model, unet, text_encoder_one, text_encoder_two, args.train_text_encoder, save_path)
  logger.info(f"Saved state to {save_path}")
  ```

   【运行微调的脚本】

    ```shell
    # 单机八卡微调
    # finetune_sdxl_controlnet_deepspeed_fp16.sh 中依赖的图片，可以通过下面命令下载
    # wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_1.png
    # wget https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_training/conditioning_image_2.png
    bash sdxl/finetune_sdxl_controlnet_deepspeed_fp16.sh      #8卡deepspeed训练 sdxl_controlnet fp16
    bash sdxl/finetune_sdxl_lora_deepspeed_fp16.sh            #8卡deepspeed训练 sdxl_lora fp16
    bash sdxl/finetune_sdxl_deepspeed_fp16.sh        #8卡deepspeed训练 sdxl_finetune fp16
    ```

<a id="jump3.2"></a>

### 性能

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| 竞品A | 8p |    LoRA    | 31.74 |     7      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |    LoRA    | 26.40 |     7      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | Controlnet | 32.44  |     5      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Controlnet | 29.98 |     5      | fp16 | 2.1 | ✔ |
| 竞品A | 8p |  Finetune  | 164.66 |     24     | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |  Finetune  | 166.71 |     24     | fp16 | 2.1 | ✔ |

**注**：Finetune任务中的FPS会在散热器性能较弱时波动幅度增大

## 推理

### 环境搭建及运行

  **同微调对应章节**

 【运行推理的脚本】

- 单机单卡推理,脚本配置
  - sdxl/sdxl_text2img_lora_infer.py
    - model_path配置为lora微调的输出目录 ，即用户在sdxl_text2img_lora_deepspeed.sh中指定的output_path
    - "stabilityai/stable-diffusion-xl-base-1.0"，无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
  - sdxl/sdxl_text2img_controlnet_infer.py
    - base_model_path配置为"stabilityai/stable-diffusion-xl-base-1.0"，无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - controlnet_path配置为controlnet微调输出的结果路径，即用户在sdxl_text2img_controlnet_deepspeed.sh中指定的output_path
  - sdxl/sdxl_text2img_infer.py
    - "/diffusion/sdxl/pretrained/"配置为"stabilityai/stable-diffusion-xl-base-1.0"，无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - "/diffusion/sdxl/pretrained/"也可以配置为微调输出的结果路径, 即微调脚本中指定的output_path
  - sdxl/sdxl_img2img_infer.py
    - MODEL_NAME配置为 "stabilityai/stable-diffusion-xl-base-1.0"，无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
    - VAE_NAME 配置为 "madebyollin/sdxl-vae-fp16-fix", 无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)
    - "Intel/dpt-hybrid-midas", 无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/Intel/dpt-hybrid-midas)
    - "diffusers/controlnet-depth-sdxl-1.0-small", 无网络时，用户可访问huggingface官网自行[下载](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0-small)

- 调用推理脚本

  ```shell
  python sdxl/sdxl_text2img_lora_infer.py        # 混精fp16 文生图lora微调任务推理
  python sdxl/sdxl_text2img_controlnet_infer.py  # 混精fp16 文生图controlnet微调任务推理
  python sdxl/sdxl_text2img_infer.py             # 混精fp16 文生图全参微调任务推理
  python sdxl/sdxl_img2img_infer.py              # 混精fp16 图生图微调任务推理
  ```

【分布式推理】

- 对`sdxl/sdxl_text2img_distrib_infer.py`文件进行修改

  ```shell
  vim sdxl/sdxl_text2img_distrib_infer.py
  ```

- 修改模型权重路径 model_path为模型权重路径或微调后的权重路径
- 如lora微调 可将lora_weights修改为Lora权重路径

  ```python
  model_path = "/stabilityai/stable-diffusion-xl-base-1.0"  # 模型权重/微调权重路径
  lora_weights = "/pytorch_lora_weights.safetensors"  # Lora权重路径
  ```

- 启动分布式推理脚本

  - 因使用accelerate进行分布式推理，config可设置：`--num_processes=卡数`，`num_machines=机器数`等

  ```shell
  accelerate launch --num_processes=4 sdxl/sdxl_text2img_distrib_infer.py # 单机四卡进行分布式推理
  ```

<a id="jump4"></a>

### 性能

| 芯片 | 卡数 |     任务     |  E2E（it/s）  |  AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:---:|:---:|:---:|
| 竞品A | 1p |    文生图lora    | 1.45 |  fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p |    文生图lora    | 2.61 |  fp16 | 2.1 | ✔ |
| 竞品A | 1p | 文生图controlnet | 1.41  |  fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |1p | 文生图controlnet | 2.97 |  fp16 | 2.1 | ✔ |
| 竞品A | 1p |  文生图全参  | 1.55 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |1p |  文生图全参  | 3.02 | fp16 | 2.1 | ✔ |
| 竞品A | 1p |  图生图  | 3.56 | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |1p |  图生图  | 3.94 | fp16 | 2.1 | ✔ |

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
