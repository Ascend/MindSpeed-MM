# Diffusers

<p align="left">
        <b>简体中文</b> |
</p>

- [SD3](#jump1)
  - [模型介绍](#模型介绍)
  - [微调](#微调)
    - [环境搭建](#环境搭建)
    - [微调](#jump2)
    - [性能](#性能)
  - [推理](#推理)
    - [环境搭建及运行](#环境搭建及运行)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

<a id="jump1"></a>

# Stable Diffusion 3

## 模型介绍

扩散模型（Diffusion Models）是一种生成模型，可生成各种各样的高分辨率图像。Diffusers 是 HuggingFace 发布的模型套件，是最先进的预训练扩散模型的首选库，用于生成图像，音频，甚至分子的3D结构。套件包含基于扩散模型的多种模型，提供了各种下游任务的训练与推理的实现。

- 参考实现：

  ```shell
  url=https://github.com/huggingface/diffusers
  commit_id=94643fac8a27345f695500085d78cc8fa01f5fa9
  ```

## 微调

### 环境搭建

【模型开发时推荐使用配套的环境版本】

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
    <th>安装指南</th>
  </tr>
  <tr>
    <td> Python </td>
    <td> 3.8 </td>
  </tr>
  <tr>
    <td> Driver </td>
    <td> AscendHDK 24.1.RC3 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0003.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit">驱动固件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Firmware </td>
    <td> AscendHDK 24.1.RC3 </td>
  </tr>
  <tr>
    <td> CANN </td>
    <td> CANN 8.0.RC3 </td>
    <td>《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/softwareinst/instg/instg_0000.html">CANN 软件安装指南</a> 》</td>
  </tr>
  <tr>
    <td> Torch </td>
    <td> 2.1.0 </td>
    <td rowspan="2">《<a href="https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html">Ascend Extension for PyTorch 配置与安装</a> 》</td>
  </tr>
  <tr>
    <td> Torch_npu </td>
    <td> release v6.0.RC3 </td>
  </tr>
</table>

1. 软件与驱动安装

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

    ```bash
    # python3.8
    conda create -n test python=3.8
    conda activate test

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl 
    pip install torch_npu-2.1.0*-cp38-cp38m-linux_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl

    # 将shell脚本中的环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

2. 克隆仓库到本地服务器

    ```shell
    git clone --branch 1.0.RC3 https://gitee.com/ascend/MindSpeed-MM.git
    ```

3. 模型搭建

    3.1 【下载 SD3 [GitHub参考实现](https://github.com/huggingface/diffusers) 或 在模型根目录下执行以下命令，安装模型对应PyTorch版本需要的依赖】

    ```shell
    git clone https://github.com/huggingface/diffusers.git -b v0.30.0
    cd diffusers
    git checkout 94643fac8a27345f695500085d78cc8fa01f5fa9
    cp -r ../MindSpeed-MM/examples/diffusers/sd3 ./sd3
    ```

    【主要代码路径】

    ```shell
    code_path=examples/dreambooth/
    ```

    3.2【安装其余依赖库】

    ```shell
    pip install -e .
    vim examples/dreambooth/requirements_sd3.txt #修改版本：torchvision==0.16.0, torch==2.1.0, accelerate==0.33.0, 添加deepspeed==0.15.2
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

2. 【配置 SD3 微调脚本】

    联网情况下，微调模型可通过以下步骤下载。无网络时，用户可访问huggingface官网自行下载[sd3-medium模型](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) `model_name`模型

    ```bash
    export model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径
    ```

    获取对应的微调模型后，在以下shell启动脚本中将`model_name`参数设置为本地预训练模型绝对路径

    ```shell
    scripts_path="./sd3" # 模型根目录（模型文件夹名称）
    model_name="stabilityai/stable-diffusion-3-medium-diffusers" # 预训练模型路径
    dataset_name="pokemon-blip-captions" 
    batch_size=4
    max_train_steps=2000
    mixed_precision="bf16" # 混精
    resolution=1024
    config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"
    ```

    数据集选择：如果选择默认[原仓数据集](https://huggingface.co/datasets/diffusers/dog-example),需修改两处`dataset_name`为`input_dir`：

    ```shell
    input_dir="dog"

    # accelerator 修改 --dataset_name=#dataset_name
    --instance_data_dir=$input_dir
    ```

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
    if global_step >= args.max_train_steps:
      break
    accelerator.print("")
    ```

4. 【如需保存checkpointing请修改代码】

    ```shell
    vim examples/dreambooth/train_dreambooth_sd3.py
    # 或
    vim examples/dreambooth/train_dreambooth_lora_sd3.py
    ```

    - 在`if accelerator.is_main_process`后增加 `or accelerator.distributed_type == DistributedType.DEEPSPEED`（dreambooth在1681行附近,lora在1833行附近）
    - 在文件上方的import栏增加`DistributedType`在`from accelerate import Acceleratore`后 （30行附近）

    ```python
    from accelerate import Accelerator, DistributedType
    if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
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
    ```

6. 【启动 SD3 微调脚本】

    本任务主要提供**混精fp16**和**混精bf16**dreambooth和dreambooth+lora的**8卡**训练脚本，使用与不使用**deepspeed**分布式训练。

    ```shell
    bash sd3/finetune_sd3_dreambooth_deepspeed_**16.sh #使用deepspeed,dreambooth微调
    bash sd3/finetune_sd3_dreambooth_fp16.sh #无使用deepspeed,dreambooth微调
    bash sd3/finetune_sd3_dreambooth_lora_fp16.sh #无使用deepspeed,dreambooth+lora微调
    ```

### 性能

#### 吞吐

SD3 在 **昇腾芯片** 和 **参考芯片** 上的性能对比：

| 芯片 | 卡数 |     任务     |  FPS  | batch_size | AMP_Type | Torch_Version | deepspeed |
|:---:|:---:|:----------:|:-----:|:----------:|:---:|:---:|:---:|
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调  |   17.08 |     4      | bf16 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调  |  17.51 |     4      | bf16 | 2.1 | ✔ |
| Atlas 900 A2 PODc | 8p | Dreambooth-全参微调 |  16.57 |     4      | fp16 | 2.1 | ✔ |
| 竞品A | 8p | Dreambooth-全参微调 |   16.36 |     4      | fp16 | 2.1 | ✔ |
| Atlas 900 A2 PODc |8p | Dreambooth-全参微调 | 11.91  | 1 | fp16 | 2.1 | ✘ |
| 竞品A | 8p | Dreambooth-全参微调 | 12.08 | 1 | fp16 | 2.1 | ✘ |
| Atlas 900 A2 PODc |8p | DreamBooth-LoRA | 122.47 | 8 | fp16 | 2.1 | ✘ |
| 竞品A | 8p | DreamBooth-LoRA | 120.32 | 8 | fp16 | 2.1 | ✘ |

## 推理

### 环境搭建及运行

  **同微调对应章节**

 【运行推理的脚本】

  图生图推理脚本需先准备图片：[下载地址](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png)
  修改推理脚本中预训练模型路径以及图生图推理脚本中的本地图片加载路径
  调用推理脚本

  ```shell
  python sd3/infer_sd3_img2img_fp16.py   # 单卡推理，文生图
  python sd3/infer_sd3_text2img_fp16.py  # 单卡推理，图生图
  ```

## 使用基线数据集进行评估

## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md
