# Diffusers

- [kolors](#kolors)
  - [模型介绍](#模型介绍)
  - [推理](#推理)
    - [环境搭建](#环境搭建)
    - [推理](#推理)
- [引用](#jump1)
  - [公网地址说明](#公网地址说明)

# kolors

## 模型介绍

可图大模型是由快手可图团队开发的基于潜在扩散的大规模文本到图像生成模型。Kolors 在数十亿图文对下进行训练，在视觉质量、复杂语义理解、文字生成（中英文字符）等方面，相比于开源/闭源模型，都展示出了巨大的优势。同时，Kolors 支持中英双语，在中文特色内容理解方面更具竞争力。

- 参考实现：

  ```
  url=https://github.com/Kwai-Kolors/Kolors
  commit_id=0fafa56a76b7acf1e147b153d1e7b8fd65f9055b
  ```

## 推理

### 权重获取

1.联网情况下，预训练模型会自动下载。

2.无网络情况下，用户可以访问huggingface官方下载，namespace为 Kwai-Kolors/Kolors-diffusers。注意如果本地下载权重需要将权重路径传入到[infer_kolors_fp16.py](infer_kolors_fp16.py)中。

### 环境搭建

  请参考昇腾社区中《[Pytorch框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》文档搭建昇腾环境，本仓已支持表2中软件版本。

  **表 2**  昇腾软件版本支持表

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

1. 三方件安装

    ```shell
    # python3.10
    conda create -n kolors python=3.10
    conda activate kolors

    # 安装 torch 
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    # apex for Ascend 参考 https://gitcode.com/Ascend/apex
    # 建议从原仓编译安装
    pip install diffusers==0.30.0 accelerate==0.27.2 transformers==4.42.4  torchvision==0.22.1
    ```

2. 克隆仓库到本地服务器

    ```shell
    # 克隆仓库
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd examples/diffusers/kolors
    ```

3. 运行推理的脚本

    ```shell

# 将下面环境变量路径修改为真实路径，下面为参考路径

   source /usr/local/Ascend/cann/set_env.sh
   python infer_kolors_fp16.py

   ```

<a id="jump1"></a>

## 引用

### 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/public_address_statement.md)
