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

1. 三方件安装

torch npu 与 CANN包参考链接：[安装包参考链接](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373/software)

    ```shell
    # python3.8
    conda create -n kolors python=3.8
    conda activate kolors

    # 安装 torch 
    pip install torch-2.1.0-cp38-cp38m-manylinux2014_aarch64.whl
    pip install apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
    pip install diffusers==0.30.0 accelerate==0.27.2 transformers==4.42.4  torchvision==0.16.0
    ```

2. 克隆仓库到本地服务器

    ```shell
    # 克隆仓库
    git clone --branch 1.0.RC3 https://gitee.com/ascend/MindSpeed-MM.git
    cd examples/diffusers/kolors
    ```

3. 运行推理的脚本

    ```shell
    # 将下面环境变量路径修改为真实路径，下面为参考路径
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python infer_kolors_fp16.py

   ```

<a id="jump1"></a>

## 引用

### 公网地址说明

代码涉及公网地址参考 public_address_statement.md
