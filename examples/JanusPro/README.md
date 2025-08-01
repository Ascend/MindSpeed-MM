# JanusPro 使用指南

<p align="left">
</p>


## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

#### 1. 仓库拉取

```shell
git clone --branch 2.1.0 https://gitee.com/ascend/MindSpeed-MM.git
git clone https://github.com/deepseek-ai/Janus.git
cd MindSpeed-MM
mkdir ckpt
cd ..

cp -r ./Janus ./MindSpeed-MM/examples/JanusPro
```

#### 2. 环境搭建

对./examples/JanusPro/Janus中的pyproject.toml文件的dependencies做如下修改：
- torch==2.6.0
- numpy==1.26.4
- 增加依赖：decorator
- 增加依赖：scipy
- 增加依赖：attrs
  


```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
# 下载路径参考 https://www.hiascend.com/document/detail/zh/Pytorch/60RC3/configandinstg/instg/insg_0001.html
pip install torch-2.6.0-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.6.0*-cp310-cp310-manylinux_2_28_aarch64.whl

# 安装JanusPro依赖库
cd MindSpeed-MM/examples/JanusPro/Janus
pip install -e .

cd ..
```

## 权重下载

#### 1. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [JanusPro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/main)；

 将下载的模型权重保存到本地的`ckpt/JanusPro-7B/`目录下。
<a id="jump2.2"></a>

## 图像理解

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**，详情可查看对应章节。

#### 2. 配置参数

在multimodal_understanding.sh中修改需要传入的参数，model_path, image_path与question，分别表示模型权重保存位置，图片位置与针对传入图片的问题。
同时还需要修改"trust_remote_code"为true。为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性。

另外还需设置CANN包安装位置。

```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

#### 3. 启动脚本
```bash 
  cd /examples/JanusPro/
  bash multimodal_understanding.sh
```
即可得到推理结果。

推理结果示例：
以examples/qwen2vl/demo.jpg为输入图片，question为“Describle this image”，得到的推理结果为：

```shell
  The image shows a person sitting on a sandy beach, interacting with a dog. 
  The person is wearing a checkered skirt and dark pants, and the dog, 
  which appears to be a Labrador Retriever, is sitting on its hind legs, 
  reaching out with its front paws towards the person. The sun is setting, 
  casting a warm glow over the screen. 
```





