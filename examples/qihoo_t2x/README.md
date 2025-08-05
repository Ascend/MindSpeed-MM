# Qihoo-T2X 1.0 使用指南

<p align="left"></p>

这里是 [Qihoo-T2X](https://360cvgroup.github.io/Qihoo-T2X/) 官方开源代码仓库

**[QIHOO-T2X: AN EFFICIENT PROXY-TOKENIZED DIFFUSION TRANSFORMER FOR TEXT-TO-ANY-TASK](https://arxiv.org/pdf/2409.04005)**  Jing Wang*, Ao Ma*†, Jiasong Feng*, Dawei Leng‡, Yuhui Yin, Xiaodan Liang‡(*Equal Contribution, †Project Lead, ‡Corresponding Authors)

## 目录

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="2.png" width="800"></img>
          <p>Prompt: Close-up of a man's face wearing glasses against a colorful background.</p>
      </td>
      <td>
          <img src="1.png" width="800"></img>
          <p>Prompt: A dog wearing virtual reality goggles in sunset, 4k, high resolution.</p>
      </td>
  </tr>
</table>

- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
  - [权重下载](#jump1.3)
- [推理](#jump2)
  - [配置参数](#jump2.1)
  - [启动推理](#jump2.2)
- [环境变量声明](#jump3)

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
    git clone https://gitee.com/ascend/MindSpeed-MM.git 
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.8.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    mkdir pretrain_models
```

<a id="jump1.2"></a>

#### 2. 环境搭建

请参考[安装指南](https://gitee.com/ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)

```bash
    # python3.10
    conda create -n qihoot2x python=3.10
    conda activate qihoot2x

    # 安装 torch 和 torch_npu，注意要选择对应python版本、x86或arm的torch、torch_npu及apex包
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl
    
    # apex for Ascend 参考 https://gitee.com/ascend/apex
    # 建议从原仓编译安装

    # 安装加速库
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 3f09d6736571cf1e30f8ac97de77982d0ab32cc5
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..

    # 安装其余依赖库
    pip install -e .
```

<a id="jump1.3"></a>

#### 3. 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qihoo-t2i-1B](https://huggingface.co/qihoo360/Qihoo-T2X/tree/main)；

 将下载的模型权重保存到本地的`pretrain_models/qihoo_t2i/XXX.pt`目录下。(XXX表示对应的名字)

- VAE模型地址: [Open-Sora-Plan 1.2 VAE](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae)；

 将下载的模型权重保存到本地的`pretrain_models/opensoraplan_vae1_2/`目录下。

- 文本编码器模型地址: [T5-XXL (fp16)](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512/tree/main/text_encoder)；
 将下载的模型权重保存到本地的`pretrain_models/text_encoder`目录下。

- [tokenizer地址](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512/tree/main/tokenizer)；
 将下载的模型权重保存到本地的`pretrain_models/tokenizer`目录下。

## 推理

<a id="jump2.1"></a>

#### 1. 配置参数

- 将准备好的权重传入到`examples/qihoo_t2x/inference_model_image.json`中，更改其中的路径，包括from_pretrained。

- 自定义的prompt可以在`examples/qihoo_t2x/demo.txt`中修改和添加。

<a id="jump2.2"></a>

#### 2. 启动推理

i2v 启动推理脚本

```shell
sh examples/qihoo_t2x/inference_qihoo.sh
```
<a id="jump3"></a>
## 环境变量声明
ASCEND_RT_VISIBLE_DEVICES： 指定NPU设备的索引值  
NPUS_PER_NODE： 配置一个计算节点上使用的NPU数量

## 文献引用

```
@article{wang2024qihoo,
  title={Qihoo-T2X: An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Any-Task},
  author={Wang, Jing and Ma, Ao and Feng, Jiasong and Leng, Dawei and Yin, Yuhui and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2409.04005},
  year={2024}
}
```


## 许可证

本项目许可遵从以下协议 [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
