# Qihoo-T2X 1.0 Usage Guide

<p align="left"></p>

This is the official open-source code repository of [Qihoo-T2X](https://360cvgroup.github.io/Qihoo-T2X/).

[*QIHOO-T2X: AN EFFICIENT PROXY-TOKENIZED DIFFUSION TRANSFORMER FOR TEXT-TO-ANY-TASK*](https://arxiv.org/pdf/2409.04005)  Jing Wang*, Ao Ma*†, Jiasong Feng*, Dawei Leng‡, Yuhui Yin, Xiaodan Liang‡(*Equal Contribution, †Project Lead, ‡Corresponding Authors)

## Contents

<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <img src="2.png" width="800">
          <p>Prompt: Close-up of a man's face wearing glasses against a colorful background.</p>
      </td>
      <td>
          <img src="1.png" width="800">
          <p>Prompt: A dog wearing virtual reality goggles in sunset, 4k, high resolution.</p>
      </td>
  </tr>
</table>

- [Qihoo-T2X 1.0 Usage Guide](#qihoo-t2x-10-usage-guide)
  - [Contents](#contents)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
    - [3. Model Weights Download](#3-model-weights-download)
  - [Inference](#inference)
    - [1. Parameter Configuration](#1-parameter-configuration)
    - [2. Start Inference](#2-start-inference)
  - [Environment Variable Declaration](#environment-variable-declaration)
  - [Reference](#reference)
  - [License](#license)

<a id="jump1"></a>

## Environment Setup

<a id="jump1.1"></a>

### 1. Repository Cloning

```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.8.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    cd MindSpeed-MM
    mkdir pretrain_models
```

<a id="jump1.2"></a>

### 2. Environment Setup

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md).

```bash
    # python3.10
    conda create -n qihoot2x python=3.10
    conda activate qihoot2x

    # Install torch and torch_npu, making sure to select the torch, torch_npu, and apex packages that correspond to your Python version and x86 or arm architecture.
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

    # For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
    # It is recommended to compile and install from the original repository.

    # Install acceleration libraries.
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed
    git checkout 3f09d6736571cf1e30f8ac97de77982d0ab32cc5
    pip install -r requirements.txt
    pip3 install -e .
    cd ..

    # Install other required dependency libraries.
    pip install -e .
```

<a id="jump1.3"></a>

### 3. Model Weights Download

Download the corresponding model weights from the Hugging Face library:

- Model address: [Qihoo-t2i-1B](https://huggingface.co/qihoo360/Qihoo-T2X/tree/main)

Save the downloaded model weights to the local `pretrain_models/qihoo_t2i/XXX.pt` directory. (*XXX* represents the corresponding name.)

- VAE model address: [Open-Sora-Plan 1.2 VAE](https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae)

Save the downloaded model weights to the local `pretrain_models/opensoraplan_vae1_2/` directory.

- Text encoder model address: [T5-XXL (fp16)](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512/tree/main/text_encoder)

Save the downloaded model weights to the local `pretrain_models/text_encoder` directory.

- [tokenizer address](https://huggingface.co/alibaba-pai/EasyAnimateV2-XL-2-512x512/tree/main/tokenizer)

Save the downloaded model weights to the local `pretrain_models/tokenizer` directory.

## Inference

<a id="jump2.1"></a>

### 1. Parameter Configuration

- Pass the prepared weights into `examples/qihoo_t2x/inference_model_image.json` and modify the paths within, including `from_pretrained`.

- Custom prompts can be modified and added in `examples/qihoo_t2x/demo.txt`.

<a id="jump2.2"></a>

### 2. Start Inference

i2v inference startup script:

```shell
sh examples/qihoo_t2x/inference_qihoo.sh
```

<a id="jump3"></a>

## Environment Variable Declaration

`ASCEND_RT_VISIBLE_DEVICES` Specifies the index value of the NPU device.
`NPUS_PER_NODE` Configures the number of NPUs used on a single compute node.

## Reference

```shell
@article{wang2024qihoo,
  title={Qihoo-T2X: An Efficient Proxy-Tokenized Diffusion Transformer for Text-to-Any-Task},
  author={Wang, Jing and Ma, Ao and Feng, Jiasong and Leng, Dawei and Yin, Yuhui and Liang, Xiaodan},
  journal={arXiv preprint arXiv:2409.04005},
  year={2024}
}
```

## License

This project is licensed under the [Apache License (Version 2.0)](https://github.com/modelscope/modelscope/blob/master/LICENSE).
