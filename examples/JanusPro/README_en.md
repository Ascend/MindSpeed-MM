# JanusPro User Guide

<p align="left">
</p>

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/install_guide.md.

### 1. Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/deepseek-ai/Janus.git
cd MindSpeed-MM
mkdir ckpt
cd ..

cp -r ./Janus ./MindSpeed-MM/examples/JanusPro
```

### 2. Environment Setup

Modify the dependencies in the `pyproject.toml` file located in `./examples/JanusPro/Janus` as follows:

- torch==2.7.1
- numpy==1.26.4
- Add dependency: decorator
- Add dependency: scipy
- Add dependency: attrs

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# Install torch and torch_npu. Make sure to select the torch, torch_npu, and apex packages that match your Python version and architecture (x86 or arm).
# Download link: https://www.hiascend.com/document/detail/en/Pytorch/60RC3/configandinstg/instg/insg_0001.html
pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
pip install torch_npu-2.7.1*-cp310-cp310-manylinux_2_28_aarch64.whl

# Install JanusPro Dependencies
cd MindSpeed-MM/examples/JanusPro/Janus
pip install -e .

cd ..
```

## Weight Download

### 1. Weight Download

Download the corresponding model weights from the Hugging Face library.

- Model address: [JanusPro-7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/main)

Save the downloaded model weights to the local `ckpt/JanusPro-7B/` directory.
<a id="jump2.2"></a>

## Image Understanding

### 1. Preparation

Before configuring the script, you need to complete the prerequisite preparations, including: **Environment Setup** and **Weight Download**. For details, refer to the corresponding sections.

### 2. Parameter Configuration

In `multimodal_understanding.sh`, modify the parameters to be passed: `model_path`, `image_path`, and `question`, representing the location where the model weights are saved, the image location, and the question about the input image, respectively.
You also need to modify `trust_remote_code` to `True`. To ensure code security, `trust_remote_code` is configured to `False` by default. You need to set it to `True` and ensure the security of the models and data you download.

Additionally, you need to set the installation location of the CANN package.

```shell
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh
```

### 3. Launch Script

```bash
  cd ./examples/JanusPro/
  bash multimodal_understanding.sh
```

You can then obtain the inference result.

Inference result example:
Using `examples/qwen2vl/demo.jpg` as the input image and the question "Describe this image", the inference result is:

```shell
  The image shows a person sitting on a sandy beach, interacting with a dog.
  The person is wearing a checkered skirt and dark pants, and the dog,
  which appears to be a Labrador Retriever, is sitting on its hind legs,
  reaching out with its front paws towards the person. The sun is setting,
  casting a warm glow over the screen.
```
