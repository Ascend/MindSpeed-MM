# Diffusers

- [Diffusers](#diffusers)
- [kolors](#kolors)
  - [Model Introduction](#model-introduction)
  - [Inference](#inference)
    - [Weight Acquisition](#weight-acquisition)
    - [Environment Setup](#environment-setup)
  - [Reference](#reference)
    - [Public IP Address Description](#public-ip-address-description)

# kolors

## Model Introduction

Kolors is a large-scale text-to-image generation model based on latent diffusion, developed by the Kuaishou Kolors team. Trained on billions of image-text pairs, Kolors demonstrates significant advantages over both open-source and closed-source models in terms of vision quality, complex semantic understanding, and text generation (both Chinese and English characters). Additionally, Kolors supports both Chinese and English, making it more competitive in understanding Chinese-specific content.

- Reference Implementation:

  ```shell
  url=https://github.com/Kwai-Kolors/Kolors
  commit_id=0fafa56a76b7acf1e147b153d1e7b8fd65f9055b
  ```

## Inference

### Weight Acquisition

1. With the Internet connected, the pre-trained model can be downloaded automatically.

2. Without network access, you can download the model from the official Hugging Face repository, with the namespace `Kwai-Kolors/Kolors-diffusers`. Note that if weights are downloaded locally, the weight path needs to be passed into [infer_kolors_fp16.py](infer_kolors_fp16.py).

### Environment Setup

  Please refer to [Pytorch Framework Training Environment Preparation](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes) released in the Ascend Community to set up the Ascend environment.

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/install_guide.md).

1. Install the third-party components.

    ```shell
    # python3.10
    conda create -n kolors python=3.10
    conda activate kolors

    # Install torch
    pip install torch-2.7.1-cp310-cp310-manylinux_2_28_aarch64.whl
    # For apex for Ascend, refer to https://gitcode.com/Ascend/apex.
    # It is recommended to compile and install components from the source repository.
    pip install diffusers==0.30.0 accelerate==0.27.2 transformers==4.42.4  torchvision==0.22.1
    ```

2. Clone the repository to the local server.

    ```shell
    # Clone the repository.
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd examples/diffusers/kolors
    ```

3. Run the inference script.

    ```shell
    # Modify the environment variable paths below to the actual paths. Example:
    source /usr/local/Ascend/cann/set_env.sh
    python infer_kolors_fp16.py
    ```

<a id="jump1"></a>

## Reference

### Public IP Address Description

For public IP addresses used in the code, refer to [Public IP Address](../../../docs/en/public_address_statement.md)
