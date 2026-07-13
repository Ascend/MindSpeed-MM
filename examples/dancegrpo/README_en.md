# FLUX DanceGRPO Usage Guide

<p align="left">
</p>

## Contents

- [FLUX DanceGRPO Usage Guide](#flux-dancegrpo-usage-guide)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Reference Implementation](#reference-implementation)
  - [Environment Setup](#environment-setup)
    - [1. Repository Cloning](#1-repository-cloning)
    - [2. Environment Setup](#2-environment-setup)
    - [3. Decord Setup](#3-decord-setup)
  - [Weight Download](#weight-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Training](#training)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Third-party Library Modifications](#2-third-party-library-modifications)
    - [3. Launch Training](#3-launch-training)
  - [Performance Metrics](#performance-metrics)
  - [FAQ](#faq)

<a id="jump0"></a>

## Introduction

This guide helps you quickly get started with reproducing the [DanceGRPO](https://arxiv.org/abs/2505.07818) post-training method using the MindSpeed MM repository. You will first need to complete the prerequisites, including the code repository, environment, dataset, and weights, and then launch training according to the instructions provided. The specific steps are detailed below.

### Reference Implementation

The DanceGRPO open-source code repository and its corresponding commit ID are as follows:

```shell
url=https://github.com/XueZeyue/DanceGRPO
commit_id=2149f36f22db601f9dbf70472fea11576f62a0f6
```

<a id="jump1"></a>

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../docs/en/pytorch/installation.md)

> For the DanceGRPO scenario, Python version 3.10 is recommended.

<a id="jump1.1"></a>

### 1. Repository Cloning

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..

cd MindSpeed-MM
mkdir -p logs data ckpt
cd ..
```

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# For x86 devices, if you encounter import issues related to torchvision, it is recommended to first check whether the torchvision version in the environment is +cpu. It is suggested to use the following source configuration to resolve such issues.
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# Install torch and torch_npu.
pip install torch-2.7.1+cpu-cp310-cp310-*.whl
pip install torch_npu-2.7.1*.whl

# Install acceleration libraries.
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
cp -r mindspeed ../MindSpeed-MM/
cd ..

# Install DanceGRPO dependency libraries.
cd MindSpeed-MM
pip install -r ./examples/dancegrpo/requirements-lint.txt
cd ..

git clone https://github.com/tgxs002/HPSv2.git
cd HPSv2
git checkout 866735ecaae999fa714bd9edfa05aa2672669ee3
pip install -e .
cd ..
```

### 3. Decord Setup

[X86 Installation]

```bash
pip install decord==0.6.0
```

[ARM Installation]

For installation via `apt`, please [click this link](https://github.com/dmlc/decord).

For installation via `yum`, please [refer to this script](https://github.com/dmlc/decord/blob/master/tools/build_manylinux2010.sh).

<a id="jump2"></a>

## Weight Download

Create a directory to save the weights:

```bash
cd MindSpeed-MM
mkdir ckpt/flux
mkdir ckpt/hps_ckpt
cd ..
```

Download the [FLUX Pre-trained Weights](https://huggingface.co/black-forest-labs/FLUX.1-dev) to the `ckpt/flux` directory under the root directory of the MindSpeed MM project.

Download the [HPS-v2.1 Pre-trained Weights](https://huggingface.co/xswu/HPSv2/tree/main).Save the `HPS_v2.1_compressed.pt` file to the `ckpt/hps_ckpt` directory under the root directory of the MindSpeed MM project.

Download the [CLIP Pre-trained Weights](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/tree/main). Save the `open_clip_pytorch_model.bin` file to the `ckpt/hps_ckpt` directory under the root directory of the MindSpeed MM project.

<a id="jump3"></a>

## Dataset Preparation and Processing

Download the [prompt dataset](https://github.com/XueZeyue/DanceGRPO/blob/main/assets/prompts.txt) used by FLUX DanceGRPO. On the file page, click **download raw file** to save the file to the `data` directory under the MindSpeed MM project root directory.

The downloaded dataset needs to be preprocessed. Before starting the preprocessing, modify the configuration of the [data preprocessing script](./preprocess_flux_rl_embeddings.sh) according to your training configuration needs. Taking the FLUX model as an example:

1. The path for the VAE model weights is `LOAD_PATH`, which defaults to `ckpt/flux`.
2. The storage path for the preprocessed dataset is `OUTPUT_DIR`, which defaults to `data/rl_embeddings`.
3. The prompt file path is `PROMPT_DIR`, which defaults to `data/prompts.txt`.

After modifying the above points, you can launch the script for data preprocessing:

```bash
cd MindSpeed-MM
bash examples/dancegrpo/preprocess_flux_rl_embeddings.sh
```

The processed data will be stored by default in the `data/rl_embeddings` directory under the MindSpeed MM root directory.

<a id="jump4"></a>

## Training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the scripts, you need to complete the preliminary preparations by referring to the corresponding sections, including: environment installation, weight download, and dataset preparation and processing.

<a id="jump4.2"></a>

### 2. Third-party Library Modifications

Locate the root directory of the Python environment you are using. For environments installed via conda, use the following command to find it:

```bash
echo $(conda info --envs | grep test) | awk '{print $NF}'
```

1. In the file `lib/python3.10/site-packages/diffusers/models/embeddings.py`, locate the following code within the `forward` function of the `FluxPosEmbed` class:

    ```python
    is_mps = ids.device.type == "mps"
    freqs_dtype = torch.float32 if is_mps else torch.float64
    ```

    Modify it to:

    ```python
    is_mps = ids.device.type == "mps"
    is_npu = ids.device.type == "npu"
    freqs_dtype = torch.float32 if is_mps or is_npu else torch.float64
    ```

2. In the file `lib/python3.10/site-packages/diffusers/models/embeddings.py`, locate the following code within the `get_1d_rotary_pos_embed` function:

    ```python
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
    ```

    Modify it to:

    ```python
    freqs_cos = freqs.cos().T.repeat_interleave(2, dim=0).T.contiguous().float()
    freqs_sin = freqs.sin().T.repeat_interleave(2, dim=0).T.contiguous().float()
    ```

3. In the file `lib/python3.10/site-packages/diffusers/models/attention_processor.py`, locate the following code within the `__init__` function of the `Attention` class:

    ```python
    elif qk_norm == "rms_norm":
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
    ```

    Modify it to:

    ```python
    elif qk_norm == "rms_norm":
        self.norm_q = NpuFusedRMSNorm(dim_head, eps=eps)
        self.norm_k = NpuFusedRMSNorm(dim_head, eps=eps)
    ```

    Add the following class:

    ```python
    class NpuFusedRMSNorm(torch.nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x):
            return torch_npu.npu_rms_norm(x.to(self.weight.dtype), self.weight, epsilon=self.eps)[0]
    ```

### 3. Launch Training

The following using the FLUX model as an example. Before launching training, you can modify the configuration in the [launch script](./posttrain_flux_dancegrpo.sh) according to your own training setup needs:

1. Modify the `NNODES` and `NPUS_PER_NODE` configurations as required. For example, for a single node with 8 NPUs, set `NNODES` to `1` and `NPUS_PER_NODE` to `8`;
2. For multi-node training, ensure that the `MASTER_ADDR` is consistent across all nodes and is the IP address of one of the nodes. `MASTER_PORT` of each node should be configured with the same port number. Starting from the node whose IP is `MASTER_ADDR`, configure `NODE_RANK` of each node as an integer incrementing from 0.
3. The dataset configuration path is `MM_DATA`, which defaults to `./examples/dancegrpo/data_dancegrpo.json`.
4. The model configuration path is `MM_MODEL`, which defaults to `./examples/dancegrpo/model_dancegrpo.json`.
5. The loading path for the DiT model's pre-trained weights is `LOAD_PATH`, which defaults to `ckpt/flux`. Adjust it based on where your weights are stored.
6. The save path for the training weights is `SAVE_PATH`, which defaults to `save_dir`.
7. The path for saving the `reward` values during model training is `HPS_REWARD_SAVE_PATH`, which defaults to `./hps_reward.txt`.

Before starting training, you can modify the dataset configuration file [data_dancegrpo.json](./data_dancegrpo.json) according to your training configuration needs:

1. `dataset_param.basic_parameters.data_path` indicates the path to the metadata file `videos2caption.json` in the preprocessed data.

Before starting training, you can modify the model configuration file [model_dancegrpo.json](./model_dancegrpo.json) according to your own training configuration needs:

1. `reward.ckpt_dir` indicates the path to the pre-trained weights of the reward model.

After modifying the above points, you can launch the script to start training:

```bash
bash examples/dancegrpo/posttrain_flux_dancegrpo.sh
```

> *Note: The directory hierarchy for code, weights, data, and other paths must be consistent across all nodes, and the training script must be launched from the MindSpeed MM directory.*

After training is complete, run log files will be generated in the `logs` directory, along with a training reward record file.

## Performance Metrics

| Model               | Machine      | Cluster | Task | GBS | End-to-End SPS |
|----------------------|---------------------|------|-----|-----|------------|
| FLUX DanceGRPO       | Atlas 200T A2 Box16 | 1*8 | Fine-tuning | 32 | 0.1123     |

Note: SPS here stands for Samples per Second.

<a id="jump5"></a>

## FAQ

1. For devices with an x86 CPU, it is recommended to use torchvision `0.22.1+cpu`. If you encounter import issues related to torchvision, it is advisable to first check whether the torchvision version in your environment is the `+cpu` version.
