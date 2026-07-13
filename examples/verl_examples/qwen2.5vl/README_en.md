# Qwen2_5_VL GRPO User Guide

<p align="left">
</p>

## Contents

- [Qwen2\_5\_VL GRPO User Guide](#qwen2_5_vl-grpo-user-guide)
  - [Contents](#contents)
  - [Introduction](#introduction)
    - [Reference Implementation](#reference-implementation)
    - [Changelog](#changelog)
  - [Environment Installation](#environment-installation)
    - [1. Environment Dependencies](#1-environment-dependencies)
    - [2. Environment Setup](#2-environment-setup)
    - [3. Plugin Setup](#3-plugin-setup)
  - [Weight Download](#weight-download)
  - [Dataset Preparation and Processing](#dataset-preparation-and-processing)
  - [Training](#training)
    - [1. Prerequisites](#1-prerequisites)
    - [2. Start Training](#2-start-training)
    - [3. Logging Metrics Description](#3-logging-metrics-description)
  - [Notes](#notes)

<a id="jump0"></a>

## Introduction

This document uses the MindSpeed MM repository to reproduce the [Group Relative Policy Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) post-training method as an example to help you get started quickly. Before starting training, you need to complete the preparatory work including the code repository, environment, dataset, and model weights. Then, launch the training following the instructions provided. The detailed steps are as follows.

<a id="jump0.1"></a>

### Reference Implementation

```shell
url=https://github.com/volcengine/verl
commit_id=97b65c63c729c61ca607315cf7084012aabc6bba
```

<a id="jump0.2"></a>

### Changelog

2025.09.03: Initial support for Qwen2.5-VL 7B/32B GRPO training

<a id="jump1"></a>

## Environment Installation

It is recommended to use the matching environment version for model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/installation.md).

> In the GRPO scenario, the environment dependencies are as follows.

<a id="jump1.1"></a>

### 1. Environment Dependencies

| PyTorch Version | torch_npu Version | Python Version |
|-----------|-------------| ---------- |
| 2.5.1     | 2.5.1       | 3.11 |

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
# python 3.11
conda create -n test python=3.11
conda activate test

# for torch-npu dev version or x86 machine (Optional)
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# The pip install command should be appended with --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com.
# Install torch and torch_npu.
pip install torch-2.5.1-cp311-cp311-*.whl
pip install torch_npu-2.5.1*.manylinux2014_*.whl


# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh

# vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 2961f2f
pip install -v -e .
cd ..

# verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 97b65c63c729c61ca607315cf7084012aabc6bba
pip install -r requirements-npu.txt
pip install -v -e .
cd ..

# Install third-party libraries.
pip install transformers==4.52.4 mathruler==0.1.0 decorator qwen-vl-utils==0.0.11 viztracer cloudpickle==2.1.0 setuptools==80.9.0

# If the installation environment may cause overwriting, you need to reinstall torch_npu.
pip install torch_npu-2.5.1*.manylinux2014_*.whl
```

<a id="jump1.3"></a>

### 3. Plugin Setup

```bash
# Ensure that vllm is correctly installed and will not be overwritten afterwards.
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM/verl_plugin
export MODEL_SELECT="Qwen2_5vl"
# Replace path_to_verl with the verl source code path, e.g., /home/code/verl.
export VERL_PATH=path_to_verl
pip install -v -e .
cp -r ../examples/verl_examples/qwen2.5vl/* ../../verl/examples/grpo_trainer/
cd ../../verl/
```

<a id="jump2"></a>

## Weight Download

Download the corresponding model weights from the Hugging Face library.

- [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)
- [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)

 <a id="jump3"></a>

## Dataset Preparation and Processing

The multimodal model uses the [geo3k](https://huggingface.co/datasets/hiyouga/geometry3k) dataset. Execute the command in the model's root directory to download and process the dataset. `--local_dir` is an optional parameter; if not set, the default location is `~/data/geo3k`.

Ensure network connectivity during the dataset download process. If downloading the original data is slow, you can manually download the original data to a local path and process the data by modifying the code.

```shell
vim ./examples/data_preprocess/geo3k.py
```

```python
## Around line 34, modify data_source to the path of the original data.
dataset = dataset.load_dataset(data_source) # Modify data_source to the dataset download path.
```

```shell
# Download raw data online and preprocess it.
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

<a id="jump4"></a>

## Training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the preliminary preparations, including: **Environment Installation**, **Weight Download**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Start Training

The following uses the Qwen2.5VL 7B model as an example. Before starting the training, you need to modify the configuration of the [startup script](train_qwen2_5_vl_7b_grpo_full.sh).

1. Modify the `NNODES` and `NPUS_PER_NODE` configurations based on the machine being used. For example, for a single-node A2, set `NNODES` to 1 and `NPUS_PER_NODE` to 8.
2. If running on a single machine, ensure that `MASTER_ADDR` is consistent with `CURRENT_IP`. If running on multiple machines, ensure that `MASTER_ADDR` is consistent across all machines, and `CURRENT_IP` is the IP address of each node (Note that neither `MASTER_ADDR` nor `CURRENT_IP` can be set to `localhost`).

    ```shell
    vim examples/grpo_trainer/ray_start.sh
    ```

    Modify the following configuration:

    ```shell
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh # Modify the CANN path.
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Modify the NNAL path.

    NNODES=1 # Number of nodes
    NPUS_PER_NODE=8 # Number of NPUs
    MASTER_ADDR="localhost" # Node IP
    SOCKET_IFNAME="Your SOCKET IFNAME" # Node NIC name
    ```

3. If running on multiple machines, modify the `main_ppo.py` file in verl.

    ```shell
    vim verl/trainer/main_ppo.py
    ```

    Modify the following code around line 64 to enable automatic node information retrieval:

    ```python
    def run_ppo(config) -> None:
        ...
        print(f"ray init kwargs: {ray_init_kwargs}") # Original Code
        # ray.init(**OmegaConf.to_container(ray_init_kwargs)) # Original Code
        ray.init(**OmegaConf.to_container(ray_init_kwargs), address="auto") # New Code
    ```

4. When the actual running scenario is inconsistent with the default configuration script, you need to adjust relevant parameter configuration such as `micro_batch_size` and `dispatch_size` according to the actual scenario.

5. (Qwen2.5VL 32B as an example). If the machine scale is increased, you can modify the configuration of the [startup script](train_qwen2_5_vl_32b_grpo_performance.sh).

    ```shell
    vim examples/grpo_trainer/train_qwen2_5_vl_32b_grpo_performance.sh
    ```

    Modify the following configuration to set the multi-machine scale:

    ```shell
    nnodes=2 # Number of nodes
    n_npus_per_node=16 # Number of NPUs
    ```

    As the scale increases, you can modify the following configurations to reduce FSDP sharding parameter values for the inference part, thereby reducing redundant communication.

     ```shell
    train_batch_size=256 # Increase GBS, for example, modify to 512.
    log_prob_micro_batch_size_per_gpu=4 # Increase communication overlap during inference, for example, modify to 16.
    parameters=0  # Reduce the number of wrapped parameters, for example, modify to 1e7.
    ```

6. (Qwen2.5VL 32B as an example). If you want to increase the number of steps, you can modify the following parameter configuration.

    ```shell
    vim examples/grpo_trainer/train_qwen2_5_vl_32b_grpo_performance.sh
    ```

    Increase the `total_epochs` value to the desired total number of steps (300 epochs as an example), and set `total_training_steps` to `null`.

    ```shell
    trainer.total_epochs=300 \
    trainer.total_training_steps='null' \
    ```

7. If you need to use deterministic computation, add `export DETERMINISTIC=True` during [Plugin Installation](#jump1.3).

    ```bash
    # Add:
    export DETERMINISTIC=True
    # Then, perform the following operations:
    export MODEL_SELECT="Qwen2_5vl"
    # Replace `path_to_verl` with the actual path to the verl source code, e.g., `/home/code/verl`
    export VERL_PATH=path_to_verl
    pip install -v -e .
    ```

    Deterministic computation can be disabled by commenting out `seed_all()` around line 114 in the `verl/workers/fsdp_workers.py` file.

8. Start Ray. If running on multiple machines, this script must be run sequentially from the master node to the slave nodes.

    ```bash
    bash examples/grpo_trainer/ray_start.sh
    ```

9. Start the training script.
    - `model_path` is the model weight path, and `data_path` is the dataset path.
    - For multi-machine running, run this script only **on the master node**.

    ```bash
    bash examples/grpo_trainer/train_qwen2_5_vl_7b_grpo_full.sh --data_path=xxx --model_path=xxx
    ```

> Note: The directory hierarchy for code, weights, data, and other paths must be consistent across all nodes, and Ray must be started from within the verl directory.

<a id="jump4.3"></a>

### 3. Logging Metrics Description

**Time-related Metrics Description**

| Metric                                 | Description                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `timing_s/gen`                     | Time elapsed for generation in one iteration                     |
| `timing_s/reward`         | Time elapsed for reward in one iteration                                     |
| `timing_s/old_log_prob`                   | Time elapsed for the actor model to calculate log prob in one iteration                       |
| `timing_s/ref`             | Time elapsed for the reference model calculation in one iteration                   |
| `timing_s/adv`                         | Time elapsed for calculating advantages                                       |
| `timing_s/reshard`         | Time elapsed for resharding in one iteration                                     |
| `timing_s/update_actor`                      | Time elapsed for the actor model update in one iteration                      |
| `timing_s/step`                      | Total time for one iteration                      |
| `timing_s/generate_sequence`                      | Time elapsed for generate_sequence in one iteration                      |

**Other Metrics**

| Metric                                    | Description                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| `actor/entropy`                         | Policy entropy, indicating the randomness or exploration capability of the policy                           |
| `actor/kl_loss`                         | KL divergence, measuring the degree of deviation between the current policy and a reference policy (e.g., old policy or reference model) |
| `actor/pg_loss`                         | PG loss based on the advantage function, representing the current policy's ability to learn from reward improvement |
| `actor/pg_clipfrac`                     | Proportion of clipping mechanism activation in GRPO, reflecting the stability of policy update magnitude         |
| `actor/ppo_kl`                          | Actual KL divergence of the PPO algorithm                                        |
| `actor/grad_norm`                             | Gradient norm, representing the overall magnitude of parameter gradients during backpropagation               |
| `actor/lr`                               | Learning rate, currently used by the optimizer                               |
| `critic/score/mean`                       | Mean reward value when a reward model is enabled                                   |
| `critic/score/max`                        | Maximum reward value from both the reward model and rule-based rewards for the same sample                 |
| `critic/score/min`                       | Minimum reward value from both the reward model and rule-based rewards for the same sample                 |
| `critic/rewards/mean`                     | Mean of rule-based rewards; mean normalized reward from the reward model for samples |
| `critic/rewards/max`                      | Maximum of rule-based rewards; maximum normalized reward from the reward model for samples |
| `critic/rewards/min`                      | Minimum of rule-based rewards; minimum normalized reward from the reward model for samples |
| `response_length/mean`                  | Average response length, the average number of tokens generated in model responses        |
| `response_length/min`                   | Minimum response length, the shortest response length in the current batch          |
| `response_length/max`                   | Maximum response length, the longest response length in the current batch          |
| `prompt_length/mean`                    | Average prompt length, the average length of input prompts                         |
| `prompt_length/max`                     | Maximum prompt length, the longest prompt length in the current batch                 |
| `prompt_length/min`                     | Minimum prompt length, the shortest prompt length in the current batch                 |
| `perf/total_num_tokens`                               | Total number of tokens                                       |
| `perf/time_per_step`                            | Time elapsed per step                                         |
| `perf/throughput`                              | Throughput metric                                         |

<a id="jump5"></a>

## Notes

1. When starting inside a container, you may encounter an error indicating that the `ip` command does not exist. You can run the following command for installation:

    ```shell
    sudo apt-get install iproute2
    ```

2. If the vllm-ascend installation fails with the error `fatal error: 'cstdint' file not found`, it may be a GCC version issue. Refer to [this guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0086.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) for a solution. For more vllm-ascend issues, you can seek help from the [community](https://github.com/vllm-project/vllm-ascend).

3. You can use the [Ray Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html) to debug the code. After installing the plugins, you need to install the dependencies in your environment:

    ```shell
    pip install "ray[default]" debugpy
    ```
