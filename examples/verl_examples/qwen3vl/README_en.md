# Qwen3_VL GRPO User Guide

<p align="left">
</p>

<a id="jump0"></a>

## Introduction

Taking the reproduction of the [Group Relative Policy Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) post-training method in the MindSpeed MM repository as an example to help you get started quickly. Before starting training, you need to complete the preparatory work including the code repository, environment, dataset, and model weights. Then, launch the training following the instructions provided. The detailed steps are as follows.

<a id="jump0.1"></a>

### Reference Implementation

```shell
url=https://github.com/volcengine/verl
commit_id=7df2afb936cd37b7b3a262edc119b2a57f070e3b
```

<a id="jump1"></a>

## Environment Setup

It is recommended to use the matching environment version during model development.

Please refer to the [Installation Guide](../../../docs/en/pytorch/installation.md).

> For GRPO scenarios, CANN 8.5.1 or later is recommended. Other environment dependencies are as follows:

<a id="jump1.1"></a>

### 1. Environment Dependencies

| PyTorch Version | torch_npu Version | Python Version |
|-----------|-------------| ---------- |
| 2.7.1     | 2.7.1       | 3.11 |

<a id="jump1.2"></a>

### 2. Environment Setup

```bash
# python3.11
conda create -n test python=3.11
conda activate test

# for x86 machine
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# The pip install suffix must include `--trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com`.

# for arm64 machine
pip config set global.extra-index-url "https://mirrors.huaweicloud.com/ascend/repos/pypi"
# For pip install, simply append the suffix `--trusted-host mirrors.huaweicloud.com`.

# Install cmake. If the cmake version is too low, you need to upgrade it; version 3.26.4 is recommended.
conda install -c conda-forge cmake=3.26.4

# Install torch, torch_npu, and pybind11==3.0.1
pip install torch-2.7.1-cp311-cp311-*.whl
pip install torch_npu-2.7.1*.manylinux_*.whl
pip install pybind11==3.0.1

# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/cann/set_env.sh
# Modify the ascend-toolkit path according to the actual situation.
source /usr/local/Ascend/nnal/atb/set_env.sh

# vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.11.0
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout fed8145
pip install -r requirements.txt
pip install -v -e .
# If you encounter compilation failures, please check point 2 in the "Notes" section below.
cd ..

# verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 7df2afb
pip install -r requirements.txt
pip install -v -e .
cd ..

# transformers
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 7a833d1ccd41673030c85107f65f454c0c3222f5
pip install '.[torch]'
cd ..

# Install third-party libraries.
pip install qwen-vl-utils==0.0.11 mathruler viztracer uvloop==0.21.0 setuptools==80.9.0 cloudpickle==3.1.2

# The installation environment may cause overwrites, In this case, torch_npu needs to be reinstalled.
pip install torch_npu-2.7.1*.manylinux_*.whl
```

<a id="jump1.3"></a>

### 3. Plugin Installation

```bash
# Ensure that vllm is correctly installed and will not be overwritten later.
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM/verl_plugin
export MODEL_SELECT="Qwen3vl"
# Replace path_to_verl with the path to the verl source code, for example: /home/code/verl.
export VERL_PATH=path_to_verl
pip install -v -e .
# To confirm whether the plugin is installed, please refer to point 4 in the "Notes" below.
cp -r ../examples/verl_examples/qwen3vl/* ../../verl/examples/grpo_trainer/
cd ../../verl/
```

<a id="jump2"></a>

## Weight Download

Download the corresponding model weights from the Hugging Face library:

- [Qwen3-VL-8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main)
- [Qwen3-VL-30B](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main)

<a id="jump3"></a>

## Dataset Preparation and Processing

For multimodal models, use the [geo3k](https://huggingface.co/datasets/hiyouga/geometry3k) dataset. Execute the command in the model's root directory to download and process the dataset. `--local_save_dir` is an optional parameter; if not set, the default location is `~/data/geo3k`.

Ensure a stable network connection while downloading the dataset.

```shell
# Download raw data online and preprocess it
python ./examples/data_preprocess/geo3k.py --local_save_dir=./data/geo3k
```

If downloading the raw data is slow, you can manually download it to your local machine and process the data by modifying the code.

```shell
vim ./examples/data_preprocess/geo3k.py
```

```python
## Around line 44, modify `data_source` to the path of the raw data.
dataset = dataset.load_dataset(data_source) # Modify data_source to the dataset download path.
```

<a id="jump4"></a>

## Training

<a id="jump4.1"></a>

### 1. Prerequisites

Before configuring the script, you need to complete the prerequisite preparations, including: **Environment Setup**, **Weight Download**, and **Dataset Preparation and Processing**. For details, refer to the corresponding sections.

<a id="jump4.2"></a>

### 2. Start Training

The following uses the Qwen2.5VL 7B model as an example. Before starting the training, you need to modify the configuration of the [startup script](train_qwen3_vl_8b_grpo_full.sh).

1. Modify the `NNODES` and `NPUS_PER_NODE` configurations according to the machines used. For example, for a single-node A2, you can set `NNODES` to `1` and `NPUS_PER_NODE` to `8`.

2. For a single machine, ensure that `MASTER_ADDR` is consistent with `CURRENT_IP`. For multiple machines, ensure that the `MASTER_ADDR` is consistent across all machines, and `CURRENT_IP` is the IP address of each node (note that `MASTER_ADDR` and `CURRENT_IP` cannot be set to `localhost`).

    ```shell
    vim examples/grpo_trainer/ray_start.sh
    ```

    Modify the following configurations:

    ```shell
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/cann/set_env.sh # Modify the CANN path.
    # Modify the ascend-toolkit path according to the actual situation.
    source /usr/local/Ascend/nnal/atb/set_env.sh # Modify nnal path.

    NNODES=1 # Number of nodes
    NPUS_PER_NODE=8 # Number of NPUs
    MASTER_ADDR="IP FOR MASTER NODE" # Node IP
    SOCKET_IFNAME="Your SOCKET IFNAME" # Node NIC name
    ```

3. When the actual running scenario is inconsistent with the default configuration script, you need to adjust relevant parameter configuration such as `micro_batch_size` and `dispatch_size` according to the actual scenario.

4. (Qwen2.5VL 32B as an example). If the machine scale is increased, you can modify the configuration of the [startup script](train_qwen3_vl_30b_grpo_full.sh).

    ```shell
    vim examples/grpo_trainer/train_qwen3_vl_30b_grpo_full.sh
    ```

    Modify the following configurations to set the multi-machine scale:

    ```shell
    nnodes=1 # Number of Nodes
    n_npus_per_node=16 # Number of NPU Cards
    ```

    To scale up, you can modify the following configurations to reduce FSDP sharding parameter values in the inference part and decrease redundant communication:

     ```shell
    train_batch_size=64 # Increase GBS, for example, modify to 512.
    log_prob_micro_batch_size_per_gpu=4 # Increase communication overlap during inference, for example, modify to 16.
    parameters=0  # Reduce the number of wrapped parameters, for example, modify to 1e7.
    ```

5. (Qwen2.5VL 32B as an example). If you want to increase the number of steps, you can modify the following parameter configuration.

    ```shell
    vim examples/grpo_trainer/train_qwen3_vl_30b_grpo_full.sh
    ```

Increase the `total_epochs` configuration parameter to the desired total number of training steps (300 epochs as an example), and set `total_training_steps` to `null`.

    ```shell
    trainer.total_epochs=300 \
    trainer.total_training_steps='null' \
    ```

6. For the Qwen3VL 30B model, using the same dataset, modify `prompt_length` to `16k` and `response_length` to `1k`. The[train_qwen3_vl_30b_16k_grpo_full.sh](train_qwen3_vl_30b_16k_grpo_full.sh) script can be used. In this script, `padding_mode=1` indicates the padding mode for the input and output lengths specified above, corresponding to the following two configurations:

   ```shell
    +data.padding_mode=$padding_mode,
    +actor_rollout_ref.rollout.engine_kwargs.padding_mode=$padding_mode \
    ```
7. If deterministic computation is required, add `export DETERMINISTIC=True` during the [Plugin Installation](#jump1.3).

    ```bash
    # Add:
    export DETERMINISTIC=True
    # Then, perform the following operations:
    export MODEL_SELECT="Qwen3vl"
    # Replace `path_to_verl` with the source code path of verl, for example: `/home/code/verl`
    export VERL_PATH=path_to_verl
    pip install -v -e .
    ```

    Deterministic computation can be disabled by commenting out `seed_all()` near line 114 in the `verl/workers/fsdp_workers.py` file.

8. Start Ray. If running on multiple machines, this script must be run sequentially from the master node to the slave nodes.

    ```bash
    bash examples/grpo_trainer/ray_start.sh
    ```

9. Start the training script.
    - `model_path` is the model weight path, and `data_path` is the dataset path.
    - For multi-machine running, run this script only **on the master node**.

    ```bash
    bash examples/grpo_trainer/train_qwen3_vl_8b_grpo_full.sh --data_path=xxx --model_path=xxx
    ```

> Note: The directory hierarchies for code, weights, data, and other paths must remain consistent across all nodes, and Ray must be started from the `verl` directory.

<a id="jump4.3"></a>

### 3. Log Metrics Description

**Time-Related Metrics Description**

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

2. If the vllm-ascend installation fails with the error `fatal error: 'cstdint' file not found`, it may be due to a GCC version issue. Refer to [this guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0086.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit) for a solution. For more vllm-ascend issues, you can seek help from the [community](https://github.com/vllm-project/vllm-ascend).

3. You can use the [Ray Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html) to debug the code. After installing the plugin, you need to install the dependencies in your environment:

    ```shell
    pip install "ray[default]" debugpy
    ```

4. To confirm that the plugin installation is complete, check whether the file has been modified successfully (run `vi ../../verl/verl/__init__.py` to verify that code such as `import verl_npu` has been appended to the end of the file). If the modification was not successful, it is recommended to change the plugin installation command to:

    ```shell
    pip install -v .
    ```
