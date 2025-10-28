# Qwen2_5_VL GRPO 使用指南

<p align="left">
</p>

## 目录

- [简介](#jump0)
  - [参考实现](#jump0.1)
  - [变更记录](#jump0.2)
- [环境安装](#jump1)
  - [环境依赖](#jump1.1)
  - [环境搭建](#jump1.2)
  - [安装插件](#jump1.3)
- [权重下载及转换](#jump2)
- [数据集准备及处理](#jump3)
- [训练](#jump4)
  - [准备工作](#jump4.1)
  - [启动训练](#jump4.2)
  - [日志打点指标说明](#jump4.3)
- [注意事项](#jump5)
- [性能数据](#jump6)

<a id="jump0"></a>

## 简介

以 MindSpeed MM 仓库复现 [Group Relative Policy Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) 后训练方法为例来帮助用户快速入门，前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。

<a id="jump0.1"></a>

#### 参考实现

```
url=https://github.com/volcengine/verl
commit_id=97b65c63c729c61ca607315cf7084012aabc6bba
```

<a id="jump0.2"></a>

#### 变更记录

2025.09.03: 首次支持Qwen2.5-VL 7B/32B GRPO训练

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/user-guide/installation.md)

> GRPO场景下，环境依赖如下：

<a id="jump1.1"></a>

#### 1. 环境依赖

| MindSpeed RL版本 | PyTorch版本 | torch_npu版本 | CANN版本  | Python版本 |
| ---------------- | ------------ |-----------| ---------- | ---------- |
| master（主线）   | 2.5.1     | 2.5.1       | 8.3.RC1 | Python3.11 |

<a id="jump1.2"></a>

#### 2. 环境搭建

```bash
# python3.11
conda create -n test python=3.11
conda activate test

# for torch-npu dev version or x86 machine [Optional]
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# pip install 后缀需加上 --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
# 安装torch和torch_npu
pip install torch-2.5.1-cp311-cp311-*.whl
pip install torch_npu-2.5.1*.manylinux2014_*.whl


source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
VLLM_TARGET_DEVICE=empty pip install -v -e .
# for x86 machine [Optional]
# VLLM_TARGET_DEVICE=empty pip install -v -e . --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
cd ..

# vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 2961f2f
pip install -v -e .
# for x86 machine [Optional]
# pip install -v -e . --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
cd ..

# verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout 97b65c63c729c61ca607315cf7084012aabc6bba
pip install -r requirements-npu.txt
pip install -v -e .
# for x86 machine [Optional]
# pip install -r requirements-npu.txt --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
# pip install -v -e . --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
cd ..

# 安装三方库
pip install transformers==4.52.4 mathruler==0.1.0 decorator qwen-vl-utils==0.0.11 viztracer cloudpickle==2.1.0
# for x86 machine [Optional]
# pip install transformers==4.52.4 mathruler==0.1.0 decorator qwen-vl-utils==0.0.11 viztracer cloudpickle==2.1.0 --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com

# 因安装环境可能导致覆盖，需重新安装torch_npu
pip install torch_npu-2.5.1*.manylinux2014_*.whl
```

<a id="jump1.3"></a>

#### 3. 安装插件

```bash
# 请确保 vllm 已正确安装并且之后不会做覆盖
git clone --branch 2.2.0 https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM/examples/verl_plugin
pip install -v -e .
scp -r scripts/* ../../../verl/examples/grpo_trainer/
cd ../../../verl/
```

**注意**：安装插件前需要保证verl源码安装，否则插件不能生效。如果无法源码安装verl，需要指定verl源码路径：

```bash
VERL_PATH=path_to_verl pip install -e .
```

<a id="jump2"></a>

## 权重下载

从Huggingface库下载对应的模型权重:

- 模型地址: [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main)；
- 模型地址: [Qwen2.5-VL-32B](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/tree/main)；

<a id="jump3"></a>

## 数据集准备及处理

多模态模型使用[geo3k](https://huggingface.co/datasets/hiyouga/geometry3k)数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/geo3k`。

下载数据集的过程中，保证网络正常。若下载原始数据比较慢，可以手动下载原始数据到本地，通过修改代码进行数据处理：

```shell
vim ./examples/data_preprocess/geo3k.py
```

```python
## 34行附近修改data_source为原始数据路径
dataset = dataset.load_dataset(data_source) # 将data_source修改为数据集下载路径
```

```shell
# 在线下载原始数据并预处理
python ./examples/data_preprocess/geo3k.py --local_dir=./data/geo3k
```

<a id="jump4"></a>

## 训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 启动训练

以 Qwen2.5VL 7B 模型为例,在启动训练之前，需要修改[启动脚本](../../examples/verl_plugin/scripts/train_qwen2_5_vl_7b_grpo_full.sh)的配置：

1. 根据使用机器的情况，修改`NNODES`、`NPUS_PER_NODE`配置， 例如单机 A2 可设置`NNODES`为 1 、`NPUS_PER_NODE`为8；
2. 如果是单机，需要保证`MASTER_ADDR`与`CURRENT_IP`一致，如果为多机，需要保证各个机器的`MASTER_ADDR`一致，`CURRENT_IP`为各个节点的 IP (需要注意的是`MASTER_ADDR`与`CURRENT_IP`不能设置为`localhost`)；

    ```shell
    vim examples/grpo_trainer/ray_start.sh
    ```

    修改以下配置：

    ```shell
    source /usr/local/Ascend/ascend-toolkit/set_env.sh # 修改cann路径
    source /usr/local/Ascend/nnal/atb/set_env.sh # 修改nnal路径

    NNODES=1 # 节点数
    NPUS_PER_NODE=8 # NPU卡数
    MASTER_ADDR="localhost" # 节点IP
    SOCKET_IFNAME="Your SOCKET IFNAME" # 节点网卡名称
    ```

3. 若多机运行，需修改verl中的`main_ppo.py`文件：

    ```shell
    vim verl/trainer/main_ppo.py
    ```

    在64行附近修改以下代码，添加自动获取节点信息：

    ```python
    def run_ppo(config) -> None:
        ...
        print(f"ray init kwargs: {ray_init_kwargs}") # 原代码
        # ray.init(**OmegaConf.to_container(ray_init_kwargs)) # 原代码
        ray.init(**OmegaConf.to_container(ray_init_kwargs), address="auto") # 新代码
    ```

4. 实际运行场景与默认配置脚本不一致时，需要根据实际场景调整`micro_batch_size`、`dispatch_size`等相关配置参数。

5. 以 Qwen2.5VL 32B 模型为例,若增加机器规模的情况下，可通过修改[启动脚本](../../examples/verl_plugin/scripts/train_qwen2_5_vl_32b_grpo_full.sh)的配置：

    ```shell
    vim examples/grpo_trainer/train_qwen2_5_vl_32b_grpo_full.sh
    ```

    修改以下配置，设置多机规模：

    ```shell
    nnodes=1 # 节点数
    n_gpus_per_node=16 # NPU卡数
    ```

    规模增加，可修改以下配置，将推理部分减少FSDP切分参数，减少冗余通讯：

     ```shell
    train_batch_size=256 # 增大GBS，如修改为512
    log_prob_micro_batch_size_per_gpu=4 # 增加推理阶段对通讯的掩盖，如修改为16
    parameters=0  # 减少wrap参数，如修改为1e7
    ```

6. 以 Qwen2.5VL 32B 模型为例，若想要增大步数，可修改以下配置参数：

    ```shell
    vim examples/grpo_trainer/train_qwen2_5_vl_32b_grpo_full.sh
    ```

    增加`total_epochs`配置参数（需大于total_training_steps）：

    ```shell
    trainer.total_epochs=300 \
    trainer.total_training_steps=150 \ # 原代码
    ```

7. 启动ray, 若多机运行，需主节点到副节点依次运行此脚本：

    ```bash
    bash examples/grpo_trainer/ray_start.sh
    ```

8. 启动训练脚本：
    - `model_path`为模型权重路径，`data_path`为数据集路径
    - 若多机运行，*仅需主节点*需运行此脚本

    ```bash
    bash examples/grpo_trainer/train_qwen2_5_vl_7b_grpo_full.sh --data_path=xxx --model_path=xxx
    ```

> *注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于verl目录下*

<a id="jump4.3"></a>

#### 3. 日志打点指标说明

**时间相关指标说明**

| 指标                                 | 说明                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `timing_s/gen`                     | 一次迭代中generation耗时                     |
| `timing_s/reward`         | 一次迭代中reward耗时                                     |
| `timing_s/old_log_prob`                   | 一次迭代中actor model计算log prob耗时                       |
| `timing_s/ref`             | 一次迭代中reference model计算耗时                   |
| `timing_s/adv`                         | 计算advantages耗时                                       |
| `timing_s/reshard`         | 一次迭代中reshard耗时                                     |
| `timing_s/update_actor`                      | 一次迭代中actor model进行update耗时                      |
| `timing_s/step`                      | 一次迭代总时间                      |
| `timing_s/generate_sequence`                      | 一次迭代中generate_sequence耗时                      |

**其他指标**

| 指标                                    | 说明                                                         |
| --------------------------------------- | ------------------------------------------------------------ |
| `actor/entropy`                         | 策略熵，表示策略的随机性或探索能力                           |
| `actor/kl_loss`                         | kl散度，衡量当前策略与参考策略（如旧策略或参考模型）之间的偏离程度 |
| `actor/pg_loss`                         | pg_loss，基于优势函数的策略梯度目标函数值，表示当前策略对提升奖励的学习能力。 |
| `actor/pg_clipfrac`                     | GRPO中裁剪机制生效的比例，反映了策略更新幅度的稳定性         |
| `actor/ppo_kl`                          | PPO算法的实际 KL 散度                                        |
| `actor/grad_norm`                             | 梯度范数，表示当前反向传播中参数梯度的整体幅度               |
| `critic/rewards/mean`                     | 规则奖励打分的平均总奖励值                                   |
| `actor/lr`                               | 学习率，优化器当前使用的学习率                               |
| `critic/score/mean`                       | 开启奖励模型时的reward均值                                   |
| `critic/score/max`                        | 奖励模型及规则奖励对同一个样本的reward最大值                 |
| `critic/score/min`                       | 奖励模型及规则奖励对同一个样本的reward最小值                 |
| `critic/rewards/mean`                     | 规则奖励的reward均值；奖励模型对样本的reward经过归一化后的均值 |
| `critic/rewards/max`                      | 规则奖励的reward最大值；奖励模型对样本的reward经过归一化后的最大值 |
| `critic/rewards/min`                      | 规则奖励的reward最小值；奖励模型对样本的reward经过归一化后的最小值 |
| `response_length/mean`                  | 平均生成长度，模型生成回复（response）的平均 token 数        |
| `response_length/min`                   | 最短生成长度，当前 batch 中生成最短的 response 长度          |
| `response_length/max`                   | 最长生成长度，当前 batch 中生成最长的 response 长度          |
| `prompt_length/mean`                    | 平均输入长度，输入 prompt 的平均长度                         |
| `prompt_length/max`                     | 最长输入长度，当前 batch 中最长的 prompt长度                 |
| `prompt_length/min`                     | 最短输入长度，当前 batch 中最长的 prompt长度                 |
| `perf/total_num_tokens`                               | 总tokens数                                       |
| `perf/time_per_step`                            | 每步耗时                                         |
| `perf/throughput`                              | 吞吐指标                                         |

---
<a id="jump5"></a>

## 注意事项

1. 容器内启动时可能会遇到不存在`ip`命令的错误，可使用如下命令进行安装：

```shell
sudo apt-get install iproute2
```

2. 如果安装vllm ascend失败，提示`fatal error: 'cstdint' file not found`，可能是gcc版本问题，可参考[此处](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/softwareinst/instg/instg_0086.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)解决。更多vllm ascend问题可以向[社区](https://github.com/vllm-project/vllm-ascend)求助。

3. 可以使用[Ray Debugger](https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html)对代码进行调试，在安装完插件后，需要在环境中安装依赖：

```shell
pip install "ray[default]" debugpy
```

---

<a id="jump6"></a>

## 性能数据

**注**：geo3k数据集。

| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_response_length | max_num_batched_tokens | 端到端 tps |
|---------------------|----------|-----|-----------|-------------------|------------|---------|---------|
| Qwen2.5VL-7B          | Atlas 200T A2 Box16 | 512  | 5        | 1024              | 2048       | 8192 | 142.42     |
| Qwen2.5VL-32B          | Atlas 200T A2 Box16 | 256  | 5        | 1024              | 2048       | 8192 | 103.50     |

**注**：因数据集为非公开数据，此性能结果仅供参考。

| 模型                  | 机器型号     | GBS | n_samples | max_prompt_length | max_response_length | max_num_batched_tokens | 端到端 tps |
|---------------------|----------|-----|-----------|-------------------|------------|---------|---------|
| Qwen2.5VL-7B          | Atlas 200T A2 Box16 | 16  | 4        | 18,000              | 512       | 19,000 | 428.38     |
| Qwen2.5VL-32B          | Atlas 200T A2 Box16 | 32  | 8        | 18,000              | 512       | 20,000 | 99.65     |
