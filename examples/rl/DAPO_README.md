# Qwen2_5_VL DAPO 使用指南

<p align="left">
</p>

## 目录
- [简介](#jump0)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
  - [VLLM及VLLM-ASCEND安装](#jump1.3)
  - [高性能内存库jemalloc安装](#jump1.4)
- [权重下载及转换](#jump2)
- [数据集准备及处理](#jump3)
- [训练](#jump4)
  - [准备工作](#jump4.1)
  - [启动训练](#jump4.2)
  - [日志打点指标说明](#jump4.3)
- [断点续训](#jump5)
- [注意事项](#jump6)

<a id="jump0"></a>
## 简介
以 MindSpeed MM 仓库复现 [Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO) ](https://dapo-sia.github.io/static/pdf/dapo_paper.pdf) 后训练方法为例来帮助用户快速入门，前期需要完成代码仓、环境、数据集以及权重等准备工作，再按照说明中的启动方式启动训练，以下为具体的操作说明。


#### 参考实现
```
url=https://github.com/volcengine/verl
commit_id=c70b7470c108168339392a3a1908bc77e0c375db
```

#### 变更记录

2025.09.04: 首次支持Qwen2.5-VL 7B DAPO训练

<a id="jump1"></a>
## 环境安装
【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/user-guide/installation.md)

> DAPO场景下，Python版本推荐3.10，torch和torch_npu版本推荐2.5.1版本

<a id="jump1.1"></a>
#### 1. 仓库拉取

```shell
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cp -r megatron ../MindSpeed-MM/
cd ..

cd MindSpeed-MM
# 安装mm-convert工具
pip install -e . --no-deps
mkdir -p logs data ckpt
cd ..
```

<a id="jump1.2"></a>
#### 2. 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# for torch-npu dev version or x86 machine [Optional]
# pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
# 安装torch和torch_npu
pip install torch-2.5.1-cp310-cp310-*.whl
pip install torch_npu-2.5.1*.manylinux2014_aarch64.whl

# 安装MindSPeed MM依赖
pip install -r MindSpeed-MM/examples/rl/requirements.txt

# apex for Ascend 参考 https://gitcode.com/Ascend/apex
# 建议从原仓编译安装，需要先source CANN环境

# 安装加速库
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
# checkout commit from MindSpeed core_r0.12.1
git checkout 5176c6f5f133111e55a404d82bd2dc14a809a6ab
cp -r mindspeed ../MindSpeed-MM/
cd ..

# 下载MindSpeed RL
git clone https://gitcode.com/Ascend/MindSpeed-RL.git
cd MindSpeed-RL
git checkout d9212e12aec839af47f007b971a3e1454519c458
pip install -r requirements.txt
cp -r mindspeed_rl ../MindSpeed-MM/
cd ..

cd MindSpeed-MM
# MindSpeed/Megatron 中文件替换（必选，MindSpeed-RL使用的mindspeed版本是080，而MindSpeed-MM使用的是012）
bash examples/rl/scripts/apply_dapo_mindspeed_patches.sh
cd ..
```

<a id="jump1.3"></a>
#### 3. VLLM及VLLM-ASCEND安装

> 源码安装，若使用docker，建议源码存放在docker内

``` shell
# 下载vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout b6553be1bc75f046b00046a4ad7576364d03c835
cd ../

# 下载vllm-ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout 03b2910364c4e40d968f6d66992ea49119d80dee
cd ../

# 安装VLLM
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..

# VLLM安装可能会升级numpy版本，numpy版本要求为1.26.4
pip install numpy==1.26.4

# 安装VLLM-ASCEND，需导入CANN及ATB环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
cd vllm-ascend
pip install -r requirements-dev.txt
# 因为前面已经安装了对应的torch_npu版本，这里需要将vllm_ascend文件中的requirements.txt中的torch-npu==2.5.1注释
# 安装时若编译失败，可以关闭编译：export COMPILE_CUSTOM_KERNELS=0
python setup.py develop
# vllm-ascend源码安装过程中遇到相关依赖包因网络问题安装不成功，可以先尝试pip install xxx安装对应失败的依赖包，再执行上一句命令
cd ..

# 在安装完VLLM及VLLM-ASCEND后，需检查torch及torch_npu版本，若版本被覆盖，需再次安装torch及torch_npu
```

<a id="jump1.4"></a>
#### 4. 高性能内存库 jemalloc 安装
为了确保 Ray 进程能够正常回收内存，需要安装并使能 jemalloc 库进行内存管理。
#####  Ubuntu 操作系统
通过操作系统源安装jemalloc（注意： 要求ubuntu版本>=20.04）：
```shell
sudo apt install libjemalloc2
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
# arm64架构
export LD_PRELOAD="$LD_PRELOAD:/usr/lib/aarch64-linux-gnu/libjemalloc.so.2"
# x86_64架构
export LD_PRELOAD="$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"
```
##### OpenEuler 操作系统

执行如下命令从操作系统源安装jemalloc
```shell
yum install jemalloc
```
如果上述方法无法正常安装，可以通过源码编译安装
前往jemalloc官网下载最新稳定版本，官网地址:https://github.com/jemalloc/jemalloc/releases/
```shell
tar -xvf jemalloc-{version}.tar.bz2
cd jemalloc-{version}
./configure --prefix=/usr/local
make
make install
```
在启动任务前执行如下命令通过环境变量导入jemalloc：
```shell
#根据实际安装路径设置环境变量，例如安装路径为:/usr/local/lib/libjemalloc.so.2,可通过以下命令来设置环境变量
export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/libjemalloc.so.2"
```
> 如以上安装过程出现错误，可以通过提出issue获得更多解决建议。


<a id="jump2"></a>
## 权重下载及转换

根据具体任务要求，参考Qwen2.5VL[权重下载及转换](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/qwen2.5vl#权重下载及转换)获取相应的权重。

> 若未安装`mm-convert`工具，可以使用`python checkpoint/convert_cli.py`代替，如：
```
python checkpoint/convert_cli.py Qwen2_5_VLConverter hf_to_mm \
  --cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
  --cfg.parallel_config.llm_pp_layers [[36]] \
  --cfg.parallel_config.vit_pp_layers [[32]] \
  --cfg.parallel_config.tp_size 1
```

<a id="jump3"></a>
## 数据集准备及处理

多模态模型使用[geo3k](https://huggingface.co/datasets/hiyouga/geometry3k)数据集，在模型根目录下执行命令，下载并处理数据集，`--local_dir`为可选参数，不设置默认下载位置为`~/data/geo3k`。

下载数据集的过程中，保证网络正常。若下载原始数据比较慢，可以手动下载原始数据到本地，通过`--local_data`参数指定原始数据本地路径。

```shell
# 在线下载原始数据并预处理
python examples/rl/data_preprocess/geo3k.py --local_dir=./data/geo3k

# 基于本地原始数据集预处理
python examples/rl/data_preprocess/geo3k.py --local_dir=./data/geo3k --local_data=/path/geometry3k
```

<a id="jump4"></a>
## 训练

<a id="jump4.1"></a>
#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载及转换**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>
#### 2. 启动训练

以 Qwen2.5VL 7B 模型为例,在启动训练之前，需要修改[ 启动脚本 ](../../examples/rl/scripts/dapo_trainer_qwen25vl_7b.sh)的配置：
1. 根据实际安装路径设置 jemalloc 环境变量，用于更好管理内存，避免长跑过程中内存 OOM ，例如：export LD_PRELOAD="$LD_PRELOAD:/usr/local/lib/libjemalloc.so.2"
2. 可参考 [单卡多进程介绍](../../docs/features/integrated_worker.md)，进行actor与vit的共卡配置
3. 修改 DEFAULT_YAML 为指定的 yaml，目前已支持的配置文件放置在`examples/rl/configs`文件夹下，具体参数说明可见 [配置文件参数介绍](https://gitcode.com/Ascend/MindSpeed-RL/blob/master/docs/features/grpo_yaml.md)
需要注意配置以下参数：
  ```yaml
model: examples/rl/model/qwen2.5vl_7b.json
megatron_training:
    tokenizer_name_or_path: ckpt/hf_path/Qwen2.5-VL-7B-Instruct # 原始权重路径
    data_path: /path/data/geometry3k # 数据集路径
    use_deter_comp: true # 是否开启确定性计算

actor_config:
    load: ckpt/mm_path/Qwen2.5-VL-7B-Instruct # megatron权重路径
    save: ./ckpt # 权重保存路径

# 在colocate_actor_and_vit为True时，需设置如下vit_config权重路径
vit_config:
    load: ckpt/mm_path/Qwen2.5-VL-7B-Vit-Instruct # 视觉模型megatron格式的权重路径

generate_config:
    # 为保证代码安全，配置trust_remote_code默认为False，用户需要设置为True，并且确保自己下载的模型和数据的安全性
    trust_remote_code: true

rl_config:
    #token level loss策略
    token_level_loss: true                                <------- 默认开启

    #clip Higher策略
    clip_higher_enable: true                              <------- 默认关闭
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28

    #过长response惩罚措施
    overlong_buffer_enable: true                          <------- 默认关闭
    rollout_max_tokens : 2048                             <------- response最大长度
    overlong_buffer: 512                                  <------- 超长惩罚缓冲区大小
    overlong_buffer_penalty_factor: 1.0                   <------- 超长惩罚系数

    #动态采样过滤措施
    filter_groups_enable: true                            <------- 默认关闭
    filter_groups_metric: acc_for_dapo                    <------- 指定用于过滤的 metric，其值需要包含在verifier_function参数设置的列表中
    filter_groups_max_batches: -1                         <------- 设置过滤的最大次数，-1 代表不限制最大次数
    filter_groups_train_batch_size: 32                    <------- 制定需要筛选出多少条数据才停止采样，建议与gbs值一致
  ```
4. 根据使用机器的情况，修改`NNODES`、`NPUS_PER_NODE`配置， 例如单机 A2 可设置`NNODES`为 1 、`NPUS_PER_NODE`为8；
5. 如果是单机，需要保证`MASTER_ADDR`与`CURRENT_IP`一致，如果为多机，需要保证各个机器的`MASTER_ADDR`一致，`CURRENT_IP`为各个节点的 IP (需要注意的是`MASTER_ADDR`与`CURRENT_IP`不能设置为`localhost`)；

```bash
#上述注意点修改完毕后，可启动脚本开启训练
bash examples/rl/scripts/dapo_trainer_qwen25vl_7b.sh
```

> *注意：所有节点的代码、权重、数据等路径的层级要保持一致，且启动ray的时候都位于MindSpeed-MM目录下*

6. 实际运行场景与默认配置脚本不一致时，需要根据实际场景调整`micro_batch_size`、`dispatch_size`等相关配置参数。

<a id="jump4.3"></a>
#### 3. 日志打点指标说明

**时间相关指标说明**

| 指标                                 | 说明                                                     |
| ------------------------------------ | -------------------------------------------------------- |
| `timing/all`                         | 一次迭代总时间                                           |
| `timing/update`                      | 一次迭代中actor model进行update耗时                      |
| `timing/rollout`                     | 一次迭代中actor model进行rollout耗时                     |
| `timing/old_log_p`                   | 一次迭代中actor model计算log p耗时                       |
| `timing/reference_model`             | 一次迭代中reference model计算log p耗时                   |
| `timing/resharding_to_train`         | 权重转到训练mode耗时                                     |
| `timing/resharding_to_infer`         | 权重转到推理mode耗时                                     |
| `timing/adv`                         | 计算advantages耗时                                       |
| `timing/non_overlap_reference_model` | reference model计算log_p耗时的未被掩盖时间               |
| `timing/non_overlap_adv`             | advantages计算耗时的未被掩盖时间                         |
| `timing/ref_onload`                  | reference model计算logp过程中，onload耗时                |
| `timing/ref_offload`                 | reference model计算logp过程中，offload耗时               |

* 全共卡方案下总时间计算方式

`timing/all` >= `timing/rollout` +`timing/old_log_p` + `timing/update`  +  `timing/reference_model` + `timing/resharding_to_train` + `timing/resharding_to_infer`  + `timing/non_overlap_reference_model`


**其他指标**

| 指标                                    | 说明                                         |
| --------------------------------------- |--------------------------------------------|
| `actor/entropy`                         | 策略熵，表示策略的随机性或探索能力                          |
| `actor/kl_loss`                         | kl散度，衡量当前策略与参考策略（如旧策略或参考模型）之间的偏离程度         |
| `actor/pg_loss`                         | pg_loss，基于优势函数的策略梯度目标函数值，表示当前策略对提升奖励的学习能力。 |
| `actor/pg_clipfrac`                     | DAPO中裁剪机制生效的比例，反映了策略更新幅度的稳定性               |
| `actor/ppo_kl`                          | PPO算法的实际 KL 散度                             |
| `grad_norm`                             | 梯度范数，表示当前反向传播中参数梯度的整体幅度                    |
| `dapo/rewards/mean`                     | 规则奖励打分的平均总奖励值                              |
| `dapo/lr`                               | 学习率，优化器当前使用的学习率                            |
| `dapo/score/mean`                       | 开启奖励模型时的reward均值                           |
| `dapo/score/max`                        | 奖励模型及规则奖励对同一个样本的reward最大值                  |
| `dapo/score/min `                       | 奖励模型及规则奖励对同一个样本的reward最小值                  |
| `dapo/rewards/mean`                     | 规则奖励的reward均值；奖励模型对样本的reward经过归一化后的均值      |
| `dapo/rewards/max`                      | 规则奖励的reward最大值；奖励模型对样本的reward经过归一化后的最大值    |
| `dapo/rewards/min`                      | 规则奖励的reward最小值；奖励模型对样本的reward经过归一化后的最小值    |
| `response_length/mean`                  | 平均生成长度，模型生成回复（response）的平均 token 数         |
| `response_length/min`                   | 最短生成长度，当前 batch 中生成最短的 response 长度         |
| `response_length/max`                   | 最长生成长度，当前 batch 中生成最长的 response 长度         |
| `prompt_length/mean`                    | 平均输入长度，输入 prompt 的平均长度                     |
| `prompt_length/max`                     | 最长输入长度，当前 batch 中最长的 prompt长度              |
| `prompt_length/min`                     | 最短输入长度，当前 batch 中最长的 prompt长度              |
| `e2e_tps`                               | 端到端的tokens/p/s指标                           |
| `update_tps`                            | 训练的tokens/p/s指标                            |
| `vllm_tps`                              | 推理的tokens/p/s指标                            |

---
<a id="jump5"></a>
## 断点续训
进行断点续训时，需要注意配置以下参数并保证每个节点本地都保存对应的权重：
  ```yaml
actor_config:
    finetune: false       #<------- 断点续训时 finetune 参数设置为 false
    load: /path/qwen25vl_7b_ckpt      #<------- 断点续训时 load 路径应为之前保存的权重路径
    save: ./ckpt
    no_load_optim: false  #<------- 断点续训时 no_load_optim 应为 false
    no_load_rng: false    #<------- 断点续训时 no_load_rng 应为 false

rl_config:
    integrated_mode_config:
      ref_model_load_path: /path/qwen2_5_vl_7b_tp1pp1   #<------- 断点续训时，应在 ref_model_load_path 中配置原始模型megatron权重路径，供 reference model 加载
  ```

若需要加载指定迭代次数的权重、优化器等状态，需修改上述加载路径`load`中的`latest_checkpointed_iteration.txt`文件内容为指定迭代次数
```
qwen25vl_7b_ckpt
   ├── latest_checkpointed_iteration.txt
   ├── ...
```
备注：
（1）当前续训时，数据集迭代的续训依赖global_batch_size等参数，如果续训前后global_batch_size等参数不一致，续训会有精度差异；
（2）当前续训功能跳过已消耗数据的方式一定程度会增加数据集构建处理耗时；（耗时情况会依赖已消耗数据量）

---
<a id="jump6"></a>
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

