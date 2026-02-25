# Self-Forcing

<p align="left">
</p>

- [self-forcing](#jump1)
  - [模型介绍](#模型介绍)
  - [版本说明](#版本说明)
    - [参考实现](#参考实现)
    - [变更记录](#变更记录)
  - [环境搭建](#环境搭建)
  - [训练](#训练)
  - [推理](#推理)
  - [环境变量声明](#环境变量声明)
- [引用](#引用)
  - [公网地址说明](#公网地址说明)

## 模型介绍

​Self Forcing的核心技术路线是在训练自回归视频扩散模型时，摒弃传统的“教师强制”方法，转而让模型基于自身之前生成的、带有噪声的帧来预测下一帧，从而模拟真实的推理生成过程，实现完整的自回归自我展开训练。其核心创新点在于从根本上解决了“暴露偏差”问题，通过迫使模型学习如何从其自身生成的不完美上下文中进行修正，并引入整体分布匹配损失来优化整个生成序列的全局质量，从而显著提升了生成的一致性和稳定性。主要用途是实现高质量、高帧率且时间一致性强的实时视频生成

## 版本说明

### 参考实现

  ```shell
  url=https://github.com/guandeh17/Self-Forcing
  commit_id=33593df3e81fa3ec10239271dd2c100facac6de1
  ```

### 变更记录

2025.11.12：首次发布Self-Forcing

## 环境搭建

【模型开发时推荐使用配套的环境版本】

昇腾基础软件安装请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/pytorch/installation.md)

> Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

1. 在工作目录执行下列命令

    ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-MM.git
    cd MindSpeed-MM
    git clone https://github.com/guandeh17/Self-Forcing.git
    bash examples/self_forcing/replace_npu_patch.sh
    cd Self-Forcing
    ```

2. 安装依赖

    ```shell
    pip install -r requirements.txt
    ```

3. 下载权重

    ```
    huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
    huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
    huggingface-cli download gdhe17/Self-Forcing checkpoints/ode_init.pt --local-dir .
    huggingface-cli download gdhe17/Self-Forcing vidprom_filtered_extended.txt --local-dir prompts
    ```

## 训练

1. 执行以下命令即可开启训练

    ```shell
   torchrun --nnodes=1 --nproc_per_node=8 \
      train.py \
      --config_path configs/self_forcing_dmd.yaml \
      --logdir logs/self_forcing_dmd \
     --disable-wandb
   ```

## 推理

  1. 执行以下命令即可开启推理

      ```shell
      python inference.py \
          --config_path configs/self_forcing_dmd.yaml \
          --output_folder videos/self_forcing_dmd \
          --checkpoint_path checkpoints/self_forcing_dmd.pt \
          --data_path prompts/MovieGenVideoBench_extended.txt \
          --use_ema
      ```

## 环境变量声明

| 环境变量                          | 描述                                                                 | 取值说明                                                                                                               |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                                |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志   |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                                           |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                                          |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                             |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                            |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征 |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                        |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                     |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                        |
| `OMP_NUM_THREADS`             | 设置执行期间使用的线程数    |      需要配置为整数                                                  |


# 引用

## 公网地址说明

代码涉及公网地址参考 [公网地址](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/zh/public_address_statement.md)