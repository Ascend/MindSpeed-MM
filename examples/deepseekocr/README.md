# DeepseekOCR 使用指南

<p align="left">
</p>

## 目录

- [简介](#jump0)
- [环境安装](#jump1)
  - [仓库拉取](#jump1.1)
  - [环境搭建](#jump1.2)
- [权重下载](#jump2)
- [数据集准备及处理](#jump3)
- [训练](#jump4)
  - [准备工作](#jump4.1)
  - [启动训练](#jump4.2)
- [注意事项](#jump6)

<a id="jump0"></a>

## 简介
[DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) 是 DeepSeek 团队推出的视觉语言模型，专注于通过光学压缩技术高效处理长文本内容。模型由 DeepEncoder 编码器和 DeepSeek3B-MoE 解码器组成，能在保持高分辨率输入的同时，显著降低激活内存和视觉标记数量。模型在 10 倍压缩比下 OCR 精度可达 97%，在 20 倍压缩比下仍能保持 60% 的准确率。DeepSeek-OCR 支持多种分辨率模式，适用多语言文档处理，能解析图表、化学公式等复杂内容，为大规模文档处理提供高效解决方案。

#### 参考实现

```shell
url=https://github.com/deepseek-ai/DeepSeek-OCR/
commit_id=e4ac34e1e59b891163fb9325480fbedec865e1f0
```

#### 变更记录

2025.11.04: 打通 DeepSeekOCR 固定size图片训练流程。

<a id="jump1"></a>

## 环境安装

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](../../docs/user-guide/installation.md)

> 注意：Python版本推荐3.10，torch和torch_npu版本推荐2.7.1版本

<a id="jump1.1"></a>

#### 1. 仓库拉取

```shell
git clone --branch 2.3.0 https://gitcode.com/Ascend/MindSpeed-MM.git

cd MindSpeed-MM
```

<a id="jump1.2"></a>

#### 2. 环境搭建

```bash
# python3.10
conda create -n test python=3.10
conda activate test

# 安装torch和torch_npu
pip install torch-2.7.1-cp310-cp310-*.whl
pip install torch_npu-2.7.1*.manylinux2014_aarch64.whl

# 安装MindSPeed MM依赖
pip install -r examples/deepseekocr/requirements.txt

```

<a id="jump2"></a>

## 权重下载

从Huggingface等网站下载开源模型权重

- [Deepseek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)

<a id="jump3"></a>

## 数据集准备及处理

DeepSeekOCR未开源数据集，这里以[CC-OCR数据集](https://huggingface.co/datasets/wulipc/CC-OCR)为例
1、下载数据集，并放到./data/文件夹下
2、运行数据转换脚本python examples/deepseekocr/convert_ccocr_to_dsvlocr.py
预处理完后，数据格式如下：

   ```json
{
      'id': i,
      'conversations': [
          {
              "role": "<|User|>",
              "content": "Free OCR.",
              "images": [f"{save_file}"]
          },
          {
              "role": "<|Assistant|>",
              "content": answer
          }
      ]
}
    ... ...
   ```
数据路径参考如下：
```json
$playground
|--data
  |--CC-OCR
  |--convert
      |--*jpg
      ···
  |--output.jsonl
```
<a id="jump4"></a>

## 训练

<a id="jump4.1"></a>

#### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**权重下载**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump4.2"></a>

#### 2. 启动训练

1. 以图文理解的微调任务为例，可根据实际情况修改[启动脚本](../../examples/deepseekocr/finetune_ocr.sh)的配置，以下配置必须修改：

``` shell
DATA_PATH="./data/output.jsonl" # 数据集的文件
DATA_DIR="./data" # 数据集依赖图文等文件的目录
LOAD_PATH="./ckpt/deepseek-ai/DeepSeek-OCR" # huggingface下载的权重路径
```

2. 根据使用机器的情况，修改`NNODES`、`NPUS_PER_NODE`配置， 例如单机 A2 可设置`NNODES`为 1 、`NPUS_PER_NODE`为8；

3. 为保证代码安全，配置trust_remote_code默认为False，用户需要在启动脚本中使能`--trust-remote-code`，并且确保自己下载的模型和数据的安全性。

4. 上述注意点修改完毕后，可启动脚本开启训练：

```bash
bash examples/deepseekocr/finetune_ocr.sh
```

<a id="jump6"></a>

## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `ASCEND_SLOG_PRINT_TO_STDOUT` | 是否开启日志打印                                                           | `0`: 关闭日志打屏<br>`1`: 开启日志打屏                                                                   |
| `ASCEND_GLOBAL_LOG_LEVEL`     | 设置应用类日志的日志级别及各模块日志级别，仅支持调试日志                             | `0`: 对应DEBUG级别<br>`1`: 对应INFO级别<br>`2`: 对应WARNING级别<br>`3`: 对应ERROR级别<br>`4`: 对应NULL级别，不输出日志 |
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `COMBINED_ENABLE`             | 设置combined标志。设置为0表示关闭此功能；设置为1表示开启，用于优化非连续两个算子组合类场景 | `0`: 关闭<br>`1`: 开启                                                                           |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `HCCL_EXEC_TIMEOUT`           | 控制设备间执行时同步等待的时间，在该配置时间内各设备进程等待其他设备执行通信同步         | 需要配置为整数，取值范围`[68,17340]`，默认值为`1800`，单位`s`                                                    |
| `ACLNN_CACHE_LIMIT`           | 配置单算子执行API在Host侧缓存的算子信息条目个数                                  | 需要配置为整数，取值范围`[1, 10,000,000]`，默认值为`10000`                                                    |
| `TOKENIZERS_PARALLELISM`      | 用于控制Hugging Face的transformers库中的分词器（tokenizer）在多线程环境下的行为    | `False`: 禁用并行分词<br>`True`: 开启并行分词                                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |
| `NPU_ASD_ENABLE`   | 控制是否开启Ascend Extension for PyTorch的特征值检测功能 | 设置`0`或未设置: 关闭特征值检测<br>`1`: 表示开启特征值检测，只打印异常日志，不告警<br>`2`:开启特征值检测，并告警<br>`3`:开启特征值检测，并告警，同时会在device侧info级别日志中记录过程数据 |
| `ASCEND_LAUNCH_BLOCKING`   | 控制算子执行时是否启动同步模式 | `0`: 采用异步方式执行<br>`1`: 强制算子采用同步模式运行                                                               |
| `NPUS_PER_NODE`               | 配置一个计算节点上使用的NPU数量                                                  | 整数值（如 `1`, `8` 等）                                                                            |

