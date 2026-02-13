# Kimi-K2.5 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#jump1)
  - [环境准备](#jump1.1)
  - [环境搭建](#jump1.2)
- [数据集准备及处理](#jump2)
  - [数据集下载](#jump2.1)
  - [数据混合数据集处理](#jump2.2)
- [训练](#jump3)
  - [准备工作](#jump3.1)
  - [启动训练](#jump3.2)
- [环境变量声明](#jump4)
- [注意事项](#jump5)

## 版本说明
#### 参考实现
```
url=https://huggingface.co/moonshotai/Kimi-K2.5/tree/main
commit_id=3367c8d
```

#### 变更记录

2026.02.13: 首次支持Kimi-K2.5模型

---
<a id="jump1"></a>
## 环境安装

<a id="jump1.1"></a>
#### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/installation.md)，完成昇腾软件安装。

<a id="jump1.2"></a>
#### 2. 环境搭建
```bash
git clone --branch fsdp2_dev https://gitcode.com/Ascend/MindSpeed-MM.git

# 安装mindspeed及依赖
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
cp -r mindspeed ../MindSpeed-MM/

# 安装mindspeed mm及依赖
cd ../MindSpeed-MM
pip install -e .

# 安装三方库依赖
pip install tiktoken==0.12.0
```

---

<a id="jump2"></a>

## 数据集准备及处理

<a id="jump2.1"></a>

#### 1. 数据集下载（以COCO2017数据集为例）

(1) 用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2) 获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

(3) 运行数据转换脚本`python examples/qwen2vl/llava_instruct_2_mllm_demo_format.py`，转换后参考数据目录结构如下：

   ```
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为将`kimik2_5_config.yaml`中的`DATASET_PATH`参数从`/data/mllm_format_llava_instruct_data.json`修改为`/data/mllm_format_llava_instruct_data.json,/data/mllm_format_llava_instruct_data2.json`

同时注意`kimik2_5_config.yaml`中`data->dataset_param->basic_parameters->max_samples`的配置，会限制数据只读`max_samples`条，这样可以快速验证功能。正式训练时，可以把该参数去掉以读取全部的数据。

<a id="jump2.2"></a>
#### 2.纯文本或有图无图混合训练数据 (以LLaVA-Instruct-150K为例)

现在本框架已经支持纯文本/混合数据（有图像和无图像数据混合训练）。

在数据构造时，对于包含图片的数据，需要保留`image`这个键值。

```python
{
  "id": your_id,
  "image": your_image_path,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```

在数据构造时，对于纯文本数据，可以去除`image`这个键值。

```python
{
  "id": your_id,
  "conversations": [
      {"from": "human", "value": your_query},
      {"from": "gpt", "value": your_response},
  ],
}
```
---

<a id="jump3"></a>
## 训练

<a id="jump3.1"></a>
#### 1. 准备工作

从Huggingface库 （[Kimi-K2.5](https://huggingface.co/moonshotai/Kimi-K2.5/tree/main)） 下载`tiktoken.model`文件并放置于本地`mindspeed_mm/fsdp/models/kimik25/`路径下；

Kimi-K2.5模型需要配置多机训练，如需拉起多机训练，请修改启动脚本下的 `MASTER_ADDR`、`NNODES` 以及 `NODE_RANK` 变量：
``` shell
MASTER_ADDR: 主节点IP地址
NNODES: 总节点数量
NODE_RANK: 当前节点序号
```

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节。

<a id="jump3.2"></a>
#### 2. 启动训练

在 `kimik2_5_config.yaml` 文件中配置好数据集路径后，使用如下命令，即可实现Kimi-K2.5的训练：

```shell
bash examples/fsdp2/kimik2_5/finetune_kimik2_5.sh
```

<a id="jump4"></a>
## 环境变量声明

| 环境变量                      | 描述                                                                 | 取值说明                                                                                         |
|-------------------------------|--------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| `TASK_QUEUE_ENABLE`           | 用于控制开启task_queue算子下发队列优化的等级                                    | `0`: 关闭<br>`1`: 开启Level 1优化<br>`2`: 开启Level 2优化                                              |
| `CPU_AFFINITY_CONF`           | 控制CPU端算子任务的处理器亲和性，即设定任务绑核                                    | 设置`0`或未设置: 表示不启用绑核功能<br>`1`: 表示开启粗粒度绑核<br>`2`: 表示开启细粒度绑核                                     |
| `HCCL_CONNECT_TIMEOUT`        | 用于限制不同设备之间socket建链过程的超时等待时间                                  | 需要配置为整数，取值范围`[120,7200]`，默认值为`120`，单位`s`                                                     |
| `PYTORCH_NPU_ALLOC_CONF`      | 控制缓存分配器行为                                                          | `expandable_segments:<value>`: 使能内存池扩展段功能，即虚拟内存特征                                            |
| `MULTI_STREAM_MEMORY_REUSE`   | 配置多流内存复用是否开启 | `0`: 关闭多流内存复用<br>`1`: 开启多流内存复用                                                               |

---

<a id="jump5"></a>
## 注意事项