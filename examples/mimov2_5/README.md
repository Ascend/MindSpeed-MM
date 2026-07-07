# MiMo V2.5 使用指南

<p align="left">
</p>

## 目录

- [版本说明](#版本说明)
  - [参考实现](#参考实现)
  - [变更记录](#变更记录)
- [环境安装](#环境安装)
  - [环境准备](#1-环境准备)
  - [环境搭建](#2-环境搭建)
- [数据集准备及处理](#数据集准备及处理)
  - [数据集下载](#1-数据集下载以coco2017数据集为例)
  - [数据混合数据集处理](#2纯文本或有图无图混合训练数据-以llava-instruct-150k为例)
- [训练](#训练)
  - [准备工作](#1-准备工作)
  - [启动训练](#2-启动训练)
- [环境变量声明](#环境变量声明)
- [注意事项](#注意事项)

## 版本说明

### 参考实现

```shell
url=https://huggingface.co/XiaomiMiMo/MiMo-V2.5/tree/main
commit_id=13b5e3f
```

### 变更记录

2026.04.30: 首次支持MiMo V2.5模型

---
<a id="jump1"></a>

## 环境安装

<a id="jump1.1"></a>

### 1. 环境准备

【模型开发时推荐使用配套的环境版本】

请参考[安装指南](https://gitcode.com/Ascend/MindSpeed-MM/tree/master/docs/zh/pytorch/install_guide.md)，完成昇腾软件安装。

‼️ 部分特性依赖较新版本的CANN，请使用 8.5.0 以上版本:

- [CANN](https://www.hiascend.com/document/detail/zh/canncommercial/850/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=openEuler)

<a id="jump1.2"></a>

### 2. 环境搭建

拉取MindSpeed MM代码仓，并进入代码仓根目录：

```bash
git clone https://gitcode.com/Ascend/MindSpeed-MM.git
cd MindSpeed-MM
```

执行如下指令一键安装：

```bash
bash scripts/install.sh --msid eb10b92 && pip install transformers==5.2.0 accelerate==1.2.0
```

---

<a id="jump2"></a>

## 数据集准备及处理

<a id="jump2.1"></a>

### 1. 数据集下载（以COCO2017数据集为例）

(1) 用户需要自行下载COCO2017数据集[COCO2017](https://cocodataset.org/#download)，并解压到项目目录下的./data/COCO2017文件夹中。

(2) 获取图片数据集的描述文件（[LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/tree/main)），下载至./data/路径下。

> [!NOTE]
>
> 如无法顺利访问HuggingFace社区下载资源，推荐前往ModelScope下载，需关注待下载文件的正确性与安全性。

(3) 运行数据转换脚本`python mindspeed_mm/fsdp/tools/data_tool/llava_instruct_2_mllm_demo_format.py`，转换后参考数据目录结构如下：

   ```shell
   $playground
   ├── data
       ├── COCO2017
           ├── train2017

       ├── llava_instruct_150k.json
       ├── mllm_format_llava_instruct_data.json
       ...
   ```

---
当前支持读取多个以`,`（注意不要加空格）分隔的数据集，配置方式为将`mimov2_5_config.yaml`中的`DATASET_PATH`参数从`/data/mllm_format_llava_instruct_data.json`修改为`/data/mllm_format_llava_instruct_data.json,/data/mllm_format_llava_instruct_data2.json`

同时注意`mimov2_5_config.yaml`中`data->dataset_param->basic_parameters->max_samples`的配置，会限制数据只读取`max_samples`条，这样可以快速验证功能。正式训练时，可以把该参数去掉以读取全部的数据。

<a id="jump2.2"></a>

### 2.纯文本或有图无图混合训练数据 (以LLaVA-Instruct-150K为例)

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

### 1. 准备工作

配置脚本前需要完成前置准备工作，包括：**环境安装**、**数据集准备及处理**，详情可查看对应章节。当前只验证了模型减层单机的功能验证，需要下载模型的配置文件（权重文件可选），并在config.json中对模型层数按需修改，并同步修改启动脚本中需要的机器数量。

<a id="jump3.2"></a>

### 2. 启动训练

在 `mimov2_5_config.yaml` 文件中配置好数据集路径后，使用如下命令，即可实现MiMo V2.5的训练：

```shell
bash examples/mimov2_5/finetune_mimov2_5.sh
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
