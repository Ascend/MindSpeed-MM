## MindSpore后端支持说明

MindSpeed-MM已支持接入华为自研AI套件MindSpore，旨在提供华为全栈易用的端到端的多模态模型训练解决方案，以此获得更极致的性能体验。MindSpore后端提供了一套对标PyTorch的API，用户无需进行额外代码适配即可无缝切换。

---

## 🔥🔥🔥Latest News

🚀🚀🚀 MindSpore后端已支持[CogVideoX](../../examples/mindspore/cogvideox/)、[InternVL2](../../examples/mindspore/internvl2)、[Qwen2.5VL](../../examples/mindspore/qwen2.5vl/README.md)、[Qwen25Omni](../../examples/mindspore/qwen2.5omni/README.md)、[Qwen3VL](../../examples/mindspore/qwen3vl/README.md)模型。

## 版本配套

MindSpeed-MM + MindSpore后端的依赖配套如下表，安装步骤参考[基础安装指导](./install_guide.md)。

| 依赖软件        |                                                                                                                                    |
| --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 昇腾NPU驱动固件 | 在研版本 |
| 昇腾 CANN       | 在研版本                                               |
| MindSpore       | [2.7.2](https://www.mindspore.cn/install/)                                                                                        |
| Python          | >=3.9  

## 环境部署

具体部署步骤请查看[部署文档](./install_guide.md)。

## 快速上手

快速上手操作请查看[快速上手文档](./getting_start.md)。

---

## 模型/特性介绍

### 已支持特性概览

|       模型 \ 特性       | [TP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/tensor-parallel.md) | [TP-SP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/sequence-parallel.md) | [VPP](docs/features/virtual_pipeline_parallel.md) | [PP](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/pipeline-parallel.md) | CP | EP |  [Distributed Optimizer](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/distributed-optimizer.md) | [Recomputation](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/recomputation.md) | [LoRA](./docs/features/lora_finetune.md) |
|:-------------------:|:------:|:------:|:------:|:---------------------------------------------------------------------------------------:|:------:|:-------:|:------:|:------:|:------:|
|   CogVideoX系列-T2V   | ✔ | ✔ |  |                                                                                         | CP (Ulysses) |   | ✔ | ✔ |  |
|   CogVideoX系列-I2V   | ✔ | ✔ |  |                                                                                         | CP (Ulysses) |   | ✔ | ✔ |  |
|    InternVL2-2B     |  |  |  |                                            ✔                                            |  |   | ✔ | ✔ |  |
|    InternVL2-8B     |  |  | ✔ |                                            ✔                                            |  |   | ✔ | ✔ |  |
|    Qwen2.5VL-7B     | ✔ |  |  |                                            ✔                                            |  |   | ✔ |  |  |
|    Qwen2.5VL-72B    | ✔ |  |  |                                            ✔                                            |  |   | ✔ |  |  |
|    Qwen25Omni-7B    |  |  |  |                                            ✔                                            |  |   | ✔ |  |  |
|    Qwen3vl-8B    |  | ✔ |  |                                            ✔                                            |  |   | ✔ |  |  |
|    Qwen3vl-30B    |  | ✔ |  |                                            ✔                                            |  | ✔ | ✔ |  |

备注：

* TP: [Tensor Parallel](https://arxiv.org/abs/1909.08053)
* TP-SP: [Tensor Parallel with Sequence Parallel](https://arxiv.org/abs/2205.05198)
* VPP: [Virtual Pipeline Parallel](https://arxiv.org/abs/2104.04473)
* PP: [Pipeline Parallel](https://arxiv.org/abs/2104.04473)
* DSP: [Dynamic Sequence Parallel](https://arxiv.org/abs/2403.10266)
* CP (Ulysses): [Context Parallel](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html) by leveraging [Deepspeed Ulysses](https://arxiv.org/abs/2309.14509) with Sequence Parallel
* CP (Ring Attention): Context Parallel with [Ring Attention](https://arxiv.org/abs/2310.01889)
* Distributed Optimizer: [Zero Redundancy Optimizer](https://arxiv.org/abs/1910.02054) (ZeRO)
* Recomputation: Reducing Activation [Recomputation](https://arxiv.org/abs/2205.05198)
* LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### 支持模型

<table>
  <a id="jump1"></a>
  <caption>MindSpeed MM (MindSpore后端)模型支持列表</caption>
  <thead>
    <tr>
      <th>模型任务</th>
      <th>模型</th>
      <th>参数量</th>
      <th>任务</th>
      <th>集群</th>
      <th>精度格式</th>
      <th>NPU性能</th>
      <th>参考性能</th>
      <th>平均序列长度</th>
      <th>支持情况</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10"> 多模态生成 </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 2.09 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 3.03 (SPS) </td>
      <td> / </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 2.01 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 3.92 (SPS) </td>
      <td> / </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/mindspore/wan2.1">Wan2.1-I2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.61 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/master/examples/mindspore/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> 预训练 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.73 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> Lora微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="20"> 多模态理解 </td>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/internvl2">InternVL 2.0</a></td>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-2B">2B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 22.46 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-8B">8B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 11.00 (SPS) </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-26B">26B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 3.26 (SPS) </td>
      <td> / </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/OpenGVLab/InternVL2-Llama3-76B">76B</a></td>
      <td> 全参微调 </td>
      <td> 8x16 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 191 (TPS) </td>
      <td> / </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 21.79 (SPS) </td>
      <td> 563 </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 12.67 (SPS) </td>
      <td> 563 </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> 微调 </td>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> 563 </td>
      <td>支持中</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> 微调 </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 256.28 (TPS) </td>
      <td> 563 </td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/qwen2.5omni">Qwen25Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/main">7B</a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/qwen3vl">Qwen3VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main"> 8B </a></td>
      <td> 微调 </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main"> 30B </a></td>
      <td> 微调 </td>
      <td> 1x16 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
    </tr>
    </tbody>
</table>

---

## 特性规划

## 工具使用

<a id="jump2.1"></a>

## 昇腾Profiling采集工具

MindSpeed MM集成了昇腾profiling采集工具，以提供对模型运行情况的分析。该工具能够依照配置采集模型的算子、显存等关键信息，同时支持动静态两种采集方式，协助开发者分析模型瓶颈，并可根据实际场景需求选择使用。

具体方法见 [README](../../mindspeed_mm/tools/README.md) 的profiling章节。

## MindStudio Insight性能分析工具

针对大模型集群场景的性能调优，这里推荐一款优秀的可视化调优工具MindStudio Insight。
MindStudio Insight提供了包括Timeline视图、通信分析、计算耗时等的可视化呈现，以便用户分析潜在的性能瓶颈，并指导如何采取措施消除或减少这些瓶颈。

具体使用方法见[《MindStudio Insight操作指南》](https://www.hiascend.com/document/detail/zh/mindstudio/70RC3/msinsightug/msascendinsightug/Insight_userguide_0002.html)，下载地址[《MindStudio Insight》](https://support.huawei.com/enterprise/zh/ascend-computing/mindstudio-pid-251913966/software/262029358?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251913966)。
