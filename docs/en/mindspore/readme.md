# Using MindSpore asd Backend

MindSpeed MM now supports integration with Huawei's self-developed AI framework MindSpore, aiming to provide an easy-to-use, end-to-end multimodal model training solution across Huawei's full stack, thereby achieving ultimate performance. The MindSpore backend offers a set of APIs aligned with PyTorch, facilitating a seamless switchover without additional code adaptation.

## 🔥🔥🔥Latest News

🚀🚀🚀 The MindSpore backend now supports [CogVideoX](../../../examples/mindspore/cogvideox), [Qwen2.5VL](../../../examples/mindspore/qwen2.5vl/README.md), [Qwen25Omni](../../../examples/mindspore/qwen2.5omni/README.md), and [Qwen3VL](../../../examples/mindspore/qwen3vl/README.md) models.

## Version Compatibility

The dependency compatibility for the MindSpeed MM + MindSpore backend is shown in the table below. For installation steps, refer to the [Installation Guide](./install_guide.md).

| Dependency                |   Version                                                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Ascend NPU driver & firmware | In-development |
| Ascend CANN                 | In-development                                               |
| MindSpore                   | [2.7.2](https://www.mindspore.cn/install/)                                                                                        |
| Python                      | >=3.10

## Environment Deployment

For specific deployment steps, please refer to the [Installation Guide](./install_guide.md).

## Quick Start

For quick start operations, please refer to the [Quick Start](./getting_start.md).

## Model/Feature Introduction

### Supported Features

|     Model\Feature    | [TP](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/tensor-parallel.md) | [TP-SP](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/sequence-parallel.md) | [VPP](../features/virtual_pipeline_parallel.md) | [PP](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/pipeline-parallel.md) | CP | EP |  [Distributed Optimizer](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/distributed-optimizer.md) | [Recomputation](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/recomputation.md) | [LoRA](../features/lora_finetune.md) |
|:---------------:|:------:|:------:|:------:|:---------------------------------------------------------------------------------------:|:------:|:-------:|:------:|:------:|:------:|
| CogVideoX Series-T2V | ✔ | ✔ |  |                                                                                         | CP (Ulysses) |   | ✔ | ✔ |  |
| CogVideoX Series-I2V | ✔ | ✔ |  |                                                                                         | CP (Ulysses) |   | ✔ | ✔ |  |
|  Qwen2.5VL-7B   | ✔ |  |  |                                            ✔                                            |  |   | ✔ |  |  |
|  Qwen2.5VL-72B  | ✔ |  |  |                                            ✔                                            |  |   | ✔ |  |  |
| Qwen2.5Omni-7B  |  |  |  |                                            ✔                                            |  |   | ✔ |  |  |
|   Qwen3VL-8B    |  | ✔ |  |                                            ✔                                            |  |   | ✔ |  |  |
|   Qwen3VL-30B   |  | ✔ |  |                                            ✔                                            |  | ✔ | ✔ |  |  |

Note:

* TP: [Tensor Parallelism](https://arxiv.org/abs/1909.08053)
* TP-SP: [Tensor Parallelism with Sequence Parallelism](https://arxiv.org/abs/2205.05198)
* VPP: [Virtual Pipeline Parallel](https://arxiv.org/abs/2104.04473)
* PP: [Pipeline Parallelism](https://arxiv.org/abs/2104.04473)
* DSP: [Dynamic Sequence Parallelism](https://arxiv.org/abs/2403.10266)
* CP (Ulysses): [Context Parallelism](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/context_parallel.html) by leveraging [DeepSpeed Ulysses](https://arxiv.org/abs/2309.14509) with Sequence Parallel
* CP (Ring Attention): Context Parallel with [Ring Attention](https://arxiv.org/abs/2310.01889)
* Distributed Optimizer: [Zero Redundancy Optimizer](https://arxiv.org/abs/1910.02054) (ZeRO)
* Recomputation: [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)
* LoRA: [Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)

### Supported Models

<table>
  <a id="jump1"></a>
  <caption>Models Supported by MindSpeed MM +  MindSpore Backend</caption>
  <thead>
    <tr>
      <th>Model Task</th>
      <th>Model</th>
      <th>Parameter Scale</th>
      <th>Task</th>
      <th>Cluster</th>
      <th>Precision Format</th>
      <th>NPU Performance</th>
      <th>Reference Performance</th>
      <th>Average Sequence Length</th>
      <th>Support Status</th>
      <th>Supported Version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="10"> Multimodal generation </td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX-5b"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.46 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
  <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX 1.5-T2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 2.09 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> LoRA Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 3.03 (SPS) </td>
      <td> / </td>
      <td>In Development</td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/cogvideox">CogVideoX 1.5-I2V</a></td>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 2.01 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT"> 5B </a></td>
      <td> LoRA Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 3.92 (SPS) </td>
      <td> / </td>
      <td>In Development</td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/wan2.1">Wan2.1-I2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.61 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/wan2.1">Wan2.1-T2V</a></td>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> Pre-training </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 0.73 (SPS) </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers"> 1.3B </a></td>
      <td> LoRA Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>In Development</td>
    </tr>
    <tr>
      <td rowspan="7"> Multimodal understanding </td>
      <td rowspan="4"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/examples/qwen2.5vl">Qwen2.5-VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct">3B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 21.79 (SPS) </td>
      <td> 563 </td>
      <td>In Development</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 12.67 (SPS) </td>
      <td> 563 </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct">32B</a></td>
      <td> Fine-tuning </td>
      <td> 2x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> 563 </td>
      <td>In Development</td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">72B</a></td>
      <td> Fine-tuning </td>
      <td> 8x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> 256.28 (TPS) </td>
      <td> 563 </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td rowspan="1"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/qwen2.5omni">Qwen25Omni</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B/tree/main">7B</a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td rowspan="2"><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/master/examples/mindspore/qwen3vl">Qwen3VL</a></td>
      <td><a href="https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/main"> 8B </a></td>
      <td> Fine-tuning </td>
      <td> 1x8 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    <tr>
      <td><a href="https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/tree/main"> 30B </a></td>
      <td> Fine-tuning </td>
      <td> 1x16 </td>
      <td> BF16 </td>
      <td>  </td>
      <td> / </td>
      <td> / </td>
      <td>✅</td>
      <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/tree/2.3.0"> 2.3.0 </a></td>
    </tr>
    </tbody>
</table>

## Feature Planning

## Tool Usage

<a id="jump2.1"></a>

## Ascend Profiling Tool

MindSpeed MM integrates the Ascend profiling tool to provide analysis of model execution. This tool can profile key information such as model operators and memory usage according to the configuration, and supports both dynamic and static profiling methods, assisting developers in analyzing model bottlenecks and selecting the appropriate method based on actual scenario requirements.

For details, see the "profiling" section in [README](../tools.md).

## MindStudio Insight

For performance tuning in large model cluster scenarios, an excellent visualization tuning tool, [MindStudio Insight](https://support.huawei.com/enterprise/en/ascend-computing/mindstudio-pid-251913966/software/262029358?idAbsPath=fixnode01%7C23710424%7C251366513%7C22892968%7C251913966), is recommended here.
MindStudio Insight provides visual presentations including timeline view, communication analysis, and computation time, enabling users to analyze potential performance bottlenecks and guide them on how to take measures to eliminate or reduce these bottlenecks.

For specific usage, see [*MindStudio Insight User Guide*](https://www.hiascend.com/document/detail/en/mindstudio/70RC3/msinsightug/msascendinsightug/Insight_userguide_0002.html).
