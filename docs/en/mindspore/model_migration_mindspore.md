# Migration Development Guide (MindSpore as Backend)

## Overview

Large-model applications are currently evolving at a rapid pace. The PyTorch AI suite has gained widespread adoption in the industry for large-model training and inference, largely due to its developer-friendly programming paradigm. Huawei Ascend also provides a PyTorch-based [Ascend MindSpeed + Ascend NPU Training Solution](https://www.hiascend.com/software/mindspeed). To this end, MindSpore has introduced a dynamic graph execution mode and corresponding [dynamic graph APIs](https://www.mindspore.cn/docs/en/master/api_python/mindspore.mint.html) enabling you to leverage the MindSpore AI suite with an experience similar to PyTorch. Huawei Ascend MindSpeed has now also been extended to support MindSpore as a backend engine, delivering a full-stack Huawei solution. This allows you to benefit from both user-friendly programming and the unparalleled performance enabled by Huawei's integrated software-hardware stack.

This guide provides migration development guidance for MindSpeed MM with MindSpore as backend, helping you rapidly migrate large-model training from the PyTorch backend to the MindSpore backend. Before diving into the migration process, we briefly introduce MindSpore's dynamic graph mode and the API adaptation tool MSAdapter. This helps you understand the differences between the MindSpore and PyTorch backends, and provides insights for troubleshooting issues encountered during model migration.

>[!NOTICE]
>
>It is recommended that you first refer to the *[MindSpeed MM Migration and Tuning Guide](https://gitcode.com/Ascend/MindSpeed-MM/tree/2.2.0/docs/user-guide/model-migration.md)* for code development based on the torch ecosystem, and then migrate to the MindSpore backend for operations according to this guide to achieve better model training and inference performance.

### Introduction to MindSpore Dynamic Graph

[MindSpore Dynamic Graph mode](https://www.mindspore.cn/tutorials/en/r2.7.0rc1/beginner/accelerate_with_static_graph.html) is also known as PyNative mode. Compared to the small-operator concatenation approach of previous versions, the current version adopts a pybind-based direct operator invocation method, meaning forward operator execution directly calls the underlying operator interfaces. This significantly reduces the per-operator execution overhead and data structure conversion overhead, resulting in substantial performance improvements. The MindSpore dynamic graph mode is still implemented based on MindSpore's fundamental mechanisms. Therefore, certain mechanism-level differences remain between MindSpore and PyTorch dynamic graphs, as briefly elaborated below.

+ Differences in Automatic Differentiation Mechanisms

Neural network training relies on the backpropagation algorithm. Automatic differentiation is the core mechanism that enables backpropagation implementation across AI suites. PyTorch uses a dynamic computation graph that executes operations immediately. The forward and backward computation graphs are dynamically constructed during each forward pass. PyTorch's reverse differentiation is imperative, aligning with object-oriented programming conventions.

MindSpore adopts a [functional automatic differentiation](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/beginner/autograd.html?highlight=%E4%BC%A0%E6%92%AD) design philosophy, providing `grad` and `value_and_grad` interfaces that are closer to mathematical semantics. This differs from PyTorch's `Tensor.backward` mechanism. MindSpore requires calling the `grad` interface on the function object requiring automatic differentiation and specifying the positional indices of the inputs for which gradients are needed. For detailed usage of the `grad` and `value_and_grad` interfaces, see [mindspore.grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.grad.html) and [mindspore.value_and_grad](https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.value_and_grad.html).

+ Custom Operators

Similar to PyTorch, the MindSpore's dynamic graph mode also supports custom operator integration. You can refer to [CustomOpBuilder-Based Custom Operators](https://www.mindspore.cn/tutorials/en/r2.7.0/custom_program/operation/op_customopbuilder.html) to learn how to integrate custom operators.

+ Differences in Dynamic Graph APIs

Although MindSpore's dynamic graph APIs aim to be consistent with PyTorch APIs, due to framework mechanism differences and other factors, some MindSpore dynamic graph APIs may exhibit discrepancies from PyTorch APIs in terms of parameters, inputs, outputs, logical functionality, and specific scenarios. For detailed differences, please refer to the [PyTorch and MindSpore API Mapping Table](https://www.mindspore.cn/docs/en/master/note/api_mapping/pytorch_api_mapping.html).

+ MSAdapter Introduction

[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter.git) is an ecosystem adaptation tool for MindSpore designed to seamlessly migrate third-party framework code (e.g., PyTorch, JAX) to the MindSpore ecosystem without altering your existing coding habits, thereby helping you efficiently leverage Ascend computing power.

## Software Installation

To help you understand and select the appropriate MindSpeed version, we provide a detailed version compatibility table.
This table lists mappings between the MindSpeed version and the corresponding MindSpore and CANN versions, ensuring that you can accurately select the compatible version based on your own software environment to achieve optimal performance and functional support.

<table border="0">
  <tr>
    <td> MindSpeed Version </td>
    <td> 2.3.0_core_r0.12.1 </td>
  </tr>
  <tr>
    <td> Megatron-LM Version </td>
    <td> core_r0.12.1 </td>
  </tr>
  <tr>
    <td> CANN Version </td>
    <td> CANN 8.5.0 </td>
  </tr>
  <tr>
    <td> MindSpore Version </td>
    <td> 2.7.2 </td>
  </tr>
  <tr>
    <td> MSAdapter Version </td>
    <td> r0.4.0 </td>
  </tr>
  <tr>
    <td> Python Version </td>
    <td> Python3.10.x </td>
  </tr>
</table>

### Installation Procedure

- Required software

<table border="0">
  <tr>
    <th>Software</th>
    <th>Version</th>
  </tr>
  <tr>
    <td> Ascend NPU driver </td>
    <td rowspan="5">See <a href="https://www.hiascend.com/document/detail/en/canncommercial/850/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=netyum&OS=openEuler">*CANN Software Installation Guide*</a>.</td>
  </tr>
  <tr>
    <td> Ascend NPU firmware </td>
  </tr>
  <tr>
    <td> Toolkit </td>
  </tr>
  <tr>
    <td> Kernels </td>
  </tr>
  <tr>
    <td> NNAL (Ascend Transformer Boost Acceleration Library) </td>
  </tr>
  <tr>
    <td> MindSpore Framework </td>
    <td> See <a href="https://www.mindspore.cn/install/en">*MindSpore Installation Guide*</a>.</td>
  </tr>
  <tr>
    <td> MSAdapter plugin </td>
    <td> See the <a href="https://mindtorch.readthedocs.io/zh-cn/latest/docs/Install.html">*MSAdapter Installation Guide*</a>.</td>
  </tr>
</table>

- Download the master branch of the MindSpeed-Core-MS source code and perform one-click adaptation.

  ```shell
    git clone https://gitcode.com/Ascend/MindSpeed-Core-MS.git -b r0.5.0
    cd MindSpeed-Core-MS
    source auto_convert.sh mm
  ```

  The MindSpeed-Core-MS source code supports one-click adaptation, eliminating the need to manually pull source code from repositories such as MindSpeed. In `auto_convert_xxx.sh`, `xxx` represents the usage scenario, which can be `llm` (large language model scenario), `mm` (multimodal model scenario), or `rl` (reinforcement learning scenario). For specific usage, see the [README](https://gitcode.com/Ascend/MindSpeed-Core-MS).

## Adaptation to MindSpore

For models supported by the PyTorch backend but not by the MindSpore backend, you only need to make minor adaptation modifications to the model shell scripts provided for the PyTorch backend to migrate to the MindSpore backend. The main principles for shell script adaptation are as follows:

- Modify the launch command to `msrun` and add the `--ai-framework mindspore` parameter.
  The following is an example of launch command adaptation.

  Before adaptation (PyTorch backend):

  ```shell
    torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        2>&1 | tee logs/train_${logfile}.log
  ```

  After adaptation (MindSpore backend):

  ```shell
    msrun $DISTRIBUTED_ARGS pretrain_vlm.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        --ai-framework mindspore \
        2>&1 | tee logs/train_${logfile}.log
  ```

- `DISTRIBUTED_ARGS`adaptation:
  The distributed parameters used by the `torchrun` command include `--nproc_per_node`, `--nnodes`, `--node_rank`, `--master_addr`, and `--master_port`. After switching to `msrun`, the distributed parameter names need to be adapted. The table below describes distributed parameters used by `msrun` and `torchrun`. For the meaning and usage of each parameter, see [msrun Launching](https://www.mindspore.cn/tutorials/en/r2.7.0/parallel/msrun_launcher.html).

  | msrun Distributed Parameter | Relationship with torchrun Distributed Parameter | Parameter Description |
  | ------------------ | -------------------------- | ------- |
  | --local_worker_num | nproc_per_node        | Total number of worker processes participating in the distributed task |
  | --worker_num       | nproc_per_node*nnodes | Number of worker processes launched on the current node |
  | --master_addr      | master_addr           | IP address or hostname of the scheduler |
  | --master_port      | master_port           | Port bound by the scheduler |
  | --node_rank        | node_rank             | Index of the current node |
  | --log_dir          | /                          | Log output path for worker and scheduler |
  | --join             | /                          | Whether msrun waits for worker and scheduler to exit |
  | --cluster_time_out | /                          | Cluster networking timeout in seconds |
  | --bind_core        | /                          | Enable process core binding |

  The following is an example of `DISTRIBUTED_ARGS` adaptation.

  Before adaptation (PyTorch backend):

  ```shell
  # PyTorch Backend Startup Script
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

  DISTRIBUTED_ARGS="
      --nproc_per_node $NPUS_PER_NODE \
      --nnodes $NNODES \
      --node_rank $NODE_RANK \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT
  "
  ```

  After adaptation (MindSpore backend):

  ```shell
  # MindSpore backend startup script
  NPUS_PER_NODE=8
  MASTER_ADDR=localhost
  MASTER_PORT=6000
  NNODES=1
  NODE_RANK=0
  WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
  export LOCAL_WORLD_SIZE=8

  DISTRIBUTED_ARGS="
      --local_worker_num $NPUS_PER_NODE \
      --worker_num $WORLD_SIZE \
      --node_rank $NODE_RANK \
      --master_addr $MASTER_ADDR \
      --master_port $MASTER_PORT \
      --log_dir msrun_log \
      --bind_core=True
  "

- Verify that the features corresponding to the feature flags in the PyTorch shell script are supported by the MindSpore backend. If a feature is not supported by the MindSpore backend, we recommend disabling it in the launch script.

After completing the adaptation of the startup shell script described above, you can use the script to launch the model task.
