
# MindSpeed-MM MindSpore后端迁移开发指南

## 0 概述

当前大模型相关业务发展迅速，AI框架PyTorch因其编程友好受到业界大多数大模型训练、推理软件的青睐，华为昇腾也提供了基于PyTorch的[昇腾MindSpeed + 昇腾NPU训练解决方案](https://www.hiascend.com/software/mindspeed)。为此，MindSpore推出了动态图方案以及[动态图API接口](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.mint.html)，使用户也可以像使用PyTorch一样使用MindSpore AI框架。当前华为昇腾MindSpeed也已支持接入MindSpore AI框架作为后端引擎，打造华为全栈解决方案，使用户在友好编程的同时，也享受到华为全栈软硬结合带来的极致性能体验。

**建议用户先参照《[MindSpeed MM迁移调优指南](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/docs/user-guide/model-migration.md)》进行基于torch生态的代码开发，之后根据本指南迁移至Mindspore后端运行来获取更优的模型训练推理性能。**

本指南侧重提供MindSpeed-MM MindSpore后端的迁移开发指导，帮助用户快速地将大模型训练从PyTorch后端迁移至MindSpore后端。在介绍迁移开发前，先简要介绍MindSpore动态图和API适配工具MSAdapter，供用户了解MindSpore后端和PyTorch后端的差异，以启发用户在模型迁移开发遇到问题时进行问题排查。

### MindSpore动态图介绍

[MindSpore 动态图模式](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0rc1/beginner/accelerate_with_static_graph.html?highlight=%E5%8A%A8%E6%80%81)又称PyNative模式。相比之前版本的小算子拼接方案，当前版本采用了pybind算子直调的方式，即正向算子执行直接调用底层算子接口，极大地减少了单算子执行的流程开销和数据结构转换开销，在性能上有较大提升。MindSpore动态图模式仍然是基于MindSpore的基本机制实现，因此，其与PyTorch动态图仍然存在部分机制上的差异，以下进行简要阐述。

#### 自动微分机制差异

神经网络的训练主要使用反向传播算法，自动微分是各个AI框架实现反向传播的核心机制。PyTorch使用动态计算图，在代码执行时立即运算，正反向计算图在每次前向传播时动态构建；PyTorch反向微分是命令式反向微分，符合面向对象编程的使用习惯。

MindSpore使用[函数式自动微分](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/beginner/autograd.html?highlight=%E4%BC%A0%E6%92%AD)的设计理念，提供了更接近数学语义的自动微分接口`grad`和`value_and_grad`. 与PyTorch的自动微分`Tensor.backward`机制不同，MindSpore需要针对需要自动微分的函数对象调用`grad`接口获取函数微分，并指定需要求导的输入的位置索引。`grad`和`value_and_grad`接口的使用详见 [mindspore.grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.grad.html?highlight=grad#mindspore.grad) 和 [mindspore.value_and_grad](https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.value_and_grad.html).

#### 自定义算子

与PyTorch类似的，MindSpore动态图模式也支持了自定义算子接入，用户可以参考[基于CustomOpBuilder的自定义算子](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/custom_program/operation/op_customopbuilder.html)了解如何进行自定义算子接入。

#### 动态图API接口差异

尽管MindSpore动态图API接口的目标是与PyTorch API保持一致，但由于框架机制等原因，部分MindSpore动态图API接口可能在参数、输入、输出、逻辑功能和特定场景等方面与PyTorch APIs存在一定差异，具体差异情况详见[PyTorch与MindSpore API映射表](https://www.mindspore.cn/docs/zh-CN/master/note/api_mapping/pytorch_api_mapping.html)。

### MSAdapter工具介绍

[MSAdapter](https://openi.pcl.ac.cn/OpenI/MSAdapter.git)是一款MindSpore生态适配工具，在不改变用户原有使用习惯下，将PyTorch/JAX等三方框架代码快速迁移到MindSpore生态上，帮助用户高效使用昇腾算力。该工具的基本原理是使用MindSpore动态图算子来实现PyTorch API接口，由于框架的差异性，部分接口仍存在差异或者不支持，具体支持列表详见[torch接口支持列表](https://openi.pcl.ac.cn/OpenI/MSAdapter/src/branch/master/doc/readthedocs/source_zh/docs/SupportedList.md)。

## 1 软件安装

为了便于用户理解和选择合适的MindSpeed版本，我们提供了详细的版本配套表，如表1所示。
该表详细列出了MindSpeed版本与对应的MindSpore版本及CANN版本之间的匹配关系，确保用户能够根据自身软件环境准确选择相匹配的版本，以实现最优的性能与功能支持。

<table border="0">
  <tr>
    <td> MindSpeed版本 </td>
    <td> master </td>
  </tr>
  <tr>
    <td> MindSpeed代码分支名称 </td>
    <td> core_r0.8.0：配套Megatron-LM的core_r0.8.0分支 </td>
  </tr>
  <tr>
    <td> CANN版本 </td>
    <td> CANN 8.1.RC1 </td>
  </tr>
  <tr>
    <td> MindSpore版本 </td>
    <td> 2.7.0 </td>
  </tr>
  <tr>
    <td> MSAdapter版本 </td>
    <td> master </td>
  </tr>
  <tr>
    <td> Python版本 </td>
    <td> Python3.9.x, Python3.10.x </td>
  </tr>
</table>

### 安装操作

- 安装依赖的软件

<table border="0">
  <tr>
    <th>软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td> 昇腾NPU驱动 </td>
    <td rowspan="5">建议下载并安装左侧软件，具体请参见《<a href="https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit">CANN 软件安装指南</a>》</td>
  </tr>
  <tr>
    <td> 昇腾NPU固件 </td>
  </tr>
  <tr>
    <td> Toolkit（开发套件） </td>
  </tr>
  <tr>
    <td> Kenels（算子包） </td>
  </tr>
  <tr>
    <td> NNAL（Ascend Transformer Boost加速库） </td>
  </tr>
  <tr>
    <td> MindSpore框架 </td>
    <td> 建议下载并安装左侧软件，具体参见《<a href="https://www.mindspore.cn/install/">MindSpore 安装指南</a>》</td>
  </tr>
  <tr>
    <td> MSAdapter插件 </td>
    <td> 建议下载并安装左侧软件，具体参见《<a href="https://mindtorch.readthedocs.io/zh-cn/latest/docs/Install.html">MSAdapter 安装指南》</a></td>
  </tr>
</table>

- 下载MindSpeed-Core-MS源码master分支，执行一键适配。

  ```shell
    git clone https://gitee.com/ascend/MindSpeed-Core-MS.git -b master
    cd MindSpeed-Core-MS
    source auto_convert_mm.sh
  ```

  **说明：** MindSpeed-Core-MS源码提供了一键适配，用户无需再手动拉取MindSpeed等仓库源码。`auto_convert_xxx.sh`中`xxx`代表使用场景，可以是`llm`（大语言模型场景）、`mm`（多模态模型场景）、`rl`（强化学习场景），具体使用见[README](https://gitee.com/ascend/MindSpeed-Core-MS)。

## 2 MindSpore后端适配

对于MindSpeed PyTorch后端已支持但MindSpore后端不支持的模型，用户只需针对PyTorch后端提供的模型shell脚本进行少量适配修改，即可迁移至MindSpore后端。shell脚本适配的主体原则如下：

- 启动命修改为 `msrun`，并加入 `--ai-framework mindspore`参数；
  以下是一个启动命令适配的示例。

  适配前（Pytorch后端）：

  ```shell
    torchrun $DISTRIBUTED_ARGS pretrain_vlm.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        2>&1 | tee logs/train_${logfile}.log
  ```

  适配后（MindSpore后端）：

  ```shell
    msrun $DISTRIBUTED_ARGS pretrain_vlm.py \
        $GPT_ARGS \
        $MM_ARGS \
        $OUTPUT_ARGS \
        --distributed-backend nccl \
        --ai-framework mindspore \
        2>&1 | tee logs/train_${logfile}.log
  ```

- `DISTRIBUTED_ARGS`参数适配；
  `torchrun`命令使用的分布式参数包括 `--nproc_per_node, --nnodes, --node_rank, --master_addr, --master_port`，切换为 `msrun`后，分布式参数名需要适配调整。`msrun`使用的分布式参数和 `torchrun`的分布式参数关系见下表，各参数含义和使用详见[msrun启动](https://www.mindspore.cn/tutorials/zh-CN/r2.6.0/parallel/msrun_launcher.html?highlight=msrun)：

  | msrun 分布式参数   | 与torchrun分布式参数的关系 |
  | ------------------ | -------------------------- |
  | --local_worker_num | nproc_per_node        |
  | --worker_num       | nproc_per_node*nnodes |
  | --master_addr      | master_addr           |
  | --master_port      | master_port           |
  | --node_rank        | node_rank             |
  | --log_dir          | /                          |
  | --join             | /                          |
  | --cluster_time_out | /                          |
  | --bind_core        | /                          |

  以下是一个 `DISTRIBUTED_ARGS`参数适配示例。

  适配前（Pytorch后端）：

  ```shell
  # PyTorch后端启动脚本
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

  适配后（MindSpore后端）：

  ```shell
  # MindSpore后端启动脚本
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

- 确认PyTorch shell脚本中的特性开关所对应的特性在MindSpore后端已支持。若MindSpore后端未支持，我们建议在启动脚本中关闭这些特性。

完成上述启动shell脚本适配后，用户即可尝试使用脚本拉起模型任务。



