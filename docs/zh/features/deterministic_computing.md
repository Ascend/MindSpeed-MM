# 确定性计算

## 概述

在模型训练过程中，由于各种随机因素（如随机初始化、数据打乱、dropout等）的影响，即使使用相同的超参数，每次训练的结果也可能存在差异，导致LOSS曲线无法完全重合。

然而，在以下场景中需要确保计算结果的一致性和可复现性：

- 重复实验验证
- 调参优化过程
- 问题复现和调试

## 启用确定性计算

### 主要方法

在MindSpeed MM中，根据训练后端启用确定性计算：

- **原生 FSDP2（native FSDP2）**：在模型 YAML 配置文件的 `training` 段设置 `use_deter_comp: true`。
- **MCORE / 基于 Megatron 的 FSDP2（megatron-FSDP2）**：在训练脚本中添加 `--use-deter-comp` 参数。

### 替代方案

1. **使用`--npu-deterministic`参数**  
   在脚本中添加：

   ```bash
   --npu-deterministic
   ```

   详细说明请参考：[MindSpeed确定性计算](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/npu_deterministic.md)

2. **使用msprobe工具**  
   通过MindStudio Training Tools工具链中的msprobe包启用确定性计算。  
   使用说明请参考：[msprobe文档](https://gitcode.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md#seed_all)

## 必需的环境变量配置

在特定场景下，还需要配置以下环境变量以确保完整的确定性计算：

1. **多卡/多机场景**  
   为启用归约类通信算子的确定性计算和保序功能，需要设置：

   ```bash
   export HCCL_DETERMINISTIC=true
   ```

   详情请参考：[HCCL_DETERMINISTIC说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/82RC1/maintenref/envvar/envref_07_0099.html)

2. **关闭matmul错峰计算**  
   为保证矩阵乘法计算顺序一致，需要设置：

   ```bash
   export CLOSE_MATMUL_K_SHIFT=1
   ```

## 注意事项

- 启用确定性计算会对训练性能产生一定影响。
- 在生产环境中，可以根据实际需求权衡确定性和性能。
