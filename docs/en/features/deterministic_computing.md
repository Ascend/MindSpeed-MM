# Deterministic Computing

## Overview

During model training, due to various random factors (such as random initialization, data shuffling, dropout, etc.), training results may differ even when using the same hyperparameters, resulting in loss curves that do not perfectly overlap.

However, in the following scenarios, it is necessary to ensure the consistency and reproducibility of computation results:

- Repeated experimental verification
- Parameter tuning
- Problem reproduction and debugging

## Enabling Deterministic Computing

### Primary Method

In MindSpeed MM, deterministic computation is enabled based on the training backend:

- **Native FSDP2**: Set `use_deter_comp: true` in the training section of the model YAML configuration file.
- **MCORE/Megatron-based FSDP2**: Add the `--use-deter-comp` parameter to the training script.

### Alternative Methods

1. **Using the `--npu-deterministic` Parameter**
   Add the following parameter to the training script:

   ```bash
   --npu-deterministic
   ```

   For detailed instructions, refer to: [Deterministic Computing](https://gitcode.com/Ascend/MindSpeed/blob/26.1.0_core_r0.12.1/docs/en/features/npu_deterministic.md).

2. **Using the msProbe Tool**

   Deterministic computation can be enabled using the msProbe package from the MindStudio Training Tools. For usage instructions<!--, refer to [msProbe documentation](https://gitcode.com/Ascend/msprobe/blob/master/docs/en/user_guide/dump/pytorch_data_dump_instruct.md)-->.

## Required Environment Variables

In specific scenarios, the following environment variables must be configured to ensure fully deterministic computing:

1. **Multi-Card/Multi-Node Scenarios**
   To enable deterministic computation and order preservation for reduction communication operators, set:

   ```bash
   export HCCL_DETERMINISTIC=true
   ```

   For details, refer to [HCCL_DETERMINISTIC](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/82RC1/maintenref/envvar/envref_07_0099.html).

2. **Disable matmul Staggering Computation**
   To ensure consistent matrix multiplication computation order, set:

   ```bash
   export CLOSE_MATMUL_K_SHIFT=1
   ```

## Notes

- Enabling deterministic computing has a certain impact on training performance.
- In production environments, you may need to balance the trade-off between determinism and performance based on actual requirements.
