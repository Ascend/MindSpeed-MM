# Deterministic Computing

## Overview

During model training, due to various random factors (such as random initialization, data shuffling, dropout, etc.), training results may differ even when using the same hyperparameters, resulting in loss curves that do not perfectly overlap.

However, in the following scenarios, it is necessary to ensure the consistency and reproducibility of computation results:

- Repeated experimental verification
- Parameter tuning
- Problem reproduction and debugging

## Enabling Deterministic Computing

### Primary Method

In MindSpeed MM, deterministic computation can be enabled by adding the following parameter to the training script:

```bash
--use-deter-comp
```

### Alternative Methods

1. **Using the `--npu-deterministic` parameter**

   Add the following parameter to the training script:

   ```bash
   --npu-deterministic
   ```

   For detailed instructions, refer to: [Deterministic Computing](https://gitcode.com/Ascend/MindSpeed/blob/26.0.0_core_r0.12.1/docs/en/features/npu_deterministic.md).

2. **Using the msprobe Tool**

   Deterministic computation can be enabled using the msprobe package from the MindStudio Training Tools. For usage instructions, refer to: [msprobe Documentation](https://gitcode.com/Ascend/msprobe/docs/en/dump/pytorch_data_dump_instruct.md#seed_all).

## Required Environment Variables

In specific scenarios, the following environment variables must be configured to ensure fully deterministic computing:

1. **Multi-Card/Multi-Node Scenarios**

   To enable deterministic computation and order preservation for reduction communication operators, set:

   ```bash
   export HCCL_DETERMINISTIC=true
   ```

   For details, refer to "HCCL_DETERMINISTIC" in the official Ascend documentation.

2. **Disable matmul Staggering Computation**

   To ensure consistent matrix multiplication computation order, set:

   ```bash
   export CLOSE_MATMUL_K_SHIFT=1
   ```

## Notes

- Enabling deterministic computing has a certain impact on training performance.
- In production environments, you may need to balance the trade-off between determinism and performance based on actual requirements.
