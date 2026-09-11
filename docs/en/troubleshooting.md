# Troubleshooting Guide

This document provides troubleshooting methods for common MindSpeed MM issues by scenario, helping users quickly locate and resolve issues.

## Troubleshooting Installation Issues

### Environment Variables Not Taking Effect

**Symptom**: When the training script is run, an error such as `ModuleNotFoundError: No module named 'acl'` is reported.

**Troubleshooting steps**:

1. Check whether the CANN environment variables have been initialized.

   ```bash
   source /usr/local/Ascend/cann/set_env.sh
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

2. It is recommended to write the above commands into `~/.bashrc` to avoid executing them manually each time.
3. Verify whether the environment variables take effect.

   ```bash
   echo $ASCEND_HOME_PATH
   echo $LD_LIBRARY_PATH
   ```

### CANN Version Mismatch

**Symptom**: Training fails to start with errors, or operator errors occur.

**Troubleshooting steps**:

1. Check the current CANN version.

   ```bash
   cat /usr/local/Ascend/cann/version.info
   ```

2. Refer to [Version Compatibility](release_notes_mm.md#related-product-version-compatibility) to confirm whether the versions are compatible.
3. Check whether the TorchNPU version matches the CANN version.

   ```bash
   python -c "import torch_npu; print(torch_npu.__version__)"
   ```

4. If there is a version mismatch, reinstall the target version according to the compatibility table.

### libatb.so Not Found After NNAL Package Is Installed

**Symptom**: `OSError: libatb.so: cannot open shared object file`

**Troubleshooting steps**:

1. Verify the installation order: you must first run `source /usr/local/Ascend/cann/set_env.sh`, and then install the NNAL package.
2. After installing NNAL, source the environment variables again.

   ```bash
   source /usr/local/Ascend/nnal/atb/set_env.sh
   ```

3. Verify that the library file exists.

   ```bash
   find /usr/local/Ascend -name "libatb.so"
   ```

### Megatron-LM Version Mismatch

**Symptom**: `ImportError`, `AttributeError`, or runtime API incompatibility.

**Troubleshooting steps**:

1. Check the Megatron-LM version.

   ```bash
   cd Megatron-LM && git log --oneline -1
   ```

2. Confirm whether the correct version has been checked out (currently `core_v0.12.1` is recommended).
3. Check whether the Megatron directory is correctly copied to the MindSpeed-MM root directory.

---

## Troubleshooting Training Issues

### Training Loss Not Converging

**Symptom**: The training loss does not decrease or keeps oscillating.

**Troubleshooting steps**:

1. **Data check**
   - Verify that the data path and image path are correct.
   - Check data quality and cleaning logic.
   - Verify that the data preprocessing script executes correctly.
2. **Hyperparameter check**
   - Verify that the learning rate is set reasonably.
   - Verify that the warmup strategy is configured correctly.
   - Verify that the batch size is too large or too small.
3. **Weight check**
   - Verify that pretrained weights are used for initialization.
   - Verify that weight conversion is correct (verify that whether the TP/PP configurations are consistent).
4. **Precision check**
   - Verify that BF16 mixed precision training is used.
   - Verify that `NaN` or `Inf` values exist.

### NPU Out of Memory (OOM)

**Symptom**: `RuntimeError: NPU out of memory`

**Troubleshooting steps**:

1. Reduce `micro-batch-size` (down to a minimum of 1).
2. Increase parallelism.
   - Increase TP size (`--tensor-model-parallel-size`), noting that `tp_size <= num_key_value_heads`.
   - Increase PP size (`--pipeline-model-parallel-size`), noting that `TP×PP ≤ number of NPUs`.
3. Reduce `--seq-length`.
4. Enable recomputation.

   ```bash
   --recompute-granularity full --recompute-method block --recompute-num-layers <number of layers>
   ```

5. Use ChunkLoss to reduce peak memory usage (see [ChunkLoss](features/chunkloss.md)).
6. Use the distributed optimizer (`--use-distributed-optimizer`).

### LLM PP Split to 0 Layers

**Symptom**: `AssertionError: learning_rate is None`

**Troubleshooting steps**:

1. Check the PP layer configuration to ensure that each stage of the LLM part has layers.
2. Incorrect example: `llm=[0,8,10,10]` (the first stage has 0 LLM layers)
3. Correct example: `llm=[1,7,10,10]` (each stage has at least 1 LLM layer)

### tp_size Exceeds num_key_value_heads

**Symptom**: `RuntimeError: tp_size exceeds num_key_value_heads`

**Troubleshooting steps**:

1. Check the `num_key_value_heads` value in the model configuration.
2. Ensure that `--tensor-model-parallel-size` does not exceed this value.
3. If greater parallelism is required, combine PP or CP strategies.

---

## Troubleshooting Distributed Training Issues

### Multi-Node Startup Hang

**Symptom**: The script hangs during multi-node multi-device startup with no error message.

**Troubleshooting steps**:

1. Check whether `pdsh` is installed.

   ```bash
   which pdsh
   ```

   If not installed, run `apt install pdsh` or `yum install pdsh` for installation.
2. Check network connectivity between nodes.

   ```bash
   ping <MASTER_ADDR>
   ```

3. Check whether the port is occupied.

   ```bash
   netstat -tlnp | grep <MASTER_PORT>
   ```

### Communication Timeout

**Symptom**: `HCCL timeout`, `RuntimeError: Timeout`

**Troubleshooting steps**:

1. Check the NIC name and set the environment variable.

   ```bash
   ifconfig
   export HCCL_SOCKET_IFNAME=<NIC_name>
   export TP_SOCKET_IFNAME=<NIC_name>
   export GLOO_SOCKET_IFNAME=<NIC_name>
   ```

2. Increase the communication timeout interval.

   ```bash
   export HCCL_CONNECT_TIMEOUT=600
   ```

3. Check whether the CANN versions on all nodes are consistent.

### Load Imbalance Between Fast and Slow Ranks

**Symptom**: Low training efficiency, with some ranks waiting for a long time.

**Troubleshooting steps**:

1. Use profiling tools to analyze the computation time of each rank (see [Tool Usage](tools.md)).
2. Check whether heterogeneous parallel configuration is enabled.
3. Consider using multimodal heterogeneous PP splitting (see [Heterogeneous Parallelism](features/hetero_parallel.md)).
4. Consider enabling encoder data load balancing (see [Encoder Data Load Balancing](features/encoder_dp_balance.md)).

---

## Troubleshooting Weight Conversion Issues

### HuggingFace Weight Loading Failure

**Symptom**: `KeyError`, `RuntimeError: shape mismatch`

**Troubleshooting Steps**:

1. Confirm that the weights have been converted using the `mm-convert` tool (see [Weight Conversion](features/mm_convert.md)).
2. Check whether the parallel configuration used during conversion is consistent with the training script.
   - `tp_size`
   - `pp_size`
   - `llm_pp_layers` / `vit_pp_layers`
3. Check whether the weight files are fully downloaded.

### Weight Saving Timeout

**Symptom**: A timeout error occurs during checkpoint saving.

**Troubleshooting steps**:

1. Check whether the drive I/O bandwidth is normal.
2. Confirm whether the drive space is sufficient.
3. You can appropriately increase `--save-interval`.
4. This timeout usually does not affect the saved weights and can be ignored.

---

## Troubleshooting Docker Issues

### Missing Model Dependencies in the Container

**Symptom**: `ModuleNotFoundError` is reported during model run.

**Troubleshooting steps**:

1. The Docker image only pre-installs the basic dependencies: torch, TorchNPU, and decord.
2. Manually install additional dependencies in the base environment according to the README of the target model.
3. Refer to [Docker Usage](../../docker/OVERVIEW.md).

### CANN Version Conflict

**Symptom**: Training behaves abnormally in a multi-version CANN environment.

**Troubleshooting steps**:

1. It is recommended to use Docker to isolate different CANN versions.
2. Check whether the environment variables inside the container point to the correct version.
3. Avoid sharing the CANN installation path between the host and the container.

---

## Debugging Tips

### Enabling Synchronous Mode to Locate Errors

When an error occurs during training but the specific location cannot be identified, you can enable synchronous mode.

```bash
export ASCEND_LAUNCH_BLOCKING=1
```

> **Warning**: Synchronous mode severely impacts performance. Use it only for debugging, and it must be disabled in production environments.

### Enabling NPU Feature Value Detection

```bash
export NPU_ASD_ENABLE=2
```

Values:

- `0`: Disables detection.
- `1`: Prints exception logs only.
- `2`: Prints exception logs and issues warnings.
- `3`: Prints exception logs and alarms, and records process data in the device-side info log.

### Log Level Adjustment

```bash
export ASCEND_GLOBAL_LOG_LEVEL=0  # DEBUG, output the most detailed logs.
export ASCEND_SLOG_PRINT_TO_STDOUT=1  # Print logs directly to the terminal.
```

> **Note**: DEBUG-level logs generate heavy traffic and affect training performance. Enable them only briefly while troubleshooting issues.

### Using Profiling Tool

For detailed usage, see [Tool Usage](tools.md#profiling-tool).
