# FPDT (Ulysses + Offload)

## Problem Analysis

Traditional Transformer models suffer from pipeline bubbles, communication latency, and computational resource waste during multimodal and long-sequence inference, severely impacting inference efficiency and throughput. The native Ulysses model still exhibits numerous performance bottlenecks during inference/pre-training. Its coarse-grained sequence partitioning leads to significant bubble overhead and low compute resource utilization. Furthermore, the strong synchronization dependencies between its modules prevent effective overlap of communication and computation, making blocking prone to occur.

## Solution

FPDT, short for Fully Pipelined Distributed Transformer, alleviates performance and memory bottlenecks by introducing finer-grained sequence partitioning, a computation-communication masking mechanism, and CPU-NPU load/offload operations.

## Technical Approach

Based on the native Ulysses sequence partitioning logic, the partitioned sequence is further split into multiple chunks. By combining computation and communication streams with parallel scheduling, it achieves intra-module concurrency, effectively improving resource utilization.

## How to Use

- Use Case: When the video resolution or frame count is set very high, a single GPU may encounter out-of-memory (OOM) errors during computation, requiring the enablement of FPDT.
- Enabling Method: Modify the following variables in the startup script `pretrain_model.json`.

  - `--FPDT` enables FPDT (Ulysses Offload).
  - `--FPDT-chunk-number` sets the number of chunks.
  - `--FPDT-with-offload` enables the offload feature.

```json
...
"predictor":{
  ...
  ...
  "FPDT":true,
  "FPDT_chunk_number":4,
  "FPDT_with_offload":true

}
...
```

- When `CP > 1` is enabled, enable `FPDT` and `FPDT_chunk_number` simultaneously.
- It is necessary to ensure that `FPDT_chunk_number` is divisible by the `per_gpu_seq_len` value.

## Effectiveness

Results vary depending on the model and parameter size. Tuning `FPDT_chunk_number` and `FPDT_with_offload` can yield performance improvements in most scenarios.

## Acknowledgements

1. GitHub project tutorial:
<https://www.deepspeed.ai/tutorials/ulysses-offload/>

2. GitHub code:
<https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/fpdt_layer.py>
