# FPDT (Fully Pipelined Distributed Transformer, Ulysses + Offload)

## Applicable Backends

Mcore

## Problem Analysis

Traditional Transformer models suffer from pipeline blocking, communication latency, and computational resource waste during multimodal and long-sequence inference, severely impacting inference efficiency and throughput. The native Ulysses model still exhibits numerous performance bottlenecks during inference/pre-training; its sequence partitioning is relatively coarse-grained, which leads to many bubbles and low compute resource utilization. At the same time, the strong synchronization dependencies between its modules prevent effective overlap of communication and computation, making blocking prone to occur.

## Solution

Based on the native Ulysses sequence partitioning logic, FPDT further splits the partitioned sequence into multiple chunks, and combines computation and communication streams with parallel scheduling to achieve intra-module concurrency; through CPU-NPU load/offload operations, it alleviates performance and memory bottlenecks and effectively improves resource utilization. Results vary depending on the model and parameter size; tuning `FPDT_chunk_number` and `FPDT_with_offload` can yield performance gains in all cases.

## How to Use

Use Case: when the video resolution/frame count is set very high and a single device runs out of memory (OOM) under CP partitioning during training, FPDT needs to be enabled. Currently only the wan2.1 1.3B T2V example (`WanDiT`) provides this configuration.

Enabling method: enable CP > 1 and modify the following JSON configuration keys in the `predictor` section of the startup script `pretrain_model.json` (prerequisites: `context_parallel_algo` must be set to `ulysses_cp_algo` or `hybrid_cp_algo`, and the attention type must be self-attention):

```json
"predictor": {
  "FPDT": true,
  "FPDT_chunk_number": 4,
  "FPDT_with_offload": true
}
```

### Parameter Description

| Parameter | Type | Default Value | Description |
|------|------|------|------|
| `FPDT` | bool | `false` | Enabling key |
| `FPDT_chunk_number` | int | `None` | Number of chunks. Ensure that per_gpu_seq_len is divisible by it; this is mandatory when FPDT is enabled, and the default (`None`) causes an error during FPDTFlashAttention initialization |
| `FPDT_with_offload` | bool | `false` | Whether to enable the offload feature |

### Example Script

See `examples/wan2.1/1.3b/t2v/pretrain.sh` (for the feature configuration keys, see the `predictor` section of `examples/wan2.1/1.3b/t2v/pretrain_model.json`; in the example, `FPDT` defaults to `false`, and must be set to `true`).

## Notes

### Known Constraints

- The prerequisites for FPDT (CP > 1, `context_parallel_algo` set to `ulysses_cp_algo`/`hybrid_cp_algo`, and the attention type being self-attention) are not strictly verified in the code; if they are not met, attention does not enable FPDT, but embs/RoPE partitioning and FFN chunking still take effect, so avoid such a partially effective configuration.
- The check that per_gpu_seq_len is divisible by `FPDT_chunk_number` is performed at runtime during the attention forward pass, not at configuration time.

## Acknowledgements

1. GitHub project tutorial: <https://www.deepspeed.ai/tutorials/ulysses-offload/>
2. GitHub code: <https://github.com/deepspeedai/DeepSpeed/blob/master/deepspeed/sequence/fpdt_layer.py>
