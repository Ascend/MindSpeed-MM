# Async Activation Offload

## Background and Challenges

As the parameter scale of large models grows and sequence lengths increase, the demand for memory during training rises sharply. Current optimization strategies for activations and memory primarily rely on recomputation and sequence parallelism techniques. These techniques have the following bottlenecks:

- Recomputation saves memory by discarding activations from the forward pass and recomputing them during the backward pass, introducing substantial redundant computation.
- Sequence parallelism distributes the computation of a single sequence across multiple devices to reduce memory usage, but frequent cross-device communication can be difficult to effectively overlap.

To address the above challenges, an Async Activation Offload strategy can be used.

## Solution

- Memory Optimization: Offload activation tensors from the device side to the host side, significantly reducing peak memory usage.
- Asynchronous Execution: Utilize a multi-stream mechanism to asynchronously perform offload (D2H) and load (H2D), allowing data transfer to be overlapped with computation.
- Prefetching: During the backward pass, use a `prefetch` mechanism to load required tensors in advance, hiding load latency.

## How to Use

This feature organizes tensor lifecycle by block, enabling flexible management of activations across different model blocks. A usage example is as follows:

```python
with async_save_on_cpu(
    h2d_stream=h2d_stream,
    d2h_stream=d2h_stream,
    block_idx=block_idx,
    depth=depth,
    custom_check_fn=your_check_fn
):
    # Forward computation code for a model block, shown here as an example only
    output = layer(input)
```

### Parameter Details

- `h2d_stream`/`d2h_stream`: H2D and D2H streams. It is recommended to create a dedicated global stream for executing H2D and D2H tasks to achieve asynchrony with the computation stream.
- `block_idx`: Index of the current block within a model.
- `depth`: Total number of layers in a model.
- `custom_check_fn`: Custom validation function. Only activations that return `True` after validation will be offloaded. It is recommended to select tensors with high computational cost and small activation size for offloading. For tensors with large activation size but short computation time, recomputation is preferred. Otherwise, the overhead of H2D and D2H may be too large to be effectively overlapped with computation.

## Use Cases and Results

- Long-sequence scenarios for multimodal models: The computation load of self-attention grows quadratically with sequence length. By applying this scheme to offload activations from the forward pass of self-attention and skipping self-attention recomputation, end-to-end performance gains exceed 20% in typical scenarios.
- FSDP2 scenarios: Under the FSDP2 distributed strategy, model parameters are sharded and aggregated. For shorter sequence lengths, computation time cannot overlap communication overhead. By using this scheme to offload activations at recomputation entry points, memory is freed up, allowing larger micro-batch sizes or longer sequence lengths to increase the computation ratio. End-to-end performance gains exceed 60% in typical scenarios.
