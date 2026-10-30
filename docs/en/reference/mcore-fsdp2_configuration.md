# FSDP2 Configuration

## Applicable Backends

This document applies to the Mcore-FSDP2 backend.

## Configuration Description

The parameters in `examples/xxxx/fsdp2_config.yaml` are described as follows:<a id="fsdp2-config"></a>

<table class="mm-table">
    <thead>
        <tr>
            <th>Parameter Category</th>
            <th>Parameter Name</th>
            <th>Description</th>
            <th>Values</th>
            <th>Default</th>
            <th>Notes</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5"><strong>Basic Configuration</strong></td>
            <td><code>sharding_size</code></td>
            <td>Model parallel sharding size</td>
            <td><code>auto</code> or an integer value</td>
            <td>1</td>
            <td><code>auto</code> means the size of <code>world_size</code></td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy"><code>param_dtype</code></a></td>
            <td>Data type for parameter storage and computation</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td>Model dtype</td>
            <td>Training precision setting</td>
        </tr>
        <tr>
            <td><code>reduce_dtype</code></td>
            <td>Data type for gradient communication</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td><code>none</code></td>
            <td>Communication precision setting</td>
        </tr>
        <tr>
            <td><code>output_dtype</code></td>
            <td>Data type of the forward output</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td><code>none</code></td>
            <td>Output precision control</td>
        </tr>
        <tr>
            <td><code>cast_forward_inputs</code></td>
            <td>Automatic type conversion for forward inputs</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>Ensures that input types match</td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Module Wrapping</strong></td>
            <td><code>sub_modules_to_wrap</code></td>
            <td>Submodule paths to be sharded by FSDP</td>
            <td>List of module path strings</td>
            <td>-</td>
            <td>
                <strong>Pattern syntax</strong>:<br>
                • <code>model.layers.{*}</code>: matches all submodules<br>
                • <code>model.layers.{0-23}</code>: matches a range of layers<br>
                • <code>model.layers.{1,3,5}</code>: matches the specified layers
            </td>
        </tr>
        <tr>
            <td><code>ignored_modules</code></td>
            <td>Modules excluded from FSDP management</td>
            <td>List of module path strings</td>
            <td>-</td>
            <td>The format is the same as <code>sub_modules_to_wrap</code></td>
        </tr>
        <tr>
            <td rowspan="5"><strong>Memory Optimization</strong></td>
            <td><code>recompute_modules</code></td>
            <td>Modules whose activations are recomputed</td>
            <td>List of module path strings</td>
            <td>-</td>
            <td>The format is the same as <code>sub_modules_to_wrap</code><br><strong>Conflict avoidance</strong>: the Megatron recomputation feature must be disabled</td>
        </tr>
        <tr>
            <td><code>use_reentrant</code></td>
            <td>Checkpoint implementation type</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>Whether it is reentrant</td>
        </tr>
        <tr>
            <td><code>reshard_after_forward</code></td>
            <td>Timing of parameter re-aggregation</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>
                <code>true</code>: ZeRO3 (saves memory)<br>
                <code>false</code>: ZeRO2 (high performance)
            </td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy"><code>offload_to_cpu</code></a></td>
            <td>Offload parameters to the CPU</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>When enabled, set <code>--distributed-backend npu:hccl,cpu:gloo</code></td>
        </tr>
        <tr>
            <td><code>pin_memory</code></td>
            <td>Lock CPU memory</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>Takes effect only when <code>offload_to_cpu=true</code></td>
        </tr>
        <tr>
            <td rowspan="2"><strong>Performance Tuning</strong></td>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch"><code>num_to_forward_prefetch</code></a></td>
            <td>Number of layers prefetched for the forward pass</td>
            <td>Integer value</td>
            <td>0</td>
            <td>Communication and computation overlap optimization</td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch"><code>num_to_backward_prefetch</code></a></td>
            <td>Number of layers prefetched for the backward pass</td>
            <td>Integer value</td>
            <td>1</td>
            <td>Communication and computation overlap optimization</td>
        </tr>
    </tbody>
</table>
