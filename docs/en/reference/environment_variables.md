# Environment Variables

## Applicable Backends

This document applies to the Mcore, Mcore-FSDP2, and FSDP2 backends.

Common environment variables are divided into two categories according to who defines them: external environment variables and environment variables customized by this repository.

## External Environment Variables

<div class="container">
        <table class="mm-table">
            <thead>
                <tr>
                    <th>Environment Variable</th>
                    <th>Description</th>
                    <th>Value Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code class="env-var">ASCEND_SLOG_PRINT_TO_STDOUT</code></td>
                    <td>Whether to enable log printing</td>
                    <td>
                        <code class="value-desc">0</code>: Disable log printing to the screen<br>
                        <code class="value-desc">1</code>: Enable log printing to the screen
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ASCEND_GLOBAL_LOG_LEVEL</code></td>
                    <td>Set the log level of application logs</td>
                    <td>
                        <code class="value-desc">0</code>: DEBUG<br>
                        <code class="value-desc">1</code>: INFO<br>
                        <code class="value-desc">2</code>: WARNING<br>
                        <code class="value-desc">3</code>: ERROR<br>
                        <code class="value-desc">4</code>: NULL (no log output)
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">TASK_QUEUE_ENABLE</code></td>
                    <td>Control the optimization level of the task_queue operator dispatch queue</td>
                    <td>
                        <code class="value-desc">0</code>: Disabled<br>
                        <code class="value-desc">1</code>: Enable Level 1 optimization<br>
                        <code class="value-desc">2</code>: Enable Level 2 optimization
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">COMBINED_ENABLE</code></td>
                    <td>Optimize the scenario where two non-consecutive operators are combined</td>
                    <td>
                        <code class="value-desc">0</code>: Disabled<br>
                        <code class="value-desc">1</code>: Enabled
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">CPU_AFFINITY_CONF</code></td>
                    <td>Control the CPU core binding behavior of CPU tasks</td>
                    <td>
                        <code class="value-desc">0</code> or not set: no core binding<br>
                        <code class="value-desc">1</code>: coarse-grained core binding<br>
                        <code class="value-desc">2</code>: fine-grained core binding
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">HCCL_CONNECT_TIMEOUT</code></td>
                    <td>Timeout for establishing socket connections between devices</td>
                    <td>
                        Integer [120,7200]<br>
                        <code class="value-desc default-value">Default: 120</code> (seconds)
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">PYTORCH_NPU_ALLOC_CONF</code></td>
                    <td>Control the behavior of the caching allocator</td>
                    <td>
                        <code class="value-desc">expandable_segments:</code>: enable expandable segments of the memory pool
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">HCCL_EXEC_TIMEOUT</code></td>
                    <td>Synchronous execution wait time between devices</td>
                    <td>
                        Integer [68,17340]<br>
                        <code class="value-desc default-value">Default: 1800</code> (seconds)
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ACLNN_CACHE_LIMIT</code></td>
                    <td>Number of operator information entries cached on the host side</td>
                    <td>
                        Integer [1, 10,000,000]<br>
                        <code class="value-desc default-value">Default: 10000</code>
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">TOKENIZERS_PARALLELISM</code></td>
                    <td>Control the parallel behavior of Hugging Face tokenizers</td>
                    <td>
                        <code class="value-desc">False</code>: disable parallelism<br>
                        <code class="value-desc">True</code>: enable parallelism
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">MULTI_STREAM_MEMORY_REUSE</code></td>
                    <td>Control multi-stream memory reuse</td>
                    <td>
                        <code class="value-desc">0</code>: Disabled<br>
                        <code class="value-desc">1</code>: Enabled
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">NPU_ASD_ENABLE</code></td>
                    <td>Control the eigenvalue detection feature</td>
                    <td>
                        <code class="value-desc">0</code> or not set: disabled<br>
                        <code class="value-desc">1</code>: enabled, only print anomalies<br>
                        <code class="value-desc">2</code>: enabled with alarms<br>
                        <code class="value-desc">3</code>: enabled and record process data
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">ASCEND_LAUNCH_BLOCKING</code></td>
                    <td>Control the operator execution mode</td>
                    <td>
                        <code class="value-desc">0</code>: asynchronous execution<br>
                        <code class="value-desc">1</code>: synchronous execution
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

> **Note:** For more Ascend-related environment variables, see [Environment Variable List](https://www.hiascend.com/document/detail/en/canncommercial/83RC1/maintenref/envvar/envref_07_0001.html).

<div style="height: 40px;"></div>

## MindSpeed MM Custom Environment Variables

<div class="container">
        <table class="mm-table">
            <thead>
                <tr>
                    <th>Environment Variable</th>
                    <th>Description</th>
                    <th>Value Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><code class="env-var">MM_PRIME_CPU_AFFINITY</code></td>
                    <td>Whether to trigger the torch_npu CPU core binding in advance when training starts (FSDP2 backend). By default, torch_npu applies CPU_AFFINITY_CONF core binding only in the first backward pass; triggering it in advance also places the previously allocated page-locked memory (such as the swap arena slab) on the bound nodes. The gain from core binding and the correctness of the mapping are platform-dependent. See the NUMA notes in the swap_core document for details.</td>
                    <td>
                        <code class="value-desc default-value">auto (default)</code>: trigger in advance only when CPU_AFFINITY_CONF is set.<br>
                        <code class="value-desc">0</code>: do not trigger in advance (for A/B comparison or troubleshooting).<br>
                        <code class="value-desc">1</code>: always trigger in advance, even if CPU_AFFINITY_CONF is not set.
                    </td>
                </tr>
                <tr>
                    <td><code class="env-var">NPUS_PER_NODE</code></td>
                    <td>Configure the number of NPUs used by a node.</td>
                    <td>
                        Integer, such as <code class="value-desc">1</code>, <code class="value-desc">8</code>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
