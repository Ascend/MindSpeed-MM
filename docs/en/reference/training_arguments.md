# Training Arguments

## Applicable Backends

This document applies to the Mcore and Mcore-FSDP2 backends.

## Configuration Description

The parameters in `examples/xxxx/finetune_xxxx.sh` are described as follows:

<table class="mm-table">
    <thead>
        <tr>
        <th style="text-align: left;">Parameter Category</th>
        <th style="text-align: left;">Parameter Name</th>
        <th style="text-align: left;">Description</th>
        <th style="text-align: left;">Value</th>
        <th style="text-align: left;">Default Value</th>
        <th style="text-align: left;">Notes</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="6" style="vertical-align: middle; font-weight: bold;">Distributed Parameters</td>
            <td><code>--nproc_per_node</code></td>
            <td>Number of devices per node</td>
            <td>Integer value</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--nnodes</code></td>
            <td>Number of nodes</td>
            <td>Integer value</td>
            <td>-</td>
            <td>Set to <code>1</code> for a single node, and <code>&gt1</code> for multiple nodes.</td>
        </tr>
        <tr>
            <td><code>--node_rank</code></td>
            <td>Node rank</td>
            <td>Integer</td>
            <td><code>0</code></td>
            <td>Usually, 0 is used for the master node.</td>
        </tr>
        <tr>
            <td><code>--master_addr</code></td>
            <td>Master node address</td>
            <td>IP address of the node</td>
            <td><code>localhost</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--master_port</code></td>
            <td>Port number of the master node</td>
            <td>Integer</td>
            <td><code>6000</code></td>
            <td>Avoid port conflicts; it is best not to use a well-known port.</td>
        </tr>
        <tr>
            <td><code>--distributed-backend</code></td>
            <td>Distributed backend</td>
            <td>GPU: <code>hccl</code>, NPU: <code>nccl</code>:, CPU: <code>gloo</code>, HPC: <code>mpi</code></td>
            <td><code>hccl</code></td>
            <td><code>hccl</code> is patched to <code>nccl</code> by <code>torch_npu</code></td>
        </tr>
        <tr>
            <td rowspan="3" style="vertical-align: middle; font-weight: bold;">MM Configuration</td>
            <td><code>--mm-data</code></td>
            <td>Dataset-related configuration</td>
            <td>Path to the JSON configuration file</td>
            <td>-</td>
            <td><code>For the configuration, see "Data Configuration".</code></td>
        </tr>
        <tr>
            <td><code>--mm-model</code></td>
            <td>Model-related configuration</td>
            <td>Path to the JSON configuration file</td>
            <td>-</td>
            <td><code>For the configuration, see "Model Configuration".</code></td>
        </tr>
        <tr>
            <td><code>--mm-tool</code></td>
            <td>Tool-related configuration</td>
            <td>Path to the JSON configuration file</td>
            <td><code>./mindspeed_mm/tools/tools.json</code></td>
            <td><code>For the configuration, see "Tool Configuration".</code></td>
        </tr>
        <tr>
            <td rowspan="42" style="vertical-align: middle; font-weight: bold;">GPT_ARGS</td>
            <td><code>--use-mcore-models</code></td>
            <td>Use the Mcore model</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--tensor-model-parallel-size</code></td>
            <td>TP parallelism degree</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--pipeline-model-parallel-size</code></td>
            <td>PP parallelism degree</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--context-parallel-size</code></td>
            <td>Sequence parallelism degree</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--context-parallel-algo</code></td>
            <td>CP algorithm selection. Available options: ulysses_cp_algo, hybrid_cp_algo, megatron_cp_algo</td>
            <td>String</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/en/features/ulysses-context-parallel.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--expert-model-parallel-size</code></td>
            <td>Expert parallelism degree in the MoE network</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-distributed-optimizer</code></td>
            <td>Distributed optimizer, which shards optimizer states to each device so that computation and storage are completed independently. When enabled, it significantly reduces device memory consumption and improves compute resource utilization.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-granularity</code></td>
            <td>Currently only full is supported, which enables full recomputation.</td>
            <td>full</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-method</code></td>
            <td>Recomputation mode configuration.
                - uniform: Transformer layers are evenly divided into groups, the size of each group is specified by --recompute-num-layers, and the inputs and activations are stored by group.
                - block: The first --recompute-num-layers transformer layers use recomputation, and the remaining layers are skipped.</td>
            <td>block or uniform</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-num-layers</code></td>
            <td>Number of layers for recomputation. The specific effect depends on the --recompute-method setting.</td>
            <td>Integer</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--micro-batch-size</code></td>
            <td>MBS setting</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--global-batch-size</code></td>
            <td>GBS setting</td>
            <td>Integer value</td>
            <td><code>$(($MBS*$GRAD_ACC_STEP*$DP))</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--num-workers</code></td>
            <td>Number of processes for the data dataloader</td>
            <td>Integer value</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--seq_length</code></td>
            <td>Sequence length, which indicates the number of tokens contained in a single sample that the model can process at one time.</td>
            <td>Integer value</td>
            <td><code>1024</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--normalization</code></td>
            <td>Use RMSNorm. It is recommended to use it together with --use-fused-rmsnorm.</td>
            <td>RMSNorm</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--swiglu</code></td>
            <td>Use the SwiGLU activation function. It is recommended to use it together with --use-fused-swiglu.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-fused-swiglu</code></td>
            <td>Enable the related fused operator. It takes effect only when SwiGLU is used.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-fused-rmsnorm</code></td>
            <td>Enable the related fused operator. It takes effect only when RMSNorm is used.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr</code></td>
            <td>Maximum learning rate</td>
            <td>Float</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-warmup-fraction</code></td>
            <td>Proportion of the total number of steps occupied by the learning rate "warmup" phase</td>
            <td>Float (0~1)</td>
            <td><code>0.1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--min-lr</code></td>
            <td>Minimum learning rate</td>
            <td>Float</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-beta1</code></td>
            <td>First-moment hyperparameter of the Adam optimizer</td>
            <td>Float</td>
            <td><code>0.9</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-beta2</code></td>
            <td>Second-moment hyperparameter of the Adam optimizer</td>
            <td>Float</td>
            <td><code>0.999</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-eps</code></td>
            <td>adam_epsilon, which prevents the denominator from being zero.</td>
            <td>Float</td>
            <td><code>1e-8</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-decay-style</code></td>
            <td>Learning rate decay strategy</td>
            <td><code>constant</code>, <code>linear</code>, <code>cosine</code>, <code>inverse-square-root</code>, <code>WSD</code></td>
            <td><code>linear</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--weight-decay</code></td>
            <td>Hyperparameter of L2 regularization, used to prevent model overfitting</td>
            <td>Float</td>
            <td><code>0.1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--weight-decay-exclude-modules</code></td>
            <td>Parameter-level weight decay exclusion. The weight decay of specific parameters is excluded by configuring parameter name keywords (multiple keywords are supported).</td>
            <td>List of strings</td>
            <td><code>-</code></td>
            <td><a href="../features/optimization/parameter_lr_wd_tuning.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--lr-scale-modules</code></td>
            <td>Parameter-level learning rate scaling. The learning rate of specific parameters is scaled by configuring parameter name keywords (multiple keywords are supported).</td>
            <td>List of strings</td>
            <td><code>-</code></td>
            <td><a href="../features/optimization/parameter_lr_wd_tuning.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--lr-warmup-init</code></td>
            <td>Initial value of the learning rate warmup</td>
            <td>Float</td>
            <td><code>0.0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-warmup-iters</code></td>
            <td>Number of iterations for the linear learning rate warmup</td>
            <td>Integer</td>
            <td><code>0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--clip-grad</code></td>
            <td>Hyperparameter for gradient clipping</td>
            <td>Float</td>
            <td><code>1.0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--train-iters</code></td>
            <td>Number of training iterations</td>
            <td>Integer</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--seed</code></td>
            <td>Random seed</td>
            <td>Integer</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--bf16</code></td>
            <td>Train with the torch.bfloat16 format, which greatly reduces device memory consumption.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--variable-seq-lengths</code></td>
            <td>Enable variable sequence lengths</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--calculate-per-sample-loss</code></td>
            <td>Calculate the loss at the sample granularity</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td><a href="../features/optimization/vlm_model_loss_calculate_type.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--calculate-per-token-loss</code></td>
            <td>Calculate the loss at the token granularity.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td><a href="../features/optimization/vlm_model_loss_calculate_type.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--ckpt-format</code></td>
            <td>Use the DCP format when saving</td>
            <td>torch or torch_dcp</td>
            <td><code>-</code></td>
            <td><a href="../features/parallel/fsdp2.md">Details</a></td>
        </tr>
        <tr>
            <td><code>--init-model-with-meta-device</code></td>
            <td>Initialize the model with FSDP2 meta.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-cpu-initialization</code></td>
            <td>Initialize the weights on the CPU. This must be enabled.</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--load</code></td>
            <td>Model weight path</td>
            <td>String</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--save</code></td>
            <td>Weight save path</td>
            <td>String</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

For models that use `pretrain_transformers.py` as the training entry, model parameters can be passed in as command-line arguments or through the related yaml configuration files. For details, see [Configuration Overview](configuration_overview.md). The meaning of each parameter in the yaml configuration file is the same as described in the table above.

> **Note:** For more Megatron and MindSpeed args, see `megatron/training/arguments.py` and `mindspeed/arguments.py`
