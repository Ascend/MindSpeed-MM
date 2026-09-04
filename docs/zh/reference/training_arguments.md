# 训练参数

## 适用后端

该文档适用于Mcore以及Mcore-FSDP2后端。

## 配置说明

对于`examples/xxxx/finetune_xxxx.sh`里的参数描述如下：

<table class="mm-table">
    <thead>
        <tr>
        <th style="text-align: left;">参数分类</th>
        <th style="text-align: left;">参数名称</th>
        <th style="text-align: left;">描述</th>
        <th style="text-align: left;">取值</th>
        <th style="text-align: left;">默认值</th>
        <th style="text-align: left;">注意事项</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="6" style="vertical-align: middle; font-weight: bold;">分布式相关参数</td>
            <td><code>--nproc_per_node</code></td>
            <td>每个节点的device数量</td>
            <td>整数值</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--nnodes</code></td>
            <td>节点个数</td>
            <td>整数值</td>
            <td>-</td>
            <td>单机设置为<code>1</code>，多机设置<code>&gt1</code></td>
        </tr>
        <tr>
            <td><code>--node_rank</code></td>
            <td>节点编号</td>
            <td>整数</td>
            <td><code>0</code></td>
            <td>通常0为主节点</td>
        </tr>
        <tr>
            <td><code>--master_addr</code></td>
            <td>主节点地址</td>
            <td>节点的ip地址</td>
            <td><code>localhost</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--master_port</code></td>
            <td>主节点的端口号</td>
            <td>整数</td>
            <td><code>6000</code></td>
            <td>注意端口冲突，最好不要设置为知名端口</td>
        </tr>
        <tr>
            <td><code>--distributed-backend</code></td>
            <td>分布式后端</td>
            <td>GPU: <code>hccl</code>, NPU: <code>nccl</code>:, CPU: <code>gloo</code>, HPC: <code>mpi</code></td>
            <td><code>hccl</code></td>
            <td><code>hccl</code>会被<code>torch_npu</code> patch成为<code>nccl</code></td>
        </tr>
        <tr>
            <td rowspan="3" style="vertical-align: middle; font-weight: bold;">MM配置</td>
            <td><code>--mm-data</code></td>
            <td>数据集相关配置</td>
            <td>json配置文件路径</td>
            <td>-</td>
            <td><code><a href="https://mindspeed-mm.readthedocs.io/zh-cn/latest/config/数据配置.html">配置可以参考数据配置</a></code></td>
        </tr>
        <tr>
            <td><code>--mm-model</code></td>
            <td>模型相关配置</td>
            <td>json配置文件路径</td>
            <td>-</td>
            <td><code><a href="https://mindspeed-mm.readthedocs.io/zh-cn/latest/config/模型配置.html">配置可以参考模型配置</a></code></td>
        </tr>
        <tr>
            <td><code>--mm-tool</code></td>
            <td>工具相关配置</td>
            <td>json配置文件路径</td>
            <td><code>./mindspeed_mm/tools/tools.json</code></td>
            <td><code><a href="https://mindspeed-mm.readthedocs.io/zh-cn/latest/config/工具配置.html">配置可以参考工具配置</a></code></td>
        </tr>
        <tr>
            <td rowspan="42" style="vertical-align: middle; font-weight: bold;">GPT_ARGS</td>
            <td><code>--use-mcore-models</code></td>
            <td>使用mcore模型</td>
            <td>-</td>
            <td>-</td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--tensor-model-parallel-size</code></td>
            <td>TP并行度设置</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--pipeline-model-parallel-size</code></td>
            <td>PP并行度设置</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--context-parallel-size</code></td>
            <td>序列并行数量设置</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--context-parallel-algo</code></td>
            <td>CP算法选择，可选范围：ulysses_cp_algo、hybrid_cp_algo、megatron_cp_algo</td>
            <td>字符串</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/ulysses-context-parallel.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--expert-model-parallel-size</code></td>
            <td>MOE网络中专家并行设置</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-distributed-optimizer</code></td>
            <td>分布式优化器，将优化器状态切分到各个设备上去独立完成计算与存储。启用后可显著降低显存消耗，提升计算资源利用率</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-granularity</code></td>
            <td>目前仅支持配置full用于开启全量重计算</td>
            <td>full</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-method</code></td>
            <td>重计算模式配置。
                - uniform: 将transformer层均匀划分组，每组大小由 --recompute-num-layers 指定，按组存入输入和激活值。
                - block: 前 --recompute-num-layers 个transformer层使用重计算，剩余层跳过。</td>
            <td>block 或 uniform</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--recompute-num-layers</code></td>
            <td>重计算的层数配置，具体作用取决于 --recompute-method 的设置</td>
            <td>整数</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--micro-batch-size</code></td>
            <td>MBS设置</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--global-batch-size</code></td>
            <td>GBS设置</td>
            <td>整数值</td>
            <td><code>$(($MBS*$GRAD_ACC_STEP*$DP))</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--num-workers</code></td>
            <td>数据dataloader的进程数</td>
            <td>整数值</td>
            <td><code>1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--seq_length</code></td>
            <td>序列长度，表示模型一次能够处理的单个样本中包含的token数量</td>
            <td>整数值</td>
            <td><code>1024</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--normalization</code></td>
            <td>使用RMSNorm。推荐搭配 --use-fused-rmsnorm 使用。</td>
            <td>RMSNorm</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--swiglu</code></td>
            <td>使用SwiGLU激活函数，推荐搭配 --use-fused-swiglu 使用。</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-fused-swiglu</code></td>
            <td>使能相关融合算子，仅在使用SwiGLU时有效。</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-fused-rmsnorm</code></td>
            <td>使能相关融合算子，仅在使用RMSNorm时有效。</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr</code></td>
            <td>最大学习率</td>
            <td>浮点数</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-warmup-fraction</code></td>
            <td>用于学习率"预热"阶段占总步长的比例</td>
            <td>浮点数（0~1）</td>
            <td><code>0.1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--min-lr</code></td>
            <td>最小学习率</td>
            <td>浮点数</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-beta1</code></td>
            <td>Adam优化器一阶矩超参数</td>
            <td>浮点数</td>
            <td><code>0.9</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-beta2</code></td>
            <td>Adam优化器二阶矩超参数</td>
            <td>浮点数</td>
            <td><code>0.999</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--adam-eps</code></td>
            <td>adam_epsilon，防止分母为零</td>
            <td>浮点数</td>
            <td><code>1e-8</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-decay-style</code></td>
            <td>学习率衰减策略</td>
            <td><code>constant</code>, <code>linear</code>, <code>cosine</code>, <code>inverse-square-root</code>, <code>WSD</code></td>
            <td><code>linear</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--weight-decay</code></td>
            <td>L2 正则化的超参数，用于防止模型过拟合</td>
            <td>浮点数</td>
            <td><code>0.1</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--weight-decay-exclude-modules</code></td>
            <td>参数级的权重衰减排除，通过配置参数名关键词（可多个）排除特定参数的权重衰减。</td>
            <td>字符串列表</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/d34d43019a0f2572fcf42363991d6461c09ab71b/docs/zh/features/parameter_lr_wd_tuning.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--lr-scale-modules</code></td>
            <td>参数级学习率缩放，通过配置参数名关键词（可多个）来对特定参数的学习率进行缩放。</td>
            <td>字符串列表</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/d34d43019a0f2572fcf42363991d6461c09ab71b/docs/zh/features/parameter_lr_wd_tuning.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--lr-warmup-init</code></td>
            <td>学习率预热的初始值</td>
            <td>浮点数</td>
            <td><code>0.0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--lr-warmup-iters</code></td>
            <td>学习率线性预热的迭代步数参数</td>
            <td>整数</td>
            <td><code>0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--clip-grad</code></td>
            <td>梯度裁剪的超参数</td>
            <td>浮点数</td>
            <td><code>1.0</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--train-iters</code></td>
            <td>训练迭代次数</td>
            <td>整数</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--seed</code></td>
            <td>随机种子</td>
            <td>整数</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--bf16</code></td>
            <td>使用torch.bfloat16格式训练，极大降低显存消耗</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--variable-seq-lengths</code></td>
            <td>启用可变序列长度</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--calculate-per-sample-loss</code></td>
            <td>按样本粒度计算loss</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/d34d43019a0f2572fcf42363991d6461c09ab71b/docs/zh/features/vlm_model_loss_calculate_type.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--calculate-per-token-loss</code></td>
            <td>按token粒度计算loss</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed-MM/blob/d34d43019a0f2572fcf42363991d6461c09ab71b/docs/zh/features/vlm_model_loss_calculate_type.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--ckpt-format</code></td>
            <td>保存时使用DCP格式</td>
            <td>torch 或 torch_dcp</td>
            <td><code>-</code></td>
            <td><a href="https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/fsdp2.md">详细介绍</a></td>
        </tr>
        <tr>
            <td><code>--init-model-with-meta-device</code></td>
            <td>使用FSDP2的meta初始化模型</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--use-cpu-initialization</code></td>
            <td>使用CPU初始化权重，需开启</td>
            <td>store_true</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--load</code></td>
            <td>模型权重路径</td>
            <td>字符串</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
        <tr>
            <td><code>--save</code></td>
            <td>权重保存路径</td>
            <td>字符串</td>
            <td><code>-</code></td>
            <td>-</td>
        </tr>
    </tbody>
    </table>

对于以`pretrain_transformers.py`为训练入口的模型，模型参数除了以命令行参数形式传入外，也可以通过相关yaml配置文件传入，详情见[配置概览](configuration_overview.md)。yaml配置文件中各参数含义同表格所述一致。

> **备注：** 更多Megatron和MindSpeed args可以参考 `megatron/training/arguments.py` 和 `mindspeed/arguments.py`
