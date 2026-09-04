# fsdp2配置

## 适用后端

该文档适用于Mcore-FSDP2后端。

## 配置说明

对于`examples/xxxx/fsdp2_config.yaml`里的参数描述如下：<a id="fsdp2-config"></a>

<table class="mm-table">
    <thead>
        <tr>
            <th>参数分类</th>
            <th>参数名称</th>
            <th>描述</th>
            <th>取值</th>
            <th>默认值</th>
            <th>注意事项</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan="5"><strong>基本配置</strong></td>
            <td><code>sharding_size</code></td>
            <td>模型并行分片大小</td>
            <td><code>auto</code>或整数值</td>
            <td>1</td>
            <td><code>auto</code>表示<code>world_size</code>大小</td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy"><code>param_dtype</code></a></td>
            <td>参数存储和计算数据类型</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td>模型dtype</td>
            <td>训练精度设置</td>
        </tr>
        <tr>
            <td><code>reduce_dtype</code></td>
            <td>梯度通信数据类型</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td><code>none</code></td>
            <td>通信精度设置</td>
        </tr>
        <tr>
            <td><code>output_dtype</code></td>
            <td>前向输出数据类型</td>
            <td><code>bf16</code>, <code>fp16</code>, <code>fp32</code></td>
            <td><code>none</code></td>
            <td>输出精度控制</td>
        </tr>
        <tr>
            <td><code>cast_forward_inputs</code></td>
            <td>前向输入自动类型转换</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>确保输入类型匹配</td>
        </tr>
        <tr>
            <td rowspan="2"><strong>模块包装</strong></td>
            <td><code>sub_modules_to_wrap</code></td>
            <td>FSDP分片子模块路径</td>
            <td>模块路径字符串列表</td>
            <td>-</td>
            <td>
                <strong>模式语法</strong>:<br>
                • <code>model.layers.{*}</code>: 匹配所有子模块<br>
                • <code>model.layers.{0-23}</code>: 匹配层数范围<br>
                • <code>model.layers.{1,3,5}</code>: 匹配指定层数
            </td>
        </tr>
        <tr>
            <td><code>ignored_modules</code></td>
            <td>排除FSDP管理的模块</td>
            <td>模块路径字符串列表</td>
            <td>-</td>
            <td>格式同<code>sub_modules_to_wrap</code></td>
        </tr>
        <tr>
            <td rowspan="5"><strong>内存优化</strong></td>
            <td><code>recompute_modules</code></td>
            <td>激活值重计算模块</td>
            <td>模块路径字符串列表</td>
            <td>-</td>
            <td>格式同<code>sub_modules_to_wrap</code><br><strong>冲突避免</strong>: 需关闭Megatron重计算功能</td>
        </tr>
        <tr>
            <td><code>use_reentrant</code></td>
            <td>检查点实现类型</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>是否可重入</td>
        </tr>
        <tr>
            <td><code>reshard_after_forward</code></td>
            <td>参数重新聚合时机</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>true</code></td>
            <td>
                <code>true</code>: ZeRO3(省内存)<br>
                <code>false</code>: ZeRO2(高性能)
            </td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy"><code>offload_to_cpu</code></a></td>
            <td>参数卸载到CPU</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>启用时需要设置<code>--distributed-backend npu:hccl,cpu:gloo</code></td>
        </tr>
        <tr>
            <td><code>pin_memory</code></td>
            <td>锁定CPU内存</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>仅<code>offload_to_cpu=true</code>时生效</td>
        </tr>
        <tr>
            <td rowspan="2"><strong>性能调优</strong></td>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch"><code>num_to_forward_prefetch</code></a></td>
            <td>前向预取层数</td>
            <td>整数值</td>
            <td>0</td>
            <td>通信与计算重叠优化</td>
        </tr>
        <tr>
            <td><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch"><code>num_to_backward_prefetch</code></a></td>
            <td>反向预取层数</td>
            <td>整数值</td>
            <td>1</td>
            <td>通信与计算重叠优化</td>
        </tr>
    </tbody>
</table>
