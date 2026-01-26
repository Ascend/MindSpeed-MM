.. _fsdp2-config:

fsdp2配置
=============

.. raw:: html

    <a id="fsdp2_args"></a>
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            /* 基础表格样式 */
            table {
                width: 100%;
                border-collapse: collapse; /* 合并边框，使线条更细 */
                margin: 20px 0;
                font-family: sans-serif;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.05); /* 添加轻微阴影 */
            }

            /* 表头样式 */
            thead th {
                background-color: #2c3e50; /* 深色背景 */
                color: white;
                padding: 12px 15px;
                text-align: left;
                font-weight: bold;
                border-bottom: 3px solid #3498db; /* 底部边框 */
            }

            /* 单元格通用样式 */
            td, tbody th {
                padding: 12px 15px;
                border: 1px solid #ddd; /* 浅灰色边框 */
                vertical-align: top; /* 内容顶部对齐 */
                line-height: 1.5;
            }

            /* 模块类型列样式 (第一列) */
            td:first-child,
            tbody th {
                background-color: #f8f9fa; /* 轻微灰色背景 */
                font-weight: bold;
                color: #2c3e50;
            }

            /* 行交替背景色，提高可读性 */
            tbody tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            tbody tr:hover {
                background-color: #e9f7fe; /* 悬停高亮 */
            }

            /* 代码块样式 */
            code {
                background-color: #eee;
                font-family: 'Courier New', Courier, monospace;
                padding: 2px 6px;
                border-radius: 3px;
                color: #c7254e;
                font-size: 0.9em;
            }

            /* 标题样式 */
            caption {
                font-size: 1.5em;
                font-weight: bold;
                margin: 10px 0;
                color: #2c3e50;
            }
        </style>
    </head>
    <body>
        <table>
        <thead>
            <tr style="background-color: #f5f5f5;">
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
            <td rowspan="5" style="vertical-align: middle; font-weight: bold;">基本配置</td>
            <td><code>sharding_size</code></td>
            <td>模型并行分片大小</td>
            <td><code>auto</code>或整数值</td>
            <td>1</td>
            <td><code>auto</code>表示<code>world_size</code>大小</td>
            </tr>
            <tr>
            <td><code><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.MixedPrecisionPolicy">param_dtype</code></td>
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
            <td rowspan="2" style="vertical-align: middle; font-weight: bold;">模块包装</td>
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
            <td rowspan="5" style="vertical-align: middle; font-weight: bold;">内存优化</td>
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
            <td><code><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.CPUOffloadPolicy">offload_to_cpu</code></td>
            <td>参数卸载到CPU</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>启用时需要设置<code>--distributed-backend<br>npu:hccl,cpu:gloo</code></td>
            </tr>
            <tr>
            <td><code>pin_memory</code></td>
            <td>锁定CPU内存</td>
            <td><code>true</code>/<code>false</code></td>
            <td><code>false</code></td>
            <td>仅<code>offload_to_cpu=true</code>时生效</td>
            </tr>
            <tr>
            <td rowspan="2" style="vertical-align: middle; font-weight: bold;">性能调优</td>
            <td><code><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_forward_prefetch">num_to_forward_prefetch</code></td>
            <td>前向预取层数</td>
            <td>整数值</td>
            <td>0</td>
            <td>通信与计算重叠优化</td>
            </tr>
            <tr>
            <td><code><a href="https://docs.pytorch.org/docs/2.7/distributed.fsdp.fully_shard.html#torch.distributed.fsdp.FSDPModule.set_modules_to_backward_prefetch">num_to_backward_prefetch</code></td>
            <td>反向预取层数</td>
            <td>整数值</td>
            <td>1</td>
            <td>通信与计算重叠优化</td>
            </tr>
        </tbody>
        </table>
    </body>
    </html>
