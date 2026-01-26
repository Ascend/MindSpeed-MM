
RingAttention
====================

**Ring Attention** 借鉴了分块 Softmax 的计算思想，无需获取完整序列的全局矩阵，即可实现分块自注意力计算。该方法将自注意力与前馈网络的计算过程分解为多个小块，并将这些块沿序列维度分布到多个计算设备上。

具体而言，Ring Attention 在进程之间构建了一个环状的通信结构（Ring）。每个进程持有本地的 Q、K、V 分块，首先完成本地注意力计算，随后通过在环中逐次向后发送和向前接收相邻进程的 KV 块，以迭代方式逐步完成全部注意力计算。在这种设计中，本地注意力计算与 KV 块的通信可实现流水线并行，理想情况下通信开销可被计算过程完全掩盖。

此外，由于 Ring Attention 在整个计算过程中无需拼接全局数据，其支持的序列长度在理论上可无限扩展，从而为超长序列建模提供了高效且可扩展的并行解决方案。

主要流程为：

* 数据切分：根据cp_size(示例=3)大小，将数据切分，每个rank拿到对应分片数据；
* 分块attention计算：计算分块数据的self-attention值（图中用FA2计算），获得单步数据；
* KV数据交换：rank之间搭建ring网络结构，每个rank与相邻rank交换KV数据；
* 单步计算修正：计算完attention后，需要对输出中间值L进行修正，保证输出正确；
* 计算最终输出：算完所有的分块attention后，对最终结果O进行修正、合并。


.. image:: 
    ../_static/features/cp/ring.png
    :width: 400px
    :align: center

使用方式
--------------

.. raw:: html

    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>长序列并行参数说明</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 20px;
                background-color: #fff;
            }
            
            .container {
                width: 100%;
                margin: 0 auto;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                border: 1px solid #ddd;
            }
            
            th {
                background-color: #f2f2f2;
                padding: 12px 15px;
                text-align: left;
                border-right: 1px solid #ddd;
                border-bottom: 1px solid #ddd;
            }
            
            td {
                padding: 12px 15px;
                border-right: 1px solid #ddd;
                border-bottom: 1px solid #ddd;
            }
            
            tr:last-child td {
                border-bottom: none;
            }
            
            th:last-child, td:last-child {
                border-right: none;
            }
            
            .param-name {
                font-family: monospace;
                color: #c00;
                font-weight: bold;
            }
            
            .required {
                color: #c00;
                font-weight: bold;
            }
            
            .optional {
                color: #090;
                font-weight: bold;
            }
            
            .algorithm-name {
                font-family: monospace;
                color: #00c;
            }
            
            .default {
                color: #666;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <table>
                <thead>
                    <tr>
                        <th width="350">配置参数</th>
                        <th>参数说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>
                            <span class="param-name">--context-parallel-size [int]</span>
                        </td>
                        <td>
                            <span class="required">必选</span>，设置长序列并行大小，默认为1，根据用户需求配置。
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--context-parallel-algo</span>
                        </td>
                        <td>
                            <span class="optional">可选</span>，设置长序列并行算法，默认是ulysses_cp_algo<br><br>
                            <span>ulysses_cp_algo</span>：开启Ulysses长序列并行，<span class="default">缺省值</span>。<br>
                            <span class="algorithm-name">megatron_cp_algo</span>：开启Ring Attention长序列并行。<br>
                            <span>hybrid_cp_algo</span>：开启Hybrid长序列并行。
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--use-cp-send-recv-overlap</span>
                        </td>
                        <td>
                            <span class="optional">可选</span>，建议开启，开启后支持send receive overlap功能<br><br>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--attention-mask-type [general/causal]</span>
                        </td>
                        <td>
                            <span class="optional">可选</span>，默认是causal mask计算，设置general代表全量计算<br><br>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--megatron-cp-in-bnsd</span>
                        </td>
                        <td>
                            <span class="optional">可选</span>，开启表示使用bnsd格式Attention计算<br><br>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--cp-window-size [int]</span>
                        </td>
                        <td>
                            <span class="optional">可选</span>，可选，默认为1，即使用原始的Ring Attention算法；当设置为大于1时，即使用Double Ring Attention算法，优化原始Ring Attention性能，--cp-window-size即为算法中双层Ring Attention的内层窗口大小，需要确保cp_size能被该参数整除<br><br>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
    </html>


