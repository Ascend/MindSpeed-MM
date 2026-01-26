ulysses
==============

Ulysses是DeepSpeed提出的针对长序列模型训练的解决方案。官方博客： `deepspeed-ulysses <https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses>`_ 。

其核心方案如下：

* 沿序列维度将数据切分至各个NPU (N/cp_size, D)；
* 计算Attention之前，使用 All-To-All 通信把 Query、Key 和 Value 进行聚合，以便每张卡上都具有完整序列长度，同时使得各张卡上只处理部分注意力头，以便并行计算Attention；
* 最后，再使用 All-To-All 来沿着注意力头收集结果，同时沿着序列维度重新分区

.. image::
    ../_static/features/cp/ulysses.png
    :width: 800px
    :align: center

使用方式
-------------

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
                            <span class="algorithm-name">ulysses_cp_algo</span>：开启Ulysses长序列并行，<span class="default">缺省值</span>。<br>
                            <span>megatron_cp_algo</span>：开启Ring Attention长序列并行。<br>
                            <span>hybrid_cp_algo</span>：开启Hybrid长序列并行。
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
    </html>

..  # 这是一个注释，用于保持空行
.. raw:: html

   <div style="height: 40px;"></div>

.. note::

    num-attention-heads需要够被tensor-model-parallel-size * context-parallel-size整除
    
    * um-attention-heads：表示注意力头数
    * tensor-model-parallel-size：表示张量并行规模
    * context-parallel-size：表示长序列并行大小
