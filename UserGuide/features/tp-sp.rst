tp-sp
==========

TP-SP是Megatron-LM框架最早提出的一种序列并行技术，是基于Megatron TP基础上，继续对Transformer模型的 ``Dropout`` 和 ``LayerNorm`` 模块进一步做序列切分，

| **论文链接**: https://arxiv.org/pdf/2205.05198

.. image:: 
    ../_static/features/cp/tp-sp.png
    :width: 600px
    :align: center

使用


使用方式
--------------

.. raw:: html

    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>长序列并行参数说明</title>
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
                            <span class="param-name">--tensor-model-parallel-size [int]</span>
                        </td>
                        <td>
                            <span class="required">必选</span>，设置TP并行度，SP和TP同并行度。
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--sequence-parallel</span>
                        </td>
                        <td>
                            <span class="required">必选</span>，设置SP并行<br><br>
                        </td>
                    </tr>
                    <tr>
                        <td>
                            <span class="param-name">--use-ascend-mc2</span>
                        </td>
                        <td>
                            <span class="optional">可选，</span>在开启TP和SP的训练场景下，matmul和all_gather/reduce_scatter计算和通信算子融合，减少内存开销并提高计算效率<br><br>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
    </html>

|

.. note:: 

    MoE类模型暂不支持开启--use-ascend-mc2

