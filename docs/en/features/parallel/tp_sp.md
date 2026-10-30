# TP-SP

TP-SP is a sequence parallelism technique first proposed by the Megatron-LM framework. Built on Megatron TP, it further shards the `Dropout` and `LayerNorm` modules of the Transformer model along the sequence dimension,

**Paper Link**: <https://arxiv.org/pdf/2205.05198>

<div align="center"><img src="../../../../sources/images/features/cp/tp-sp.png" width="600px"></div>

Usage

## How to Use

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Long Sequence Parallelism Parameter Description</title>
</head>
<body>
    <div class="container">
        <table>
            <thead>
                <tr>
                    <th width="350">Configuration Parameter</th>
                    <th>Parameter Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="param-name">--tensor-model-parallel-size [int]</span>
                    </td>
                    <td>
                        <span class="required">Required</span>. Sets the TP parallelism degree. SP uses the same parallelism degree as TP.
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="param-name">--sequence-parallel</span>
                    </td>
                    <td>
                        <span class="required">Required</span>. Sets SP parallelism<br><br>
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="param-name">--use-ascend-mc2</span>
                    </td>
                    <td>
                        <span class="optional">Optional.</span> In training scenarios where TP and SP are enabled, fuses the matmul and all_gather/reduce_scatter computation and communication operators to reduce memory overhead and improve computational efficiency<br><br>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>

> **Note**
>
> MoE models do not currently support enabling `--use-ascend-mc2`.
