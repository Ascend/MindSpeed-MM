# Ulysses

Ulysses is a solution for long-sequence model training proposed by DeepSpeed. See [deepspeed-ulysses](https://github.com/deepspeedai/DeepSpeed/tree/master/blogs/deepspeed-ulysses) for more details.

Its core approach is as follows:

- Split the data along the sequence dimension to each NPU (N/cp_size, D);
- Before computing Attention, use All-To-All communication to aggregate Query, Key, and Value so that each device has the full sequence length, while each device processes only part of the attention heads so that Attention is computed in parallel;
- Finally, use All-To-All to collect the results along the attention heads and repartition them along the sequence dimension

<div align="center"><img src="../../../../sources/images/features/cp/ulysses.png" width="800px"></div>

## How to Use

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Long Sequence Parallelism Parameters</title>
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
                    <th width="350">Configuration Parameter</th>
                    <th>Parameter Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>
                        <span class="param-name">--context-parallel-size [int]</span>
                    </td>
                    <td>
                        <span class="required">Required</span>. Sets the long sequence parallelism size. The default value is 1; configure it as needed.
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="param-name">--context-parallel-algo</span>
                    </td>
                    <td>
                        <span class="optional">Optional</span>. Sets the long sequence parallelism algorithm. The default is ulysses_cp_algo<br><br>
                        <span class="algorithm-name">ulysses_cp_algo</span>: enables Ulysses long sequence parallelism, <span class="default">the default value</span>.<br>
                        <span>megatron_cp_algo</span>: enables Ring Attention long sequence parallelism.<br>
                        <span>hybrid_cp_algo</span>: enables Hybrid long sequence parallelism.
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>
<div style="height: 40px;"></div>

> **Note**
>
> `num-attention-heads` must be divisible by `tensor-model-parallel-size` * `context-parallel-size`.
>
> - `um-attention-heads` indicates the number of attention heads.
> - `tensor-model-parallel-size` indicates the tensor parallelism scale.
> - `context-parallel-size` indicates the long sequence parallelism size.
