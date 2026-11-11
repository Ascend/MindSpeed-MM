# USP (Hybrid Parallelism, Unified SP)

**Paper Title**: A Unified Sequence Parallelism Approach for Long-Context Generative AI

**Paper Link**: <https://arxiv.org/pdf/2405.07719.pdf>

**Project Repository**: <https://github.com/feifeibear/long-context-attention>

- Ulysses: When the sequence length and the number of compute devices increase proportionally, the communication volume remains constant. After the All2All operation, the partitioning of these four tensors moves from the sequence dimension L to the attention head dimension hc. In this way, the softmax(QK^T)V computation of each attention head can be performed completely without being interrupted by tensor partitioning.
- RingAttention: It can be regarded as a distributed version of FlashAttention. When each block of the output tensor O is computed, if the required K and V blocks are not available locally, they are obtained from other devices through point-to-point (P2P) communication. The communication of this method can be organized in a ring manner, where each device sends and receives K and V blocks at the same time, allowing communication to overlap with computation.

<div align="center"><img src="../../../../sources/images/features/cp/usp.png" width="400px"></div>

Ulysses maintains a constant communication volume when the number of compute devices and the sequence length increase, while Ring-Attention hides the point-to-point (P2P) communication cost introduced by SP through the overlap of computation and communication. However, the parallelism of DeepSpeed-Ulysses is limited by the number of attention heads, and Ring-Attention has low computational efficiency in blocked matrix multiplication.

To overcome these limitations, a unified sequence parallelism method is proposed that combines the techniques of DeepSpeed-Ulysses and Ring-Attention. Through a hybrid parallel strategy over the sequence dimension, this unified method allows DeepSpeed-Ulysses and Ring-Attention to run on different compute devices respectively, thereby optimizing the communication pattern and computational efficiency.

Specifically, the unified SP method runs in a process group of a 2D grid, where DeepSpeed-Ulysses operates along the rows of the grid and Ring-Attention operates along the columns. This setup not only improves the adaptability to model architectures and network hardware, but also improves the overall computational performance through more effective load balancing and communication patterns.

<div align="center"><img src="../../../../sources/images/features/cp/usp2.png" width="600px"></div>

## How to Use

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Long Sequence Parallelism Parameter Description</title>
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
                        <span class="required">Required</span>. Sets the long sequence parallelism size. The default value is 1. Configure it based on user requirements.
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="param-name">--context-parallel-algo</span>
                    </td>
                    <td>
                        <span class="optional">Optional</span>. Sets the long sequence parallelism algorithm. The default value is ulysses_cp_algo<br><br>
                        <span>ulysses_cp_algo</span>: Enables Ulysses long sequence parallelism, <span class="default">default value</span>.<br>
                        <span>megatron_cp_algo</span>: Enables Ring Attention long sequence parallelism.<br>
                        <span class="algorithm-name">hybrid_cp_algo</span>: Enables Hybrid long sequence parallelism.
                    </td>
                </tr>
                <tr>
                    <td>
                        <span class="param-name">--ulysses-degree-in-cp [int]</span>
                    </td>
                    <td>
                        <span class="optional">Required</span>. The parallelism degree of ulysses cp after usp is enabled<br><br>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>
</body>
</html>

> **Note**
>
> - Ensure that the value of `--context-parallel-size` is divisible by `--ulysses-degree-in-cp`. For example, when `--context-parallel-size 8` and `--ulysses-degree-in-cp 2` are set, the ring cp parallelism degree is 4.
> - Ensure that `--ulysses-degree-in-cp * --tensor-model-parallel-size` is divisible by `num-attention-heads`.
