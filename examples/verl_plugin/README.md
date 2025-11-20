# GRPO 使用指南

<p align="left">
</p>

## 目录

- [简介](#jump0)
- [支持模型](#jump1)
- [性能数据](#jump2)

<a id="jump0"></a>

## 简介

以 MindSpeed MM 仓库复现 [Group Relative Policy Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) 后训练方法为例来帮助用户快速入门，后续规划支持多个模型。

<a id="jump1"></a>

## 支持模型

- [Qwen2.5VL](./verl_examples/qwen2.5vl/README.md)
- [Qwen3VL](./verl_examples/qwen3vl/README.md)

<a id="jump2"></a>

## 性能数据

| 模型           | 数据集      | 机器型号                | GBS | n_samples | max_prompt_length | max_response_length | max_num_batched_tokens | 端到端 tps |
|---------------|----------|---------------------|-----|-----------|-------------------|---------------------|------------------------|---------|
| Qwen2.5VL-7B  | geo3k    | Atlas 200T A2 Box16 | 512 | 5        | 1024              | 2048                | 8192                   | 142.42  |
| Qwen2.5VL-32B | geo3k    | Atlas 200T A2 Box16 | 256 | 5        | 1024              | 2048                | 8192                   | 103.50  |
| Qwen2.5VL-7B  | 非公开数据集   | Atlas 200T A2 Box16 | 16  | 4        | 18,000            | 512                 | 19,000                 | 428.38  |
| Qwen2.5VL-32B | 非公开数据集   | Atlas 200T A2 Box16 | 32  | 8        | 18,000            | 512                 | 20,000                 | 99.65   |
| Qwen3VL-8B    | geo3k    | Atlas 200T A2 Box16 | 512 | 5        | 1024              | 2048                | 8192                   | 429     |
| Qwen3VL-8B    | geo3k    | Atlas 200T A3 Box8  | 512 | 5        | 1024              | 2048                | 8192                   | 364*2   |
| Qwen3VL-30B   | geo3k    | Atlas 200T A2 Box16 | 64  | 5        | 1024              | 2048                | 8192                   | 21.76   |
| Qwen3VL-30B   | geo3k    | Atlas 200T A3 Box8  | 64  | 5        | 1024             | 2048                | 8192                  | 19.1*2  |
| Qwen3VL-30B   | geo3k    | Atlas 200T A2 Box16 | 64  | 5        | 16384              | 1024                | 18000                   | 275     |
| Qwen3VL-30B   | geo3k    | Atlas 200T A3 Box8  | 64  | 5        | 16384             | 1024                | 18000                  | 267*2   |
**注**：非公开数据集性能结果仅供参考。