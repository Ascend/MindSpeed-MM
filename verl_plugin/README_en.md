# GRPO Usage Guide

<p align="left">
</p>

## Contents

## Introduction

This document uses the reproduction of the [Group Relative Policy Optimization (GRPO)](https://arxiv.org/pdf/2402.03300) post-training method in the MindSpeed MM repository as an example to help users get started quickly, with plans to support multiple models in the future.

<a id="jump1"></a>

## Supported Models

- [Qwen2.5VL](../examples/verl_examples/qwen2.5vl/README.md)
- [Qwen3VL](../examples/verl_examples/qwen3vl/README.md)

<a id="jump2"></a>

## Performance Data

| Model         | Dataset   | Server        | GBS | n_samples | max_prompt_length | max_response_length | max_num_batched_tokens | End-to-end TPS |
|---------------|-----------|----------------------|-----|-----------|-------------------|---------------------|------------------------|----------------|
| Qwen2.5VL-7B  | geo3k     | Atlas 200T A2 Box16  | 512 | 5         | 1024              | 2048                | 8192                   | 142.42         |
| Qwen2.5VL-32B | geo3k     | Atlas 200T A2 Box16  | 256 | 5         | 1024              | 2048                | 8192                   | 88.32          |
| Qwen2.5VL-7B  | Non-public dataset | Atlas 200T A2 Box16 | 16  | 4         | 18,000            | 512                 | 19,000                 | 428.38         |
| Qwen2.5VL-32B | Non-public dataset | Atlas 200T A2 Box16 | 32  | 8         | 18,000            | 512                 | 20,000                 | 99.65          |
| Qwen3VL-8B    | geo3k     | Atlas 200T A2 Box16  | 512 | 5         | 1024              | 2048                | 8192                   | 429            |
| Qwen3VL-8B    | geo3k     | Atlas 200T A3 Box8   | 512 | 5         | 1024              | 2048                | 8192                   | 364*2          |
| Qwen3VL-30B   | geo3k     | Atlas 200T A2 Box16  | 64  | 5         | 1024              | 2048                | 8192                   | 21.76          |
| Qwen3VL-30B   | geo3k     | Atlas 200T A3 Box8   | 64  | 5         | 1024              | 2048                | 8192                   | 19.1*2         |
| Qwen3VL-30B   | geo3k     | Atlas 200T A2 Box16  | 64  | 5         | 16384             | 1024                | 18000                  | 275            |
| Qwen3VL-30B   | geo3k     | Atlas 200T A3 Box8   | 64  | 5         | 16384             | 1024                | 18000                  | 267*2          |

**NOTE**: The performance results on non-public datasets are for reference only.
