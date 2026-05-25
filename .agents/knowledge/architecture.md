# Architecture Overview

MindSpeed-MM is a multimodal training repository for Ascend devices. It includes model examples, training flows, model components, checkpoint conversion, evaluation, profiling, and optimization utilities.

This document is written for AI coding agents. Use it to choose the correct backend and entry points before proposing code, documentation, example, or test changes.

## Repository Layers

| Layer | Main Paths | Agent Focus |
| --- | --- | --- |
| Examples | `examples/` | Model-specific launch scripts, configs, README files, conversion commands, and performance settings. |
| Megatron-style training | `mindspeed_mm/pretrain_*.py`, `mindspeed_mm/training.py` | Shared training loop and model/data/forward provider flow. |
| FSDP2 training | `mindspeed_mm/fsdp/`, `mindspeed_mm/config/` | YAML-driven trainer, plugin registration, model hub, data builders, and parallel plans. |
| Models | `mindspeed_mm/models/`, `mindspeed_mm/fsdp/models/` | Model implementations, common modules, Transformers adapters, and FSDP2 wrappers. |
| Data | `mindspeed_mm/data/`, `mindspeed_mm/fsdp/data/` | Dataset construction, collators, and multimodal data handling. |
| Tools | `checkpoint/`, `mindspeed_mm/tools/`, `mindspeed_mm/fsdp/tools/` | Checkpoint conversion, profiling, memory profiling, and FLOPs tooling. |
| Tests | `tests/`, `ci/` | Unit tests, system tests, and CI entry points. |

## Dual Backend Model

MindSpeed-MM has two major training backend paths.

| Backend | Description | Primary Entries | Typical Change Areas |
| --- | --- | --- | --- |
| MindSpeed Core / Megatron | Megatron-style backend using Pipeline, Tensor, and Data parallelism through MindSpeed Core and Megatron adapters. | `mindspeed_mm/training.py`, `mindspeed_mm/pretrain_*.py`, `examples/*/*.sh` | `pretrain_*.py`, model providers, data providers, forward steps, Megatron args, legacy examples. |
| FSDP2 | FSDP2-oriented backend using YAML configuration, plugin registration, `ModelHub`, FSDP2 data builders, and explicit parallel plans. | `mindspeed_mm/fsdp/train/trainer.py`, `mindspeed_mm/config/config_manager.py`, `mindspeed_mm/fsdp/utils/register.py` | `mindspeed_mm/fsdp/models/`, `mindspeed_mm/fsdp/data/`, YAML configs, FSDP2 parallel configs. |

FSDP2 examples may still include Megatron-style launch arguments for compatibility with existing argument parsing and launch conventions.
