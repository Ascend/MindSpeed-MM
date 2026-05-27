# MindSpeed-MM Agent Configuration

This directory contains shared guidance for AI coding agents working on MindSpeed-MM.

The `.agents` directory is the single source for reusable agent-facing context. Tool-specific directories such as `.codex/`, `.claude/`, `.cursor/`, or `.trae/` can be generated locally from this shared source when needed.

MindSpeed-MM follows the [Agent Skills](https://agentskills.io/home) convention for skill layout.

## Directory Layout

| Path | Purpose |
| --- | --- |
| `skills/` | Skill index and implementation conventions. |
| `knowledge/` | Shared knowledge context for agents. |
| `setup_agent.sh` | Optional helper for linking `.agents` into local tool-specific directories. |

## Usage

Link this shared configuration into a local agent directory:

```bash
bash .agents/setup_agent.sh codex
bash .agents/setup_agent.sh claude
bash .agents/setup_agent.sh cursor
bash .agents/setup_agent.sh trae
```

The script also accepts a custom agent name and creates `.<agent-name>/` as a local adapter directory. Generated adapter directories are added to `.git/info/exclude`.

## Architecture Summary

MindSpeed-MM supports two main training backend paths. Agents should identify the active backend before changing model code, data code, checkpoint conversion, examples, or tests.

| Backend | Primary Entries | Description |
| --- | --- | --- |
| MindSpeed Core / Megatron | `mindspeed_mm/training.py`, `mindspeed_mm/pretrain_*.py`, `examples/*/*.sh` | Megatron-style flow using model/data/forward providers and hybrid parallelism. |
| FSDP2 | `mindspeed_mm/fsdp/train/trainer.py`, `mindspeed_mm/config/config_manager.py`, `mindspeed_mm/fsdp/utils/register.py` | YAML-driven flow using plugin registration, `ModelHub`, FSDP2 data builders, and parallel plans. |

See `knowledge/architecture.md` for the agent-facing architecture overview.

## Skill Plan

| Skill | Domain | Status | Priority | Description |
| --- | --- | --- | --- | --- |
| mindspeed-mm-fsdp2-model-only-vlm-migration | Integration | Planned | P0 | 指导新模型接入 FSDP2 后端，覆盖参考样例、注册、配置、数据字段和端到端验收，当前阶段仅支持vlm迁移。 |
| performance-analysis-report | Optimization | Planned | P0 | 将 profiling 结果和训练日志整理为瓶颈分析报告与优化建议。 |
| fsdp2-dataset-migration | Integration | Planned | P0 | 指导新数据集接入 FSDP2 数据链路，覆盖 dataset type、collator 和 batch key。 |
| flops-mfu-analysis | Optimization | Planned | P0 | 基于模型配置、输入形状和运行指标估算 FLOPs 与 MFU。 |
| fused-operator-optimization | Optimization | Planned | P0 | 规划 RMSNorm、EP-BMM、ROPE 等融合算子替换及精度性能验证。 |
| npu-environment-setup | Integration | Planned | P1 | 梳理指定模型在 Ascend/NPU 环境下的依赖、环境变量、安装顺序和最小验证方式。 |
| best-configuration-recommendation | Optimization | Planned | P1 | 结合模型规模和并行策略，推荐可解释的训练配置组合（EP、TP、CP、FSDP）。 |
| transformers-alignment-gate | Verification | Planned | P1 | 为 Transformers 版本升级提供对齐检查。 |
| checkpoint-conversion-routing | Integration | Planned | P1 | 根据源格式、目标格式和模型类型选择合适的权重转换路径并检查关键参数。 |
| minimal-doc-sync | Collaboration | Planned | P2 | 根据代码变更识别 README、特性文档或 example 文档中的最小同步范围。 |
| pr-description-generation | Collaboration | Planned | P2 | 根据 diff、测试结果、风险和用户影响生成 PR 描述与评审申请内容。 |
| unit-test-authoring | Verification | Planned | P2 | 辅助编写符合仓库风格的单元测试 |

See `skills/README.md` for the full skill index.
