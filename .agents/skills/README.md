# MindSpeed-MM Skills

This directory is reserved for future concrete skills.

A skill should use the following structure:

```text
.agents/skills/<skill-name>/
└── SKILL.md
```

## Skill Requirements

- Define when the skill should be used.
- Define required inputs and expected outputs.
- Reference repository architecture from `../knowledge/architecture.md` when backend selection matters.
- Include validation expectations for code, docs, tests, or performance changes.

## Skill Plan

| Skill | Domain | Status | Priority | Description |
| --- | --- | --- | --- | --- |
| mindSpeed-mm-fsdp2-migration | Integration | Planned | P0 | 指导新模型接入 FSDP2 后端，覆盖参考样例、注册、配置、数据字段和端到端验收。 |
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
