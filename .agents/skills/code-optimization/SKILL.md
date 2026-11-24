---
name: code-optimization
description: Use when looking for semantics-preserving simplicity / clean-code improvements in the MindSpeed-MM repo, then proposing them as reviewable finding+diff items with complete ripple edits. Triggers： “代码优化”, “简洁度”, “clean code”, “code smell”, “optimize code”, “simplify”, “refactor”, “audit code”. Use even when the user does not name a specific file.
---

# MindSpeed-MM 代码简洁化 / 优化

## 目标与范围

在用户指定的范围内找保语义的简洁化改动，默认扫描 `mindspeed_mm/`。参考 [仓库架构](../../knowledge/architecture.md) 识别目标所在的后端或共享调用链。将疑似缺陷记录为待人工判断的事项。

## 改动边界

候选必须保持可观察行为，包括返回值与数值、异常和副作用，并保留现有公开接口、导出、注册名、配置及 checkpoint 契约。涉及训练主链路、共享数据处理或分布式通信时，只有影响范围可界定且有足够等价性证据的局部改动才可提出。跳过 NPU 设备适配、fused op 与 vendored 上游实现。

## 验收与产出

- 每条候选对应一处独立改动，说明可读性收益和保语义依据，提供在报告基线上通过 `git apply --check` 的完整 unified diff。
- 识别受到影响的调用方和契约，按改动选择核查方式。用 `ripple` 记录核查依据和必要的配套改动，确认完整后标记 `ripple_complete`。无法确认等价性或补齐影响范围的事项放入跳过项，并说明原因。
- 按 [报告格式](references/finding_schema.md) 生成 Markdown 候选报告，写明基线、已完成的验证与受限的验证。报告供评审逐条决策，采纳后才应用相应 diff。

## 按需参考

- [定位与核查](references/optimization_map.md) 提供扫描线索、共享链路入口和典型案例。
- [示例报告](examples/code_optimization_report_example.md) 展示候选、配套核查证据和跳过项。
