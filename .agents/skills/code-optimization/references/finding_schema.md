# MindSpeed-MM 代码优化 · finding + diff 输出契约

每个候选改动按下面的 schema 填写字段；全部候选汇总成一份 Markdown 报告文件（格式见文末“报告文件”一节）。只产出 diff，不直接改真实文件，是否落盘由评审后的人逐条决定。

## Schema

```json
{
  "id": "<short stable id, e.g. co-1>",
  "file": "relative/path.py",
  "lines": "start-end",
  "category": "dedup|deadcode|nesting|naming|redundant-compute|import|condition|reuse-util",
  "problem": "这段为什么不必要地复杂 / 为什么简化是安全的",
  "diff": "<包含必要配套改动的 unified diff>",
  "ripple": [
    {"file": "path", "what": "<相关调用方或契约>", "edit": "<核查依据及必要改动，或无需改动的原因>"}
  ],
  "ripple_complete": true,
  "tier": "可以改|可以小改",
  "value": 1,
  "rationale": "为什么这次改动保语义、不改变可观察行为"
}
```

字段说明：

- `category`：按主要收益从 schema 所列类别中选择。
- `diff`：标准 unified diff，包含主改动和必要的配套改动，能在报告基线上通过 `git apply --check`。需附加参数时，在候选详情记录参数和原因。
- `ripple`：记录与本候选实际相关的调用方或契约及核查依据。需要配套修改时，注明位置并纳入 `diff`。影响仅限本地时，说明核查过的范围和原因。
- `ripple_complete`：关联影响已查清且配套改动已纳入 diff 时填 `true`。影响范围仍不确定或配套修改缺失的事项放入跳过项。
- `tier`：局部改动填“可以改”。涉及共享链路、且影响范围可界定并有足够等价性证据的局部改动填“可以小改”。
- `value`：1 为局部微调，2 为明显的可读性提升，3 为让核心链路明显更清晰的改动。

## 示例（删除未使用 import）

```json
{
  "id": "co-1",
  "file": "mindspeed_mm/tools/example_tool.py",
  "lines": "3-3",
  "category": "import",
  "problem": "os 被 import 但全文件无任何使用，属未用 import，删除保语义。",
  "diff": "--- a/mindspeed_mm/tools/example_tool.py\n+++ b/mindspeed_mm/tools/example_tool.py\n@@ -1,5 +1,4 @@\n import sys\n-import os\n import json\n \n def run():\n",
  "ripple": [
    {"file": "mindspeed_mm/tools/example_tool.py", "what": "import", "edit": "none needed because os 在本文件其余处无引用，grep 'os\\.' 与 'os ' 均无命中"}
  ],
  "ripple_complete": true,
  "tier": "可以改",
  "value": 1,
  "rationale": "仅删除一个未使用的模块 import，不改变任何运行时行为或导出面。"
}
```

## 报告文件

一次扫描的全部候选写成一份 Markdown 报告，默认路径为仓库根目录 `code_optimization_report_<YYYYMMDD>.md`，不纳入版本控制；评审完成、采纳项落盘后即可删除。完整示例见 `../examples/code_optimization_report_example.md`。报告固定四个部分：

1. 头部：扫描范围、生成时的基线提交、生成日期、backend 语境、已完成和受限的验证，以及“本文件只列候选，未修改任何源码”的声明。
2. 候选清单：一张表，每行一个候选，列为 ID、文件、行、类别、value、tier、ripple_complete、决策。决策列由评审人填写（采纳 / 拒绝 / 待议），生成时留为“待定”。
3. 候选详情：每个候选一节，按 schema 字段展开，diff 放在 `diff` 代码块中，ripple 逐条列出。
4. 已识别但未提交：扫描中看到但按拒绝判据跳过的项，写明位置、现象与不提交原因（例如函数体不同的被遮蔽重复定义、禁碰区文件）。这一节让评审人知道技能主动放弃了什么，而不是漏看了。

采纳流程：评审人在清单表填决策，然后只对采纳的项把 `diff` 代码块保存为 patch 文件并 `git apply`，最后跑 `pre-commit` 与相关单测。
