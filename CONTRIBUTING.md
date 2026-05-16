# 开发者贡献指导

## 前置条件（检视之前完成，否则不检视）

1. 编译测试通过，Clean Code 问题处理完成。未通过测试需标注原因，非 Clean Code 问题需申请屏蔽；

2. PR 标题简洁完备（标题采用“头名称(backend/ops): +描述”的格式，建议使用英文描述），所有头如下表格所示：

    | 头名称      | 涉及内容                   |
    | ------       | -------------------------- |
    | feat    | 新特性、模块、模型的合入  |
    | fix      | 缺陷修复                   |
    | docs        | 添加、修改文档             |
    | style       | 修改代码以符合 Clean Code 标准 |
    | adaptor   | 模型源码合入               |
    | chore   | 单独提交的测试用例         |

    括号中的 backend/ops 项可填写 torch、mindspore、triton，不填写默认为 torch。

    下面是两个 PR 标题的示例：
    - feat(triton): optimize solve_tril of GDN （表示提交了 triton 算子的性能优化）
    - docs: Add FSDP2 Muon optimizer feature guide (表示提交了 FSDP2 后端 Muon 优化器的说明文档)

3. 按照 `.gitcode/PULL_REQUEST_TEMPLATE.md` 的模板填写 PR 内容说明，创建 pull request 后会自动生成该模板，请勿随意删除相关内容，如不涉及直接说明不涉及的原因；

4. 代码需充分自验、自检，无明显问题再要求检视；

5. 完成 CLA 签署，PR 中显示 CLA yes 标签。

6. 提交的代码需要关联 issue，版本规划内的模型代码提交和性能优化点提交可以直接关联当前版本的 Roadmap。非项目成员的开源社区贡献者如果没有设置关联 issue 的权限，可以直接在 PR 内容说明中复制 issue 链接。代码合入后，issue 应及时关闭。

## Commits要求

1. PR 做到功能单一，不同目的的修改应分成多个 PR 提交；

2. 单一 PR 内多个 commit 记录需要合并，至多两条；

3. Commit 信息需清楚描述代码功能，模糊表述不予通过，如“修复 bug”、“添加适配文件”等；

4. 正则表达式需要经过安全扫描，公网地址需要声明。

5. 涉及新特性、新模型相关的代码提交，均需要有测试用例看护。如果测试用例不在本次 PR 提交，或者已有测试用例，请在 PR 说明的 `How was this patch tested?` 章节给出关联的 PR 链接或者测试用例路径。

## 检视要求

1. 检视人应严格检视，输出有效检视意见，不可直接打标，不得以业务紧急为由强行要求合入；

2. 检视意见需尽可能描述详细，最好给出修改建议；

3. 所有检视意见都需要闭环，本项目的成员需要勾选“已解决”，确保所有未解决的问题都已经处理才能合入。非本项目成员的社区开发者如果没有处理权限，需要回复每一条审查意见；

4. PR 需尽快闭环，检视人员在检视意见闭环后，需要打标。

5. 对于没有测试用例的代码合入（文档修改除外），需要 committer 在评论中说明，并给出相关验证结论后合入。

## Commit Message 和 Changelog 编写指南

[https://www.ruanyifeng.com/blog/2016/01/commit_message_change_log.html](https://gitee.com/link?target=https%3A%2F%2Fwww.ruanyifeng.com%2Fblog%2F2016%2F01%2Fcommit_message_change_log.html)
