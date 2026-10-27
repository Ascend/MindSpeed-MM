# MindSpeed MM 测试用例编写指南

本文档详细说明如何为MindSpeed MM贡献DT用例。

## 一、背景与参考

### 1.1 MindSpeed MM仓CI门禁代码相关路径

| 用途 | 路径 |
| ------ | ------ |
| 测试用例 | `MindSpeed-MM/tests` |
| CI启动代码 | `MindSpeed-MM/ci` |

### 1.2 CI门禁范围

CI门禁看护以下两项指标：

1. **功能**：代码能够正常运行
2. **性能**：性能劣化不得超过5%

---

## 二、CI门禁看护列表

PR合入前都须通过全量CI门禁用例测试。

### 2.1 ST（系统测试）看护列表

> **说明**：ST用例看护性能指标，性能劣化不得超过5%。

| Module | Features | Scripts |
| :------- | :--------- | :-------- |
| **Pretrain** | Wan2.1 T2V, FSDP2 | [pretrain_wan2.1_t2v.sh](st/shell_scripts/pretrain_wan2.1_t2v.sh) |
| | Wan2.2 I2V, FSDP2 | [pretrain_wan2.2_i2v.sh](st/shell_scripts/pretrain_wan2.2_i2v.sh) |
| **Finetune** | Qwen3Omni, FSDP2 | [finetune_qwen3omni.sh](st/shell_scripts/finetune_qwen3omni.sh) |
| | Qwen3VL 30B, FSDP2 | [finetune_qwen3vl_30B.sh](st/shell_scripts/finetune_qwen3vl_30B.sh) |
| | Kimi-K2.5, FSDP2 | [finetune_kimik2_5.sh](st/shell_scripts/finetune_kimik2_5.sh) |
| **Inference** | Wan2.2 T2V, CP=2 | [inference_wan2.2_t2v.sh](st/shell_scripts/inference_wan2.2_t2v.sh) |

### 2.2 UT（单元测试）看护列表

> **说明**：UT用例看护功能指标，确保代码能够正常运行。

| 测试目录 | 说明 |
| ------ | ------ |
| [ut](ut/) | 依赖Megatron，测试与Megatron相关的代码实现。 |
| [ut_fsdp](ut_fsdp/) | 基于原生PyTorch FSDP的单元测试，不依赖Megatron。 |

---

### 2.3 UT 执行方式

执行UT需要可用的NPU、`torch_npu`和`triton-ascend`。

#### 2.3.1 安装环境与依赖

请先按照[安装指南](../docs/zh/guides/installation/install_guide.md#手动安装)完成Python环境、NPU驱动与固件、CANN、PyTorch、TorchNPU、MindSpeed及MindSpeed MM的安装，并加载CANN环境变量。运行`tests/ut`还需按指南安装Megatron-LM；单独运行`tests/ut_fsdp`无需安装Megatron-LM。版本配套及安装命令以安装指南和其中链接的配套说明为准；当前分支的Python、PyTorch要求见[项目依赖](../pyproject.toml)，UT使用`triton-ascend==3.2.2`，请按安装指南中的Triton-Ascend安装步骤配置。

完成上述安装后，在仓库根目录补充安装测试依赖：

```bash
python -m pip install -e '.[test]'
```

`.[test]`包含`pytest`、`pytest-mock`和`transformers[sentencepiece]`。若CANN安装路径与`ci/run_ci.sh`中的`/usr/local/Ascend/ascend-toolkit/set_env.sh`不一致，请按实际路径修改脚本中的`source`路径。

#### 2.3.2 执行UT

环境准备完成后，在仓库根目录执行：

```bash
bash ci/run_ci.sh --type=ut --skip_build=1
```

`--type=ut`依次执行`pytest -x tests/ut`和`pytest -x tests/ut_fsdp`，前者通过后执行后者；`--skip_build=1`（默认值）跳过依赖安装和项目构建。

如需由脚本安装项目测试依赖、构建项目并在检测到NPU时安装`triton-ascend==3.2.2`，可先执行`bash ci/run_ci.sh --type=ut --skip_build=0`。该模式完成安装和构建后即退出，**不会执行UT**，仍需再执行上面的`--skip_build=1`命令；NPU驱动、CANN和TorchNPU须提前安装。

本地调试也可直接执行`pytest -x tests/ut`、`pytest -x tests/ut_fsdp`，或指定单个文件，例如`pytest tests/ut/models/common/test_activations.py -v`。

UT包含Triton算子用例，首次运行会触发编译，整体耗时可达1小时以上。部分用例依赖云端预生成Tensor等数据，无法在本地跑通，需提交PR后通过流水线验证。

---

## 三、开发流程

```mermaid
flowchart LR
    A[需求分析] --> B[用例设计]
    B --> C[代码开发]
    C --> D[本地验证]
    D --> E[CI门禁]
    E --> F[PR评审]
    F --> G[合入代码]
```

---

## 四、开发规范

### 4.1 命名规范

#### 4.1.1 ST用例命名规则

| 测试类型 | 命名规则 | 示例 |
| :--------: | :--------- | :----- |
| pretrain | `pretrain_` + 模型名 + `.sh` | `pretrain_cogvideox_t2v_1_0.sh` |
| finetune | `finetune_` + 模型名 + `.sh` | `finetune_qwen2vl_7B.sh` |
| posttrain | `posttrain_` + 模型名 + `_` + 任务类型 + `.sh` | `posttrain_qwen2vl_dpo.sh` |
| inference | `inference_` + 模型名 + `.sh` | `inference_qwen2vl_7b_pp1.sh` |

#### 4.1.2 UT用例命名规则

```text
test_ + 目标文件名或特性、功能名
```

**示例**：`test_chunkloss.py`

### 4.2 用例规范

#### 4.2.1 ST用例要求

1. **环境配置**：因为CI服务器硬件是NPU，必须设置正确的NPU环境变量
2. **数据shuffle必须关闭**：多模态训练用例中需关闭数据shuffle以确保结果可复现
3. **模型减层运行**：为节省资源同时保证测试有效性，模型需要减层运行，但层数不能设置过低以避免性能波动过大
4. **基线数据**：每个ST用例需配套基线数据文件，放置于 `st/baseline_results/` 目录，文件名为 `${script_name}.json`

#### 4.2.2 UT用例要求

1. **代码编写风格**：需与现有UT用例保持一致
2. **命名规范**：所有用例以 `test` 作为命名前缀
3. **目录层级**：建议按照功能特性进行文件夹命名区分

#### 4.2.3 CI门禁时间要求

- 整个CI门禁执行时间须**小于40分钟**

#### 4.2.4 资源路径规范

| 资源类型 | 路径 |
| ---------- | ------ |
| 模型权重 | `/home/ci_resource/models` |
| 数据集 | `/home/ci_resource/data` |
