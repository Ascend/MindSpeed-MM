# MindSpeed MM HDK 安装路径批量替换指南

## 背景

MindSpeed MM 仓库中的部分安装文档使用了硬编码的 `/usr/local/Ascend/driver` 路径。
如您使用的HDK的实际安装路径为 `/usr/local/npu/driver`，需在使用前完成批量替换，确保环境变量可以正常加载。

本指南提供使用 `replace_ascend_path.py` 脚本进行批量路径替换的完整步骤。

---

## 前置条件

- Python 3.7+
- 拥有仓库目录的读写权限
- 建议在执行替换前，先通过 git 将当前状态提交或备份

---

## 受影响的文件范围

| 文件类型 | 说明 | 典型路径示例 |
|---------|------|-------------|
| Shell 脚本（`.sh`）| 训练/测试启动脚本 | `examples/*/pretrain_*.sh`、`scripts/install.sh` |
| Markdown 文档（`.md`）| 安装指南、模型使用说明 | `docs/zh/pytorch/install_guide.md`、`docker/OVERVIEW.md` |
| RST 文档（`.rst`）| 用户指南 | `UserGuide/quick_start/环境搭建.rst` |
| Python 文件（`.py`）| 源码（如有路径引用） | 各模块源文件 |
| Dockerfile | Docker 镜像构建脚本 | `docker/Dockerfile` |

> 路径变体说明：仓库中存在以下几种 Ascend/driver 路径引用，均会被一并替换：
>
> - `/usr/local/Ascend/driver/`（Docker 挂载路径）
> - `/usr/local/Ascend/driver/lib64/`（Docker 挂载路径）
> - `/usr/local/Ascend/driver/version.info`（Docker 挂载路径）

---

## 使用步骤

### 第一步：进入仓库根目录

```bash
cd /path/to/MindSpeed-MM
```

### 第二步：预览将要修改的内容（推荐）

在实际修改前，先以 `--dry-run` 模式确认变更范围：

```bash
python3 scripts/replace_ascend_path.py --dry-run
```

输出示例：

```bash
[DRY RUN] Path replacement: /usr/local/Ascend/driver -> /usr/local/npu/driver
Scan directory : /path/to/MindSpeed-MM
File types     : .md, .py, .rst, .sh + Dockerfile
------------------------------------------------------------
Found XXX candidate file(s), processing...

  [would replace  12] docker/OVERVIEW.md
  [would replace  12] docker/OVERVIEW.zh.md
  ...

============================================================
[DRY RUN] XXX file(s) would be modified, XXX replacement(s) total.
          Remove --dry-run to apply changes.
```

### 第三步：执行批量替换

确认预览无误后，执行实际替换：

```bash
# 默认：将 /usr/local/Ascend/driver 替换为 /usr/local/npu/driver
python3 scripts/replace_ascend_path.py
```

执行完毕后，脚本会输出修改的文件数和替换总次数。

### 第四步：验证替换结果

```bash
# 检查是否还有未替换的路径
grep -r "/usr/local/Ascend/driver" . \
  --include='*.sh' --include='*.md' --include='*.rst' --include='*.py' \
  --exclude-dir='.git' | wc -l
```

---

## 执行后验证

### 1. 环境变量验证

```bash
# 验证 NPU 可用
npu-smi info
```

### 2. 核心功能冒烟验证

参考对应模型的readme进行安装、配置，验证训练流程可正常启动

```bash
# 生效CANN脚本(以实际安装路径为准)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 运行示例脚本（以具体模型为准）
bash examples/<model_name>/pretrain_<model_name>.sh
```

---

## 完整脚本参数说明

```bash
usage: replace_ascend_path.py [-h] [--source SOURCE] [--target TARGET]
                               [--dir DIR] [--extensions EXT [EXT ...]]
                               [--dry-run]

选项：
  -h, --help            显示帮助信息
  --source SOURCE       源路径（默认：/usr/local/Ascend/driver）
  --target TARGET       目标路径（默认：/usr/local/npu/driver）
  --dir DIR             扫描目录（默认：当前目录 .）
  --extensions EXT...   文件扩展名白名单（默认：.sh .md .rst .py）
  --dry-run             仅预览变更，不修改文件
```
