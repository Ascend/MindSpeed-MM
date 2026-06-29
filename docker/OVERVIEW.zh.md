# MindSpeed MM Docker 镜像概述

## 快速参考

| 项目 | 说明 |
| ------ | ------ |
| **镜像名称** | mindspeed-mm |
| **维护者** | MindSpeed MM 团队 |
| **源码仓库** | [https://gitcode.com/Ascend/MindSpeed-MM](https://gitcode.com/Ascend/MindSpeed-MM) |
| **Dockerfile 路径** | `docker/` |
| **许可证** | Apache-2.0 |

## MindSpeed MM

MindSpeed MM：面向大规模分布式训练的昇腾多模态大模型套件，支持业界主流多模态大模型训练，旨在为华为 [昇腾芯片](https://www.hiAscend.com/) 提供端到端的多模态训练解决方案, 包含预置业界主流模型，数据工程，分布式训练及加速，预训练、微调、后训练、在线推理任务等特性。

MindSpeed MM 镜像基于 Ubuntu 22.04 和 openEuler 24.03 两种操作系统，支持 x86_64 和 aarch64（ARM64）两种 CPU 架构。镜像中预安装了以下软件：

- **PyTorch** + **torch_npu**：深度学习框架
- **decord 0.6.0**：高效视频解码库
- **CANN**：华为昇腾 AI 处理器基础软件栈
- **verl_qwen3vl conda 环境**：预编译完成的 VERL Qwen3VL 强化学习训练环境（包含 vllm、vllm-ascend、verl）

由于不同模型的依赖环境存在差异，镜像中仅预安装了上述基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的额外依赖。

镜像下载：请访问 [镜像中心](https://www.hiascend.com/developer/ascendhub) 搜索 mindspeed-mm，获取对应的 `docker pull` 命令。
当前仅提供`openEuler 24.03`操作系统及`aarch64`架构的镜像。

## 镜像 Tag 关键字段描述

镜像 Tag 命名遵循模板：`{版本号}-{芯片信息}-{操作系统}-py{Python版本}-{架构类型}`

| 字段 | 说明 | 示例值 |
| ------ | ------ | -------- |
| 版本号 | MindSpeed MM 版本标识，同时也是 Git 分支名称 | master |
| 芯片信息 | NPU 芯片类型（小写） | a3, 910b |
| 操作系统 | 操作系统 | openeuler24.03, ubuntu22.04 |
| Python版本 | Python 版本 | py3.11 |
| 架构类型 | CPU 架构 | x86_64, aarch64 |

### 示例 Tag

| Tag | NPU | 操作系统 | Python | 架构 |
| ----- | ----- | --------- | -------- | ------ |
| `master-a3-openeuler24.03-py3.11-x86_64` | A3 | openEuler 24.03 | 3.11 | x86_64 |
| `master-a3-openeuler24.03-py3.11-aarch64` | A3 | openEuler 24.03 | 3.11 | aarch64 |
| `master-910b-openeuler24.03-py3.11-x86_64` | 910B | openEuler 24.03 | 3.11 | x86_64 |
| `master-910b-openeuler24.03-py3.11-aarch64` | 910B | openEuler 24.03 | 3.11 | aarch64 |

## Dockerfile 归档路径

| NPU | 操作系统 | Dockerfile 路径 |
| ----- | --------- | ---------------- |
| A3 | openEuler | `docker/Dockerfile` |
| A3 | Ubuntu | `docker/Dockerfile` |
| 910B | openEuler | `docker/Dockerfile` |
| 910B | Ubuntu | `docker/Dockerfile` |

Dockerfile 命名遵循模板：`Dockerfile[.{芯片信息}.{操作系统}.{其他字段}]`

- 统一的 `Dockerfile` 通过构建参数支持所有 NPU 类型和操作系统版本
- 字段间连接符使用 `.`
- 字段内连接符使用 `-`
- 芯片信息使用小写（a3, 910b）
- 操作系统使用 PascalCase（openEuler, ubuntu）

## 项目目录结构规范

Docker 项目目录遵循清晰的分层结构，便于维护和扩展：

### 核心目录结构

```text
docker/
├── Dockerfile                 # 统一 Dockerfile，支持多 NPU 类型和操作系统
├── build.sh                   # 镜像构建脚本，支持多种参数配置
├── OVERVIEW.md                # 英文版说明文档
├── OVERVIEW.zh.md             # 中文版说明文档
└── scripts/                   # 脚本目录
    └── model_install/         # 模型环境安装脚本
        └── install_*.sh       # 具体模型安装脚本
```

### 目录说明

1. **Dockerfile**：统一的构建文件，通过构建参数支持所有 NPU 类型和操作系统版本
2. **build.sh**：镜像构建脚本，提供灵活的参数配置和自动识别功能
3. **scripts/**：按脚本功能进行目录组织
    - **model_install/**：存放模型环境安装脚本，命名格式为 `install_{环境名称}.sh`

### 脚本使用机制

`docker/build.sh` 脚本在构建过程中会：

1. 根据 `-v` 参数指定的版本号（默认为 master）定位到对应的脚本目录
2. 将 `docker/scripts/model_install/` 目录下的所有 `install_*.sh` 脚本复制到 `install_scripts/` 临时目录

**重要说明**：当前版本的 Dockerfile 仅执行预定义的特定安装脚本（如 `install_verl_qwen3vl.sh`）。如果需要添加新的模型环境安装脚本，需要同时更新 Dockerfile 以包含对新脚本的复制和执行逻辑。

## 1. 镜像使用指导

**重要提示：** 由于不同模型的依赖环境存在差异，镜像中仅预安装了 torch、torch_npu 和 decord 基础依赖包。用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

### 运行镜像

```bash
# 基本运行
docker run -it --rm \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash

# 使用 NPU 设备运行（示例：设备 /dev/davinci1）
# 根据实际情况修改 ascend-toolkit 路径
# 假设您的 NPU 设备安装在 /dev/davinci1 上，并且 NPU 驱动程序安装在 /usr/local/Ascend 上：
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash

# 挂载数据目录运行（示例：设备 /dev/davinci1）
# 根据实际情况修改 ascend-toolkit 路径
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /path/to/data:/data \
    -v /path/to/weights:/weights \
    mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64 bash
```

### 内置环境

镜像包含以下预配置环境：

| 环境 | 说明 | 工作目录 |
| ------ | ------ | --------- |
| base | 基础环境，包含 PyTorch、torch_npu、decord、MindSpeed MM | /workspace/MindSpeed-MM |
| verl_qwen3vl | VERL Qwen3VL 模型环境（vllm、vllm-ascend、verl） | /workspace/verl_qwen3vl |

**环境说明：**

- 由于 verl 环境需要进行源码编译，过程耗时较长，因此已在镜像中预先安装配置完成。
- 考虑到不同模型的依赖环境存在差异，镜像中仅预安装了 torch、torch_npu 和 decord 基础依赖包。
- 用户在拉取镜像并启动容器后，需根据目标模型的 README 文件，在 base 环境中手动安装该模型所需的依赖环境。

## 2. 本地自定义安装指导

### 构建脚本参数说明

构建脚本 `build.sh` 支持多种参数配置，以下默认值仅为示例，请根据实际需求调整：

| 参数 | 说明 | 默认值（示例） |
| ------ | ------ | ------------ |
| `-t, --npu-type` | NPU 类型：A3 或 910B | 无（必需） |
| `-o, --os` | 操作系统：openeuler24.03 或 ubuntu22.04 | openeuler24.03 |
| `-v, --version` | MindSpeed MM 版本标识，同时作为 Git 分支名称和脚本目录选择依据 | master |
| `--torch-version` | PyTorch 版本 | 2.7.1 |
| `--torch-npu-version` | torch-npu 版本 | 2.7.1 |
| `--base-image-version` | 基础镜像 CANN 版本 | 9.0.0-beta.2 |
| `--base-image` | 完整基础镜像名称 | 无 |
| `--torch-whl` | torch .whl 文件路径（离线安装） | 无 |
| `--torch-npu-whl` | torch-npu .whl 文件路径（离线安装） | 无 |
| `--cleanup-on-fail` | 构建失败时清理悬空镜像/容器 | 无 |

### 基础构建示例

```bash
cd docker

# 构建 A3 + openEuler 镜像（默认）
bash build.sh -t A3

# 构建 910B + openEuler 镜像
bash build.sh -t 910B

# 构建 A3 + Ubuntu 镜像
bash build.sh -t A3 -o ubuntu22.04

# 指定 PyTorch 版本构建
bash build.sh -t A3 --torch-version 2.7.1 --torch-npu-version 2.7.1

# 使用离线 .whl 文件构建
bash build.sh -t A3 \
    --torch-whl /path/to/torch-2.7.1+cpu-cp311-cp311-linux_x86_64.whl \
    --torch-npu-whl /path/to/torch_npu-2.7.1-cp311-cp311-linux_x86_64.whl

# 指定基础镜像版本构建
bash build.sh -t A3 --base-image-version 9.0.0

# 指定版本构建
bash build.sh -t A3 -v master
```

### 自动下载功能说明

构建脚本支持自动下载以下资源，请确保网络通畅：

1. **Miniconda 安装器**：当未指定 `--miniconda` 参数时自动下载
2. **decord 依赖包**：ARM 架构下自动下载
3. **基础镜像**：当指定 `--base-image` 且本地不存在时自动拉取

## 3. 自定义镜像构建/使用指导

### 自动识别基础镜像

构建脚本会自动识别基础镜像名称中的关键信息：

1. **NPU 类型识别**：从镜像 tag 中识别 `910b` 或 `a3` 模式
2. **操作系统识别**：从镜像 tag 中识别 `openeuler24.03` 或 `ubuntu22.04`
3. **自动生成镜像 tag**：基于识别到的信息自动生成符合命名规则的镜像 tag

### 最佳实践示例

以下示例展示了如何使用自定义基础镜像构建 MindSpeed MM 镜像：

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 自定义配置
BASE_IMAGE="swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.0.0-beta.2-910b-openeuler24.03-py3.11"
TORCH_VERSION="2.7.1"
TORCH_NPU_VERSION="2.7.1.post2"
MINDSPEED_MM_VERSION="master"

# 执行构建
bash "${SCRIPT_DIR}/build.sh" \
    --base-image "$BASE_IMAGE" \
    --torch-version "$TORCH_VERSION" \
    --torch-npu-version "$TORCH_NPU_VERSION" \
    -v "$MINDSPEED_MM_VERSION" \
    --cleanup-on-fail
```

**关键特性说明：**

1. **自动识别**：脚本会自动从 `BASE_IMAGE` 中识别 NPU 类型（910B）和操作系统（openeuler24.03）。如果`BASE_IMAGE`在系统中不存在，会自动拉取。
2. **自动生成 tag**：基于识别结果自动生成镜像 tag，如 `mindspeed-mm:master-910b-openeuler24.03-py3.11-x86_64`
3. **自动下载**：如果本地没有 Miniconda 安装器或 decord 依赖，脚本会自动下载
4. **失败清理**：`--cleanup-on-fail` 参数确保构建失败时清理悬空资源

### 添加其他模型环境安装指导

如果您需要为其他模型添加环境安装支持，可以按照以下流程操作：

#### 1. 查看模型示例

首先查看 `examples/` 目录下相关模型的 README 文件，了解模型的环境依赖和安装要求。

#### 2. 创建安装脚本并遵循目录结构规范

根据 `docker/scripts` 目录的层级结构规范，创建新的安装脚本：

1. **创建脚本**：在 `docker/scripts/model_install/` 目录下创建新的安装脚本
2. **命名规范**：脚本文件命名格式为 `install_{环境名称}.sh`

### 示例：添加 Qwen3.5 模型环境安装脚本

- **脚本路径**：`docker/scripts/model_install/install_qwen3.5.sh`

参考 `docker/scripts/master/model_install/install_verl_qwen3vl.sh` 的格式创建新的安装脚本：

```bash
#!/bin/bash
set -e

source /tmp/common_functions.sh

MINDSPEED_MM_BRANCH="${1:-master}"

ENV_NAME="your_model_name"  # 修改为您的模型环境名称
WORK_DIR="/workspace/${ENV_NAME}"

echo "=========================================="
echo "Installing ${ENV_NAME} environment"
echo "=========================================="

source /opt/conda/etc/profile.d/conda.sh

# Accept conda terms of service
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --clone base -n ${ENV_NAME} -y
conda activate ${ENV_NAME}

mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

# 根据模型需求安装依赖
# 例如：pip_install_retry "package_name==version" 3
# 例如：git clone 仓库并安装

# 重新安装 torch 和 torch_npu（如果需要）
reinstall_torch_and_npu

conda clean -ya && rm -rf /root/.cache/pip

echo "=========================================="
echo "${ENV_NAME} environment installation completed"
echo "=========================================="
```

#### 3. 更新 Dockerfile 以包含新脚本

由于当前 Dockerfile 仅执行预定义的特定安装脚本，需要手动更新 Dockerfile 以包含对新脚本的复制和执行逻辑：

1. **在 Dockerfile 中添加对新脚本的复制**：

   ```dockerfile
   # 复制新的安装脚本
   COPY install_scripts/install_your_model.sh /tmp/install_your_model.sh
   ```

2. **在 Dockerfile 中添加对新脚本的执行**：

   ```dockerfile
   # 执行新的安装脚本
   RUN chmod +x /tmp/install_your_model.sh && \
       bash /tmp/install_your_model.sh "$MINDSPEED_MM_BRANCH" && \
       rm -f /tmp/install_your_model.sh
   ```

3. **构建脚本的自动复制机制**：`docker/build.sh` 脚本会自动将 `docker/scripts/model_install/` 目录下的所有 `install_*.sh` 脚本复制到 `install_scripts/` 临时目录，因此新脚本会被自动复制到构建上下文。

4. **版本对应关系**：通过 `-v` 参数指定版本号，该版本号用指定git clone MindSpeed-MM的分支。

#### 4. 构建镜像测试

使用以下命令构建包含新模型环境的镜像：

```bash
# 使用 -v 参数指定版本号，构建脚本会自动找到对应的安装脚本
bash build.sh -t A3 -v master
```

#### 5. 更新文档

在文档的"内置环境"部分添加新环境的信息：

| 环境 | 说明 | 工作目录 |
| ------ | ------ | --------- |
| your_model_name | 您的模型环境描述 | /workspace/your_model_name |

#### 注意事项

1. **目录结构规范**：必须遵循 `docker/scripts/model_install/` 的目录结构
2. **脚本命名规范**：安装脚本必须命名为 `install_{环境名称}.sh` 格式
3. **Dockerfile 更新**：需要手动更新 Dockerfile 以包含对新脚本的复制和执行逻辑
4. **可执行权限**：确保安装脚本具有可执行权限
5. **依赖安装**：使用 `pip_install_retry` 函数进行重试安装
6. **清理工作**：在脚本末尾清理临时文件和缓存以减少镜像大小
7. **测试验证**：构建镜像后测试新环境是否正常工作

### 二次开发

基于此镜像创建自定义 Dockerfile：

```dockerfile
FROM mindspeed-mm:master-a3-openeuler24.03-py3.11-x86_64

RUN pip install your-package==1.0.0

COPY . /workspace/your-project

WORKDIR /workspace/your-project
```

构建并运行（示例：设备 /dev/davinci1）：

```bash
# 根据实际情况修改 ascend-toolkit 路径
docker build -t my-mindspeed-app:latest .
docker run -it --rm \
    --device=/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    my-mindspeed-app:latest bash
```

### 软件栈

| 组件 | 版本 |
| ------ | ------ |
| CANN | 9.0.0-beta.2 |
| Python | 3.11 |
| Miniconda | 26.1.1-1 |
| PyTorch | 2.7.1 |
| torch_npu | 2.7.1 |
| decord | 0.6.0 |
| MindSpeed MM | master |

### 兼容性变更说明

#### 日期：2026-04-20

- 初始发布版本
- 基于 CANN 9.0.0-beta.2
- PyTorch 2.7.1 + torch_npu 2.7.1
- Python 3.11（Miniconda 26.1.1-1）
- 包含 verl_qwen3vl conda 环境
- 支持 openEuler 24.03 和 Ubuntu 22.04

## 许可证

MindSpeed MM 基于 Apache License 2.0 许可证发布。详见 [LICENSE](https://gitcode.com/Ascend/MindSpeed-MM/blob/master/LICENSE) 文件。

与所有 Docker 镜像一样，这些镜像可能还包含受其他许可证约束的其他软件（例如基础发行版中的 Bash，以及所包含主要软件的任何直接或间接依赖项）。

对于预构建镜像的任何使用，镜像用户有责任确保对此镜像的任何使用符合其中包含的所有软件的相关许可证。
