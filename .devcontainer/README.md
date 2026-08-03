# MindSpeed-MM Dev Container

在 VS Code 中一键启动 MindSpeed-MM 开发环境。

## 工作模式

[devcontainer.json](./devcontainer.json) 默认基于预先构建好的镜像（`image` 字段），拉取即用，无需本地构建。

> 配置文件中填写的 `image` 仅为示例，请根据实际需求选择所需镜像。镜像获取方式与更多信息请查看 [docker/OVERVIEW.zh.md](../docker/OVERVIEW.zh.md)。

如需自行修改构建参数，请在 `devcontainer.json` 中：

1. 注释掉 `image` 行
2. 取消注释 `build` 块与 `initializeCommand` 行

即可切换为从 [../docker/Dockerfile](../docker/Dockerfile) 本地构建的模式。

## 前置要求

- [VS Code](https://code.visualstudio.com/) + [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展
- 本地安装 Docker
- 宿主机已安装昇腾 NPU 驱动，用于容器内访问 NPU

## 使用方式

1. 用 VS Code 打开本项目根目录
2. 按 `F1` 打开命令面板，执行 **Dev Containers: Reopen in Container**
3. 首次启动会拉取预构建镜像（默认模式）或本地构建镜像（自建模式，约 20-40 分钟，取决于网络），完成后进入开发环境

> 自建模式下，`initializeCommand` 会调用 [prepare-context.sh](./prepare-context.sh)，自动下载 Miniconda 并准备 Dockerfile 所需的构建上下文文件，无需手动操作。

## 构建参数调整（自建模式）

编辑 [devcontainer.json](./devcontainer.json) 的 `build.args` 即可调整构建参数：
如果参数只是一个示例，请根据实际需求填写。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `NPU_TYPE` | `a3` | NPU 型号：`910b` 或 `a3` |
| `OS` | `openeuler24.03` | 操作系统：`openeuler24.03` 或 `ubuntu22.04` |
| `BASE_IMAGE_VERSION` | `9.0.0` | CANN 基础镜像版本 |
| `TORCH_VERSION` | `2.7.1` | PyTorch 版本（在线安装） |
| `TORCH_NPU_VERSION` | `2.7.1.post8` | torch_npu 版本（在线安装） |
| `MINDSPEED_MM_BRANCH` | `26.1.0` | MindSpeed-MM 分支 |

## NPU 设备配置

默认挂载 `/dev/davinci0` 与 `/dev/davinci1`。如需使用其他卡号，修改 `devcontainer.json` 中 `runArgs` 的 `--device=/dev/davinci2` 索引，或新增 `--device=/dev/davinci2` 等。

宿主机无 NPU 时，注释掉 `runArgs` 与 `mounts` 即可作为纯代码编辑环境使用。

## 工作区说明

- 本地源码通过 bind mount 挂载到容器 `/workspace/MindSpeed-MM`，编辑实时生效
- 容器创建后自动执行 `pip install -e . --no-deps` 以可编辑模式安装项目，并配置 pre-commit
- 默认 Python 解释器：`/opt/conda/bin/python`（conda base 环境）

## 常见问题排查

- **Miniconda 下载超时（自建模式）**
  手动下载 [Miniconda3-py311_26.1.1-1-Linux-${ARCH}.sh](https://repo.anaconda.com/miniconda/) 放到 `docker/downloads/` 目录后重新构建

- **decord 依赖下载失败（仅 ARM 架构需要 & 自建模式）**
  decord 编译依赖较多且常因网络问题下载失败，可用浏览器手动下载后放入 `docker/decord_deps/` 目录，再重新执行 **Reopen in Container**。下载链接参考 [docker/common/download_decord_deps.sh](../docker/common/download_decord_deps.sh)。
  完成后 `docker/decord_deps/` 目录结构应包含上述全部文件与目录。x86_64 架构无需下载这些依赖（decord 在镜像构建阶段一已通过 pip 安装）。
