#!/bin/bash
# shellcheck source=/dev/null
# 使用场景：PR流水线触发任务，云端服务器承载 UT / ST 任务运行。

set -e
WORKSPACE="$1"
# shellcheck disable=SC2034  # pr_id 由 CI 平台传入，保留参数位置
pr_id="$2"
branch="$3"
type="$4"
# 进入工作目录
echo "init env"
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 显式加载 conda 初始化钩子（非交互式 shell 不会自动 source ~/.bashrc）
source /opt/conda/etc/profile.d/conda.sh
conda activate "ci_${branch}"
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# ============================================================
# 打印当前 conda 环境的 pip 安装列表
# ============================================================
echo ""
echo "############################################################"
echo "##                  pip list 输出                          ##"
echo "############################################################"
echo ""
pip list
echo ""
echo "############################################################"
echo "##                pip list 输出结束                        ##"
echo "############################################################"
echo ""

# ============================================================
# 打印 NPU 硬件信息
# ============================================================
echo ""
echo "############################################################"
echo "##                npu-smi info 输出                        ##"
echo "############################################################"
echo ""
npu-smi info
echo ""
echo "############################################################"
echo "##              npu-smi info 输出结束                      ##"
echo "############################################################"
echo ""

#导出修改文件列表
cd "${WORKSPACE}/CODE/"
git diff-tree -r --name-only --no-commit-id "origin/${branch}" HEAD > "${WORKSPACE}/modify.txt"
echo "cat modify.txt"
cat "${WORKSPACE}/modify.txt"

cd "${WORKSPACE}"

# ============================================================
# Step A: 确定分支版本映射（显式指定各分支对应的 Megatron-LM / MindSpeed 版本）
#   MEGATRON_BRANCH  : Megatron-LM 分支（从镜像预下载目录复制）
#   MINDSPEED_BRANCH : MindSpeed 分支名（按分支下载）
#   MINDSPEED_COMMIT : MindSpeed 固定 commit id（按 commit 下载，用于 copy 到 CODE/）
# ============================================================
case "${branch}" in
    26.0.0)
        MEGATRON_BRANCH="core_v0.12.1"
        MINDSPEED_BRANCH="v${branch}_core_r0.12.1"
        MINDSPEED_COMMIT="26ba4eb1"
        ;;
    26.1.0)
        MEGATRON_BRANCH="core_v0.12.1"
        MINDSPEED_BRANCH="v${branch}_core_r0.12.1"
        MINDSPEED_COMMIT="26ba4eb1"
        ;;
    *)
        MEGATRON_BRANCH="core_v0.12.1"
        MINDSPEED_BRANCH="master"
        MINDSPEED_COMMIT="26ba4eb1"
        ;;
esac

echo "[CI] MindSpeed-MM branch : ${branch}"
echo "[CI] Megatron-LM branch   : ${MEGATRON_BRANCH}"
echo "[CI] MindSpeed branch     : ${MINDSPEED_BRANCH}"
echo "[CI] MindSpeed commit     : ${MINDSPEED_COMMIT}"

# ============================================================
# Step B: Copy Megatron-LM（从镜像预下载目录复制）
# ============================================================
echo "[CI] Copying Megatron-LM ..."
cp -r "/workspace/Megatron-LM_${MEGATRON_BRANCH}" "${WORKSPACE}/Megatron-LM"
echo "[CI] Megatron-LM commit: $(cd "${WORKSPACE}/Megatron-LM" && git rev-parse HEAD)"

# ============================================================
# Step C: Clone MindSpeed（直接下载到 /workspace，无缓存复用）
#   下载路径:
#     按分支下载: /workspace/MindSpeed-${MINDSPEED_BRANCH}
#     按commit下载: /workspace/MindSpeed-${MINDSPEED_COMMIT}
# ============================================================
# ------------------------------------------------------------
# C1: 按分支下载 MindSpeed
# ------------------------------------------------------------
TARGET_PATH_BRANCH="/workspace/MindSpeed-${MINDSPEED_BRANCH}"
echo "[CI] Cloning MindSpeed by branch (${MINDSPEED_BRANCH}) to ${TARGET_PATH_BRANCH} ..."
git clone https://gitcode.com/Ascend/MindSpeed.git -b "${MINDSPEED_BRANCH}" "${TARGET_PATH_BRANCH}"
echo "[CI] MindSpeed-${MINDSPEED_BRANCH} commit: $(cd "${TARGET_PATH_BRANCH}" && git rev-parse HEAD)"

# ------------------------------------------------------------
# C2: 按 commit 下载 MindSpeed
# ------------------------------------------------------------
TARGET_PATH_COMMIT="/workspace/MindSpeed-${MINDSPEED_COMMIT}"
echo "[CI] Cloning MindSpeed by commit (${MINDSPEED_COMMIT}) ..."
git clone https://gitcode.com/Ascend/MindSpeed.git -b "${MINDSPEED_BRANCH}" "${TARGET_PATH_COMMIT}"
cd "${TARGET_PATH_COMMIT}"
git checkout "${MINDSPEED_COMMIT}"
cd "${WORKSPACE}"
echo "[CI] MindSpeed-${MINDSPEED_COMMIT} commit: $(cd "${TARGET_PATH_COMMIT}" && git rev-parse HEAD)"

# ------------------------------------------------------------
# C3: 复制 mindspeed 到 CODE/ 目录（从固定 commit 版本复制，不存在则报错）
# ------------------------------------------------------------
echo "init mindspeed"
if [ -d "${TARGET_PATH_COMMIT}/mindspeed" ]; then
    echo "[CI] Copying mindspeed from commit version: ${TARGET_PATH_COMMIT}"
    cp -r "${TARGET_PATH_COMMIT}/mindspeed" "${WORKSPACE}/CODE/"
else
    echo "[ERROR] mindspeed not found in commit version: ${TARGET_PATH_COMMIT}"
    exit 1
fi

echo "init megatron"
cd "${WORKSPACE}/Megatron-LM"
cp -r megatron "${WORKSPACE}/CODE/"
cd "${WORKSPACE}"

echo "init mindspeed-mm"
cd "${WORKSPACE}/CODE/"
pip install -e .
pip install -e .[test]

echo "start test"
cd "${WORKSPACE}/CODE/ci"
export PYTHONPATH="$PYTHONPATH:${WORKSPACE}/CODE"
echo "TRITON_CACHE_DIR: /home/ci_resource/triton_cache/"
mkdir -p "/home/ci_resource/triton_cache/"
export TRITON_CACHE_DIR="/home/ci_resource/triton_cache/"
python access_control_test.py --type="${type}"

# ============================================================
# Step D: 清理资源
#   - 直接删除 /workspace 下的所有代码（Megatron-LM、两个 MindSpeed 代码）
# ============================================================
echo "[CI] Cleaning up /workspace ..."
rm -rf /workspace/Megatron-LM*
rm -rf /workspace/MindSpeed-*
rm -rf "${WORKSPACE}/Megatron-LM"
echo "[CI] Cleanup done."
