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
source /root/miniconda3/bin/activate "ci_${branch}"
rm -rf /root/.cache/torch_extensions/py38_cpu
rm -rf /root/.cache/torch_extensions/py310_cpu
rm -rf /root/.cache/torch_extensions/py311_cpu

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
# Step C: Clone MindSpeed（按日期缓存，已下载则直接复用）
#   缓存根目录: /home/ci_resource/mindspeed-mm_code/
#   缓存子目录:
#     固定 commit 缓存: MindSpeed-commit-date/  （存放 MindSpeed-${MINDSPEED_COMMIT}-${COMMIT_DATE}）
#     固定分支缓存:    MindSpeed-branch-date/  （存放 MindSpeed-${MINDSPEED_BRANCH}-${DATE_SUFFIX}）
# ============================================================
DATE_SUFFIX=$(date +%Y%m%d)
MINDSPEED_CACHE_BASE="/home/ci_resource/mindspeed-mm_code"
# 固定 commit 缓存目录与固定分支缓存目录
COMMIT_CACHE_DIR="${MINDSPEED_CACHE_BASE}/MindSpeed-commit-date"
BRANCH_CACHE_DIR="${MINDSPEED_CACHE_BASE}/MindSpeed-branch-date"

# 确保缓存根目录及两个子目录存在（若不存在则创建）
mkdir -p "${COMMIT_CACHE_DIR}" "${BRANCH_CACHE_DIR}"

# ------------------------------------------------------------
# C1: 固定分支缓存——当天同分支已下载则复用，否则克隆
#   缓存路径: ${BRANCH_CACHE_DIR}/MindSpeed-${MINDSPEED_BRANCH}-${DATE_SUFFIX}
# ------------------------------------------------------------
TARGET_PATH_BRANCH="${BRANCH_CACHE_DIR}/MindSpeed-${MINDSPEED_BRANCH}-${DATE_SUFFIX}"
if [ -d "${TARGET_PATH_BRANCH}" ]; then
    echo "[CI] Reusing cached MindSpeed (branch): ${TARGET_PATH_BRANCH}"
else
    echo "[CI] Cloning MindSpeed by branch (${MINDSPEED_BRANCH}) to ${TARGET_PATH_BRANCH} ..."
    git clone https://gitcode.com/Ascend/MindSpeed.git -b "${MINDSPEED_BRANCH}" "${TARGET_PATH_BRANCH}"
fi
echo "[CI] MindSpeed-${MINDSPEED_BRANCH} commit: $(cd "${TARGET_PATH_BRANCH}" && git rev-parse HEAD)"

# ------------------------------------------------------------
# C2: 固定 commit 缓存——查找已有缓存，命中则复用；否则克隆并归档
#   缓存路径: ${COMMIT_CACHE_DIR}/MindSpeed-${MINDSPEED_COMMIT}-${COMMIT_DATE}
#   - 若已存在该 commit 的任意缓存目录，直接复用（不再下载）
#   - 否则下载到临时目录，取该 commit 的合入日期（YYYYMMDD）作为后缀归档
# ------------------------------------------------------------
COMMIT_CACHE=$(ls -1dt "${COMMIT_CACHE_DIR}/MindSpeed-${MINDSPEED_COMMIT}-"*/ 2>/dev/null | head -n 1)
if [ -n "${COMMIT_CACHE}" ]; then
    TARGET_PATH_COMMIT="${COMMIT_CACHE%/}"
    echo "[CI] Reusing cached MindSpeed (commit): ${TARGET_PATH_COMMIT}"
else
    echo "[CI] Cloning MindSpeed by commit (${MINDSPEED_COMMIT}) ..."
    TMP_CLONE=$(mktemp -d "${COMMIT_CACHE_DIR}/tmp_clone_XXXXXX")
    git clone https://gitcode.com/Ascend/MindSpeed.git -b "${MINDSPEED_BRANCH}" "${TMP_CLONE}"
    cd "${TMP_CLONE}"
    git checkout "${MINDSPEED_COMMIT}"
    # 以 commit 合入日期（YYYYMMDD）作为后缀
    COMMIT_DATE=$(git show -s --format=%ci "${MINDSPEED_COMMIT}" | awk '{print $1}' | tr -d '-')
    cd "${WORKSPACE}"
    TARGET_PATH_COMMIT="${COMMIT_CACHE_DIR}/MindSpeed-${MINDSPEED_COMMIT}-${COMMIT_DATE}"
    if [ -d "${TARGET_PATH_COMMIT}" ]; then
        # 并发情况下已被下载，删除临时目录并复用
        rm -rf "${TMP_CLONE}"
        echo "[CI] Reusing cached MindSpeed (commit): ${TARGET_PATH_COMMIT}"
    else
        mv "${TMP_CLONE}" "${TARGET_PATH_COMMIT}"
        echo "[CI] Archived MindSpeed (commit) to ${TARGET_PATH_COMMIT}"
    fi
fi
echo "[CI] MindSpeed-${MINDSPEED_COMMIT} commit: $(cd "${TARGET_PATH_COMMIT}" && git rev-parse HEAD)"

# ------------------------------------------------------------
# C3: 复制 mindspeed 到 CODE/ 目录
#   优先使用固定 commit 缓存（确定性版本）；若不存在则回退到固定分支缓存
# ------------------------------------------------------------
echo "init mindspeed"
if [ -d "${TARGET_PATH_COMMIT}/mindspeed" ]; then
    # 固定 commit 情况：从 commit 缓存复制 mindspeed
    echo "[CI] Copying mindspeed from commit cache: ${TARGET_PATH_COMMIT}"
    cp -r "${TARGET_PATH_COMMIT}/mindspeed" "${WORKSPACE}/CODE/"
elif [ -d "${TARGET_PATH_BRANCH}/mindspeed" ]; then
    # 固定分支情况：从分支缓存复制 mindspeed
    echo "[CI] Copying mindspeed from branch cache: ${TARGET_PATH_BRANCH}"
    cp -r "${TARGET_PATH_BRANCH}/mindspeed" "${WORKSPACE}/CODE/"
else
    echo "[ERROR] mindspeed not found in commit or branch cache!"
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
python access_control_test.py --type="${type}"

# ============================================================
# Step D: 清理资源
#   - MindSpeed 固定 commit 缓存：保留最新 6 份，超出则删除最老的
#   - MindSpeed 固定分支缓存：保留最新 6 份，超出则删除最老的
#   - 删除本次运行复制的 Megatron-LM
# ============================================================
echo "[CI] Cleaning up MindSpeed commit cache (keep latest 6) ..."
# 按修改时间倒序列出固定 commit 缓存，保留最新 6 份，删除其余
mapfile -t COMMIT_CACHES < <(ls -1dt "${COMMIT_CACHE_DIR}/MindSpeed-"*/ 2>/dev/null)
if [ "${#COMMIT_CACHES[@]}" -gt 6 ]; then
    printf '%s\n' "${COMMIT_CACHES[@]}" | tail -n +7 | xargs -r rm -rf
fi

echo "[CI] Cleaning up MindSpeed branch cache (keep latest 6) ..."
# 按修改时间倒序列出固定分支缓存，保留最新 6 份，删除其余
mapfile -t BRANCH_CACHES < <(ls -1dt "${BRANCH_CACHE_DIR}/MindSpeed-"*/ 2>/dev/null)
if [ "${#BRANCH_CACHES[@]}" -gt 6 ]; then
    printf '%s\n' "${BRANCH_CACHES[@]}" | tail -n +7 | xargs -r rm -rf
fi

echo "[CI] Cleaning up cloned Megatron-LM ..."
rm -rf "${WORKSPACE}/Megatron-LM"
echo "[CI] Cleanup done."
