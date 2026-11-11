# coding=utf-8
"""CCE 融合 loss 端到端集成测试（小模型 + 随机权重 + mock 数据，自包含）。

对比 baseline（legacy chunk loss）与 cce（chunk_loss_cce_fused 路线）在真实训练管线
（trainer.py）下的 loss 对齐，覆盖 6 个精选场景。

自包含设计（区别于被替换的 tests/ut/chunk_loss_cce_qwen，该目录不会合入）：
  - 模型：从 CI 上的 HF 目录拷贝非权重文件生成 reduced config（减层数、保持 hidden=2048、
    关手机), 权重用固定 seed 随机初始化（training.load=""), 不加载任何预训练权重 → 不依赖 DCP。
  - 数据：调用仓库 generate_mock_data_for_vlmodel.py 生成 mock json（session 级幂等）。
  - gdn/causal_conv1d 强制 triton 实现。
  - 路径本身的存在性由环境守卫控制: 默认 CI 路径, 可用环境变量 QWEN3_5_PATH 覆盖;
    路径缺失时整模块 skip, 不阻塞 CI。

运行：
  pytest tests/ut/loss/test_cce_fused_integration.py -v

环境变量：
  QWEN3_5_PATH    HF 目录（含 config.json + tokenizer），默认 /home/ci_resource/models/Qwen3.5-35B-A3B
  CCE_UT_NPU      训练卡数，默认 8
  CCE_UT_LAYERS   reduced config 层数，默认 8
"""
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml
import pytest

_PROJ_ROOT = Path(__file__).resolve().parents[3]
_TEST_DIR = Path(__file__).resolve().parent

# 模型总路径（可用环境变量覆盖，默认 CI 资源路径）
PATH_MODEL = os.environ.get(
    "QWEN3VL_PATH",
    "/home/ci_resource/models/Qwen3-VL-8B-Instruct",
)
_HF_DIR = Path(PATH_MODEL)
if not (_HF_DIR / "config.json").exists():
    pytest.skip(f"HF model not found: {PATH_MODEL} (set QWEN3VL_PATH to override)",
                allow_module_level=True)

def _auto_npus():
    """探测可用 NPU 数并封顶到 8，适配 CI 容器卡数（如 4 卡）与本地 8 卡。"""
    try:
        import torch
        available = torch.npu.device_count()
        return 8 if available <= 0 else min(8, available)
    except Exception:
        return 8


NPUS = int(os.environ.get("CCE_UT_NPU", _auto_npus()))
LAYERS = int(os.environ.get("CCE_UT_LAYERS", "8"))

BASE_CONFIG = _PROJ_ROOT / "examples" / "qwen3vl" / "qwen3vl_8B_config_v1.yaml"
MOCK_SCRIPT = (_PROJ_ROOT / "mindspeed_mm" / "fsdp" / "tools" / "data_tool"
               / "generate_mock_data_for_vlmodel.py")
MODELING_SRC = _PROJ_ROOT / "mindspeed_mm" / "fsdp" / "models" / "qwen3vl"
MOCK_DIR = _PROJ_ROOT / "ut_out" / "loss_e2e" / "mocked_vl_data"
REDUCED_DIR = _PROJ_ROOT / "ut_out" / "loss_e2e" / "reduced_config"
LOG_DIR = _PROJ_ROOT / "ut_out" / "loss_e2e" / "logs"

SEED = 42
IGNORE_INDEX = -100
TOLERANCE = 1e-3


# ============================================================
# 配置生成辅助（原 create_reduced_config.py 的核心，内联）
# ============================================================
def copy_config_files(src_dir: Path, dst_dir: Path) -> None:
    """复制 HF 目录下所有非权重文件(含 tokenizer/processor config)。"""
    skip = (".safetensors", "model.safetensors.index.json")
    for f in src_dir.iterdir():
        if f.is_file() and not (f.name.endswith(skip) or f.name in skip):
            shutil.copy2(f, dst_dir / f.name)


def patch_layer_config(d, num_layers):
    """递归减层数/截断 layer_types/关 mtp（hidden_size 保持不变）。"""
    if not isinstance(d, dict):
        return
    if "num_hidden_layers" in d:
        d["num_hidden_layers"] = num_layers
    if isinstance(d.get("layer_types"), list):
        d["layer_types"] = d["layer_types"][:num_layers]
    if "mtp_num_hidden_layers" in d:
        d["mtp_num_hidden_layers"] = 0


def _copy_modeling_code(dst_dir: Path) -> None:
    """把 qwen3_5_moe 建模 .py 拷入 config 目录, 供 trust_remote_code 解析。"""
    if MODELING_SRC.is_dir():
        for f in MODELING_SRC.glob("*.py"):
            shutil.copy2(f, dst_dir / f.name)


def _make_reduced_config() -> Path:
    """幂等生成 reduced config（减层数、保持 hidden=2048、无权重）。"""
    out = REDUCED_DIR / f"Qwen3VL-layers{LAYERS}"
    if (out / "config.json").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    copy_config_files(_HF_DIR, out)
    cfg_path = out / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    patch_layer_config(cfg, LAYERS)
    if "text_config" in cfg:
        patch_layer_config(cfg["text_config"], LAYERS)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    _copy_modeling_code(out)
    return out


def _generate_mock_data(text_length: int) -> Path:
    """幂等生成 mock 数据, 返回 json 路径。"""
    mock_file = f"mock_data_pic_num_1_1024x1024_textlen_{text_length}_num_samples_512.json"
    mock_path = MOCK_DIR / mock_file
    if mock_path.exists():
        return mock_path
    MOCK_DIR.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(MOCK_SCRIPT),
        "--tokenizer_path", str(_HF_DIR),
        "--pic_width", "1024", "--pic_height", "1024",
        "--num_pics", "1", "--text_length", str(text_length),
        "--num_samples", "512", "--save_dir", str(MOCK_DIR),
    ]
    subprocess.run(cmd, cwd=str(_PROJ_ROOT), check=True)
    return mock_path


@pytest.fixture(scope="session", autouse=True)
def _prepare_env_assets():
    """session 级: 预建 reduced config + 各序列长度 mock 数据。"""
    _make_reduced_config()
    for tl in (1024, 4096, 8192, 16384):
        _generate_mock_data(tl)


# ============================================================
# 配置构建（baseline / cce 路线 overrides）
# ============================================================
def deep_merge(base, override):
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _base_overrides(reduced_dir, dataset, dataset_dir, max_samples, use_deter):
    """通用覆盖: random 权重 + triton + mock 数据。"""
    return {
        "model": {
            "model_name_or_path": str(reduced_dir),
            "gdn_implementation": "triton",
            "causal_conv1d_implementation": "triton",
        },
        "training": {
            "load": "",                      # 不加载权重 -> 固定 seed 随机初始化
            "save": "",
            "train_iters": 1,                # 由用例覆盖
            "init_model_with_meta_device": True,
            "seed": SEED,
            "use_deter_comp": use_deter,
        },
        "data": {
            "dataset_param": {
                "preprocess_parameters": {"model_name_or_path": str(reduced_dir)},
                "basic_parameters": {
                    "dataset": dataset, "dataset_dir": dataset_dir,
                    "cache_dir": "./cache_dir/loss_e2e", "max_samples": max_samples,
                    "overwrite_cache": True,
                },
            },
        },
        "tools": {"profile": {"enable": False}, "memory_profile": {"enable": False}},
    }


def _baseline_chunk_overrides():
    """legacy 路线: chunk_loss + chunk_mbs=true。"""
    return {
        "features": {
            "enable_chunk_loss": True,
            "enable_chunk_mbs": True,
            "chunkmbs_plan": {
                "apply_modules": ["model.language_model.layers.{*}"],
                "chunk_mbs": 2,
                "batch_dim": 0,
            },
            "chunkloss_plan": {"apply_module": "lm_head", "chunk_size": 512},
        },
    }


def _cce_chunk_overrides(vt=4096, seq_chunk_size=None):
    """CCE 路线: cce 融合 + chunk_mbs=false + vt/seq_chunk 可调。"""
    return {
        "features": {
            "enable_chunk_loss": True,
            "enable_chunk_mbs": False,
            "chunkloss_plan": {
                "apply_module": "lm_head",
                "impl_type": "cce",
                "chunk_size": seq_chunk_size,
                "vocab_tile_size": vt,
            },
        },
    }


def _build_config(mode, reduced_dir, dataset, iters, mbs, seq_len, max_samples, use_deter,
                  vt, seq_chunk_size, extra_overrides=None):
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    ds_path = Path(dataset)
    merged = deep_merge(base, _base_overrides(reduced_dir, str(ds_path), str(ds_path.parent),
                                              max_samples, use_deter))
    if mode == "baseline":
        merged = deep_merge(merged, _baseline_chunk_overrides())
    elif mode == "cce":
        merged = deep_merge(merged, _cce_chunk_overrides(vt=vt, seq_chunk_size=seq_chunk_size))
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    if extra_overrides:
        merged = deep_merge(merged, extra_overrides)

    merged["training"]["train_iters"] = iters
    merged["training"]["micro_batch_size"] = mbs
    merged["data"]["dataset_param"]["basic_parameters"]["cutoff_len"] = seq_len

    fd, path = tempfile.mkstemp(suffix=f"_{mode}.yaml", dir=str(_TEST_DIR))
    with os.fdopen(fd, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return path


# ============================================================
# 训练执行与日志解析
# ============================================================
def _parse_losses(log_text):
    """解析日志为按 iteration 排序的 (loss, grad_norm) 列表。"""
    rows = {}
    for m in re.finditer(
        r"iteration\s*(\d+)[^\n]*?loss\s*[=:]\s*([-+0-9.eE]+)" \
        r"[^\n]*?grad norm\s*[=:]\s*([-+0-9.eE]+)", log_text):
        it = int(m.group(1))
        rows[it] = (float(m.group(2)), float(m.group(3)))
    ordered = [rows[it] for it in sorted(rows)]
    losses = [x[0] for x in ordered]
    grad_norms = [x[1] for x in ordered]
    return losses, grad_norms


def _run_trainer(config_path, npus, timeout_s):
    """torchrun 启动 trainer, 逐行读取并实时同步打印到屏幕, 返回 (rc, log_text)。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{Path(config_path).stem}.log"
    env = os.environ.copy()
    env.update({"NON_MEGATRON": "true", "TRITON_CACHE_DIR": str(_PROJ_ROOT / "triton_cache")})

    cmd = [
        "torchrun",
        "--nproc_per_node", str(npus), "--nnodes", "1",
        "--node_rank", "0", "--master_addr", "localhost", "--master_port", "6000",
        str(_PROJ_ROOT / "mindspeed_mm" / "fsdp" / "train" / "trainer.py"),
        config_path,
    ]
    proc = subprocess.Popen(cmd, cwd=str(_PROJ_ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            text=True, encoding="utf-8", errors="replace")
    lines = []
    try:
        for line in iter(proc.stdout.readline, ""):
            print(line, end="")
            sys.stdout.flush()
            lines.append(line)
        proc.stdout.close()
        proc.wait(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        rc = -1
    out = "".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(out)
    return rc, out


class _RunResult:
    def __init__(self, rc, losses, grad_norms, log_path):
        self.rc = rc
        self.losses = losses
        self.grad_norms = grad_norms
        self.log_path = log_path


# 相同配置训练结果缓存, 避免重复跑 baseline (如 seq_chunk 与 vt_8192 共用 8192 baseline)
_ROUTE_CACHE = {}


def _run_route(mode, iters, mbs, seq_len, max_samples, use_deter=False, vt=4096,
               seq_chunk_size=None, extra_overrides=None, npus=None, timeout_s=1800):
    # 带 extra_overrides 的用例(如 no_chunkloss)不可哈希, 不缓存
    cache_key = None if extra_overrides is not None else (
        mode, iters, mbs, seq_len, use_deter, vt, seq_chunk_size)
    if cache_key in _ROUTE_CACHE:
        return _ROUTE_CACHE[cache_key]

    reduced_dir = _make_reduced_config()
    dataset = _generate_mock_data(seq_len)
    cfg = _build_config(mode, reduced_dir, str(dataset), iters, mbs, seq_len, max_samples,
                        use_deter, vt, seq_chunk_size, extra_overrides)
    try:
        rc, log_text = _run_trainer(cfg, npus or NPUS, timeout_s)
    finally:
        try:
            os.unlink(cfg)
        except OSError:
            pass
    log_path = LOG_DIR / f"{Path(cfg).name}.log"
    if rc in (0, -1):
        losses, grad_norms = _parse_losses(log_text)
    else:
        losses, grad_norms = [], []
    result = _RunResult(rc, losses, grad_norms, log_path)
    if cache_key is not None:
        _ROUTE_CACHE[cache_key] = result
    return result


def _assert_no_nan(result):
    assert result.rc == 0, f"training failed rc={result.rc}, log: {result.log_path}"
    assert all(t == t and t != float("inf") for t in result.losses), \
        f"loss 含 NaN/Inf, log: {result.log_path}"
    assert len(result.losses) >= 1, f"未解析到 loss, log: {result.log_path}"


def _assert_compare(baseline: _RunResult, cce: _RunResult, warmup=1):
    assert baseline.rc == 0, f"baseline failed rc={baseline.rc}, log: {baseline.log_path}"
    assert cce.rc == 0, f"cce failed rc={cce.rc}, log: {cce.log_path}"
    assert not any((t != t or t == float("inf")) for t in baseline.losses), \
        f"baseline loss NaN, log: {baseline.log_path}"
    assert not any((t != t or t == float("inf")) for t in cce.losses), \
        f"cce loss NaN, log: {cce.log_path}"

    nb, nc = len(baseline.losses), len(cce.losses)
    assert nb == nc, f"步数不一致: baseline={nb}, cce={nc}"
    n = nb - warmup if nb - warmup > 0 else 0
    assert n > 0, f"有效对齐步数不足(总{nb}, warmup={warmup})"
    # 打印对比表格(断言前, fail 时也能直观看到 loss / grad norm 对比)
    print("[CCE对比] iter | baseline_loss | cce_loss | rel_err | baseline_gn | cce_gn")
    worst = 0.0
    for i in range(nb):
        b, c = baseline.losses[i], cce.losses[i]
        rel = abs(c - b) / abs(b) if abs(b) >= 1e-12 else float("nan")
        if i >= warmup and not (abs(b) < 1e-12) and (rel == rel):
            worst = max(worst, rel)
        gn_b = baseline.grad_norms[i] if i < len(baseline.grad_norms) else float("nan")
        gn_c = cce.grad_norms[i] if i < len(cce.grad_norms) else float("nan")
        mark = f"  <- warmup(skip)" if i < warmup else ""
        print(f"  {i:>4} | {b:.6e} | {c:.6e} | {rel:.3e} | {gn_b:.3e} | {gn_c:.3e}{mark}")
    print(f"[CCE对比] worst_rel(排除前{warmup}步) = {worst:.3e}  (容忍 {TOLERANCE})")
    assert worst < TOLERANCE, \
        f"loss 未对齐 worst_rel={worst:.2e}>={TOLERANCE}, logs: baseline={baseline.log_path}, cce={cce.log_path}"


def _max_samples(iters, mbs):
    """保证样本数 >= iters * GBS(≈ mbs*NPUS) 且留余量(不超过生成的512)。"""
    return min(max(int(iters * mbs * NPUS * 1.5), 128), 512)


# ============================================================
# 6 个集成用例
# ============================================================
class TestCceFusedIntegration:
    def test_e2e_cce_smoke(self):
        """纯 CCE 路线 5 步 fwd+bwd, 验证融合 loss 端到端可训练。"""
        r = _run_route("cce", iters=5, mbs=2, seq_len=8192,
                       max_samples=_max_samples(5, 2), use_deter=False,
                       vt=4096, npus=NPUS)
        _assert_no_nan(r)

    def test_e2e_compare_default(self):
        """核心: baseline vs cce 常规 10 步 loss 对齐。"""
        b = _run_route("baseline", iters=10, mbs=2, seq_len=16384,
                       max_samples=_max_samples(10, 2), use_deter=False, npus=NPUS)
        c = _run_route("cce", iters=10, mbs=2, seq_len=16384,
                       max_samples=_max_samples(10, 2), use_deter=False,
                       vt=4096, npus=NPUS)
        _assert_compare(b, c, warmup=1)

    def test_e2e_no_chunkloss(self):
        """回归: enable_chunk_loss=false 走 plain logits 路径不报错。"""
        r = _run_route("baseline", iters=5, mbs=1, seq_len=1024,
                       max_samples=_max_samples(5, 1), use_deter=False,
                       extra_overrides={
                           "features": {"enable_chunk_loss": False, "enable_chunk_mbs": False}
                       }, npus=NPUS)
        _assert_no_nan(r)

    def test_e2e_compare_seq_chunk(self):
        """cce 外层序列分片(chunk_size=2048<N)应仍与 baseline 对齐。"""
        b = _run_route("baseline", iters=10, mbs=2, seq_len=8192,
                       max_samples=_max_samples(10, 2), use_deter=False, npus=NPUS)
        c = _run_route("cce", iters=10, mbs=2, seq_len=8192,
                       max_samples=_max_samples(10, 2), use_deter=False,
                       vt=4096, seq_chunk_size=2048, npus=NPUS)
        _assert_compare(b, c, warmup=1)

    def test_e2e_compare_vt_8192(self):
        """cce 大 vt=8192 应与 baseline 对齐。"""
        b = _run_route("baseline", iters=10, mbs=2, seq_len=8192,
                       max_samples=_max_samples(10, 2), use_deter=False, npus=NPUS)
        c = _run_route("cce", iters=10, mbs=2, seq_len=8192,
                       max_samples=_max_samples(10, 2), use_deter=False,
                       vt=8192, npus=NPUS)
        _assert_compare(b, c, warmup=1)

    def test_e2e_compare_deter(self):
        """确定性计算开启下 baseline vs cce 对齐。"""
        b = _run_route("baseline", iters=5, mbs=2, seq_len=4096,
                       max_samples=_max_samples(5, 2), use_deter=True, npus=NPUS)
        c = _run_route("cce", iters=5, mbs=2, seq_len=4096,
                       max_samples=_max_samples(5, 2), use_deter=True,
                       vt=4096, npus=NPUS)
        _assert_compare(b, c, warmup=1)
