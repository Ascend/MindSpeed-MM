# coding=utf-8
"""CCE fused 算子单元测试。

目标：对 chunk_loss_cce_fused（Triton/NPU 融合算子）用固定种子的合成输入，与两条独立
参考做黑盒对比（loss + grad_hidden + grad_weight）：
  1. legacy 路线: ChunkLoss + calculate_lm_loss（loss_func.py 中 CCE 替换的那套, 语义一致）
  2. 纯 torch golden: F.linear + F.cross_entropy（float logits, 独立更高精度基准)

数据策略：纯合成随机（torch.manual_seed 固定），不使用任何权重文件 / 数据文件。
环境：假设 torch_npu + Triton 存在；无 NPU 时整模块 skip。

运行：
  pytest tests/ut/loss/test_cce_fused.py -v
"""
import gc
import pytest

try:
    import torch  # noqa: F401
    import torch.nn.functional as F  # noqa: F401
    import torch_npu  # noqa: F401
    DEVICE = "npu"
    _HAS_NPU = torch.npu.is_available()
except Exception:  # pragma: no cover - 环境守卫
    DEVICE = "cuda"
    _HAS_NPU = False

if not _HAS_NPU:
    pytest.skip("torch / torch_npu / NPU not available", allow_module_level=True)

from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss import (
    ChunkLoss,
    calculate_lm_loss,
)
from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_fused import (
    chunk_loss_cce_fused,
)


SEED = 42
IGNORE_INDEX = -100

LOSS_REL_THRESH = 1e-4
GH_REL_THRESH = 1e-3
GW_REL_THRESH = 5e-2


# ============================================================
# 输入生成与指标
# ============================================================
def make_inputs(B, S, H, V, ignore_ratio=0.03, all_ignore=False, single_valid=False):
    """构造合成输入。labels 为原始(未 shift)，返回已 shift 的 shift_labels。"""
    torch.manual_seed(SEED)
    h = torch.randn(B, S, H, device=DEVICE, dtype=torch.bfloat16)
    w = (torch.randn(V, H, device=DEVICE, dtype=torch.float32) * 0.02).bfloat16()
    labels = torch.randint(0, V, (B, S), device=DEVICE)

    if all_ignore:
        labels[:] = IGNORE_INDEX
    elif single_valid:
        # 有效 token 放在 labels[0,1]; shift 后 shift_labels[0,0] 仍有效(避免失效导致假失败)
        labels[:] = IGNORE_INDEX
        labels[0, 1] = torch.randint(0, V, (1,)).item()
    else:
        n_ignore = int(B * S * ignore_ratio)
        if n_ignore > 0:
            idx = torch.randperm(B * S, device=DEVICE)[:n_ignore]
            labels.reshape(-1)[idx] = IGNORE_INDEX

    # 与 loss_func.get_loss_func_params 一致: pad + shift
    labels_padded = F.pad(labels, (0, 1), value=IGNORE_INDEX)
    shift_labels = labels_padded[..., 1:].contiguous()
    return h, w, shift_labels


def rel_err(x, ref):
    x = x.float().cpu()
    ref = ref.float().cpu()
    return ((x - ref).norm() / (ref.norm() + 1e-12)).item()


def has_nan(*tensors):
    return any(torch.isnan(t).any().item() for t in tensors)


def _alpha(shift_labels):
    return (shift_labels > IGNORE_INDEX).sum()


# ============================================================
# 三条路线的封装（all 输出归一化平均 loss + 梯度）
# ============================================================
def run_fused(hidden_states, head_weight, shift_labels, vt=4096, seq_chunk_size=None):
    """CCE fused 路线: chunk_loss_cce_fused + alpha 归一化。"""
    h = hidden_states.detach().clone().requires_grad_(True)
    w = head_weight.detach().clone().requires_grad_(True)
    alpha = _alpha(shift_labels)
    loss = chunk_loss_cce_fused(
        h, w, shift_labels, vt=vt, ignore_index=IGNORE_INDEX,
        seq_chunk_size=seq_chunk_size,
    )
    if alpha.item() > 0:
        loss = loss / alpha
    loss.backward()
    torch.npu.synchronize()
    return (loss.detach().float().cpu(), h.grad.detach().float().cpu(),
            w.grad.detach().float().cpu())


def run_legacy(hidden_states, head_weight, shift_labels, chunk_size=1024):
    """legacy 路线: ChunkLoss + calculate_lm_loss + alpha 归一化(与 loss_func default 一致)。"""
    h = hidden_states.detach().clone().requires_grad_(True)
    w = head_weight.detach().clone().requires_grad_(True)
    alpha = _alpha(shift_labels)
    one = torch.tensor(1.0, device=hidden_states.device)
    cs = min(chunk_size, shift_labels.shape[1])
    label_chunks = torch.split(shift_labels, cs, dim=1)
    loss_kwargs_chunks = [
        dict(shift_labels=lc, alpha=one, ignore_index=IGNORE_INDEX, reduction="sum")
        for lc in label_chunks
    ]
    loss = ChunkLoss.apply(h, w, None, calculate_lm_loss, loss_kwargs_chunks, cs)
    if alpha.item() > 0:
        loss = loss / alpha
    loss.backward()
    torch.npu.synchronize()
    return (loss.detach().float().cpu(), h.grad.detach().float().cpu(),
            w.grad.detach().float().cpu())


def run_torch_golden(hidden_states, head_weight, shift_labels):
    """纯 torch golden: float logits 上 F.cross_entropy, 独立更高精度基准。"""
    h = hidden_states.detach().clone().requires_grad_(True)
    w = head_weight.detach().clone().requires_grad_(True)
    alpha = _alpha(shift_labels)
    logits = F.linear(h, w).float()
    logits = logits.reshape(-1, logits.shape[-1])
    labels_flat = shift_labels.reshape(-1)
    loss = F.cross_entropy(logits, labels_flat, ignore_index=IGNORE_INDEX, reduction="sum")
    if alpha.item() > 0:
        loss = loss / alpha
    loss.backward()
    torch.npu.synchronize()
    return (loss.detach().float().cpu(), h.grad.detach().float().cpu(),
            w.grad.detach().float().cpu())


# ============================================================
# 用例
# ============================================================
class TestCceFusedAgainstLegacy:
    """核心: fused 对 legacy 的对齐(loss + 梯度)。"""

    @pytest.mark.parametrize("B,S,H,V", [
        (1, 8, 64, 1000),
        (2, 16, 128, 2560),
        (4, 32, 256, 4096),
    ])
    def test_loss_and_grads_default(self, B, S, H, V):
        h, w, shift_labels = make_inputs(B, S, H, V, ignore_ratio=0.0)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=4096)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=S)

        assert not has_nan(lf, ghf, gwf), "fused 输出含 NaN"
        assert rel_err(lf, ll) < LOSS_REL_THRESH, f"loss_rel={rel_err(lf, ll):.2e}"
        assert rel_err(ghf, ghl) < GH_REL_THRESH, f"gh_rel={rel_err(ghf, ghl):.2e}"
        assert rel_err(gwf, gwl) < GW_REL_THRESH, f"gw_rel={rel_err(gwf, gwl):.2e}"

    @pytest.mark.parametrize("ignore_ratio", [0.3, 0.7])
    def test_with_ignore(self, ignore_ratio):
        h, w, shift_labels = make_inputs(2, 16, 128, 2560, ignore_ratio=ignore_ratio)
        lf, ghf, gwf = run_fused(h, w, shift_labels)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=16)

        assert not has_nan(lf, ghf, gwf)
        assert rel_err(lf, ll) < LOSS_REL_THRESH
        assert rel_err(ghf, ghl) < GH_REL_THRESH
        assert rel_err(gwf, gwl) < GW_REL_THRESH

    def test_large_vocab_multi_tile(self):
        """真实规模词表 V=151936(>vt=4096), 多 vocab tile。"""
        h, w, shift_labels = make_inputs(1, 32, 512, 151936, ignore_ratio=0.03)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=4096)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=32)

        assert not has_nan(lf, ghf, gwf)
        assert rel_err(lf, ll) < LOSS_REL_THRESH
        assert rel_err(ghf, ghl) < GH_REL_THRESH
        assert rel_err(gwf, gwl) < GW_REL_THRESH

    @pytest.mark.parametrize("vt", [256, 1024, 4096, 8192])
    def test_vt_variants(self, vt):
        """不同 vocab tile 大小, 覆盖单 tile 与多 tile 调度。"""
        h, w, shift_labels = make_inputs(1, 16, 128, 8192, ignore_ratio=0.1)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=vt)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=16)

        assert not has_nan(lf, ghf, gwf)
        assert rel_err(lf, ll) < LOSS_REL_THRESH
        assert rel_err(ghf, ghl) < GH_REL_THRESH
        assert rel_err(gwf, gwl) < GW_REL_THRESH

    def test_seq_chunk_size(self):
        """fused 外层序列分片(seq_chunk_size<N)应等价于不分片。"""
        h, w, shift_labels = make_inputs(1, 64, 256, 2048, ignore_ratio=0.1)
        single = run_fused(h, w, shift_labels, vt=1024, seq_chunk_size=None)
        chunked = run_fused(h, w, shift_labels, vt=1024, seq_chunk_size=16)

        assert not has_nan(*chunked)
        assert rel_err(chunked[0], single[0]) < LOSS_REL_THRESH, "seq_chunk loss 与单段不一致"
        assert rel_err(chunked[1], single[1]) < GH_REL_THRESH
        assert rel_err(chunked[2], single[2]) < GW_REL_THRESH

    def test_flat_input_equiv_3d(self):
        """2D 展平输入 (N,H)/(N,) 应等价于 reshape 自 (B,S,H)/(B,S)。"""
        B, S, H, V = 2, 16, 128, 4096
        h, w, shift_labels = make_inputs(B, S, H, V, ignore_ratio=0.0)
        # fused 内部会把 3D reshape 成 2D; 直接给 2D 应得到相同结果
        h2 = h.reshape(-1, H)
        l2 = shift_labels.reshape(-1)
        flat = run_fused(h2, w, l2, vt=1024)
        cube = run_fused(h, w, shift_labels, vt=1024)

        # cube 的 grad 是 3D (B,S,H), flat 是 2D (N,H), 统一展平后比较
        cube_flat = (cube[0], cube[1].reshape(-1, H), cube[2])
        assert rel_err(flat[0], cube_flat[0]) < LOSS_REL_THRESH
        assert rel_err(flat[1], cube_flat[1]) < GH_REL_THRESH
        assert rel_err(flat[2], cube_flat[2]) < GW_REL_THRESH


class TestCceFusedAgainstTorchGolden:
    """fused 对纯 torch float golden 的 sanity 检查(宽松)。"""

    def test_loss_and_grads(self):
        h, w, shift_labels = make_inputs(1, 16, 128, 4096, ignore_ratio=0.05)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=4096)
        lg, ghg, gwg = run_torch_golden(h, w, shift_labels)

        assert not has_nan(lf, ghf, gwf)
        # bf16 输入 vs float golden, 允许更大余量
        assert rel_err(lf, lg) < 5e-2
        assert rel_err(ghf, ghg) < 1e-1
        assert rel_err(gwf, gwg) < 1e-1


class TestCceFusedEdgeCases:
    """边界/健壮性。"""

    def test_all_ignore(self):
        """全 ignore(alpha=0), 不得 NaN/Inf。"""
        h, w, shift_labels = make_inputs(1, 8, 64, 1000, all_ignore=True)
        lf, ghf, gwf = run_fused(h, w, shift_labels)
        assert not has_nan(lf, ghf, gwf), "全 ignore 场景出现 NaN"

    def test_single_valid_token(self):
        """仅 1 个有效 token。"""
        h, w, shift_labels = make_inputs(1, 8, 64, 1000, single_valid=True)
        lf, ghf, gwf = run_fused(h, w, shift_labels)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=8)
        assert not has_nan(lf, ghf, gwf)
        assert rel_err(lf, ll) < LOSS_REL_THRESH

    def test_small_n_lt_vt(self):
        """N < vt 退化场景。"""
        h, w, shift_labels = make_inputs(1, 4, 64, 4096, ignore_ratio=0.0)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=4096)
        ll, ghl, gwl = run_legacy(h, w, shift_labels, chunk_size=4)
        assert not has_nan(lf, ghf, gwf)
        assert rel_err(lf, ll) < LOSS_REL_THRESH

    def test_finite_output(self):
        """输出均为有限值。"""
        h, w, shift_labels = make_inputs(4, 32, 256, 151936, ignore_ratio=0.2)
        lf, ghf, gwf = run_fused(h, w, shift_labels, vt=4096)
        for t in (lf, ghf, gwf):
            assert torch.isfinite(t).all(), "输出含 Inf/NaN"


@pytest.fixture(autouse=True)
def _cleanup():
    yield
    gc.collect()
    torch.npu.empty_cache()
