"""Losslessness of the ignore_index skip fast paths in the FSDP chunk loss.

Both ``chunk_loss`` (ChunkLoss route) and ``chunk_loss_cce_fused`` (CCE route) drop
``ignore_index`` tokens *before* the lm_head projection. For the scalar-normalized sum
reduction this is numerically exact: masked tokens contribute exactly 0 to both loss and
gradient, and boolean-mask indexing is differentiable (its backward scatters gradients to
the valid rows and zeros the dropped ones). These tests pin loss + gradient equality
against the dense (non-filtered) path, which is what makes the optimization safe to ship.

The ChunkLoss tests are device-agnostic (pure PyTorch). The CCE tests require Triton + NPU
and self-skip otherwise.
"""
import pytest
import torch

from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss import (
    ChunkLoss,
    chunk_loss,
    calculate_lm_loss,
)
from tests.ut_fsdp.utils.utils import judge_expression


def _build_chunks(shift_labels, alpha, chunk_size, ignore_index=-100):
    return [
        {"shift_labels": c, "ignore_index": ignore_index, "reduction": "sum",
         "alpha": alpha, "chunk_size": chunk_size}
        for c in torch.split(shift_labels, chunk_size, dim=1)
    ]


def _run_chunkloss(fn, hidden0, weight0, labels, alpha, chunk_size):
    h = hidden0.clone().detach().requires_grad_(True)
    w = weight0.clone().detach().requires_grad_(True)
    chunks = _build_chunks(labels, alpha, chunk_size)
    loss = fn(h, w, None, calculate_lm_loss, chunks, chunk_size)
    loss.backward()
    return loss.detach(), h.grad, w.grad


class TestChunkLossSkipIgnored:
    """ChunkLoss route: filter ignore_index tokens then re-chunk == dense path."""

    device = torch.device(get_device_type())
    batch = 1
    seq_len = 2048
    hidden_dim = 512
    vocab_size = 4096
    chunk_size = 512
    ignore_index = -100

    def _inputs(self, mask_ratio):
        torch.manual_seed(0)
        labels = torch.randint(0, self.vocab_size, (self.batch, self.seq_len), device=self.device)
        n_mask = int(round(mask_ratio * self.seq_len))
        if n_mask:
            idx = torch.randperm(self.seq_len, device=self.device)[:n_mask]
            labels[0, idx] = self.ignore_index
        num_valid = int((labels != self.ignore_index).sum())
        # loss_type 'default' normalization: alpha == number of valid tokens (scalar).
        alpha = torch.tensor(float(max(num_valid, 1)), device=self.device)
        hidden = torch.randn(self.batch, self.seq_len, self.hidden_dim,
                             device=self.device, dtype=torch.float32)
        weight = torch.randn(self.vocab_size, self.hidden_dim,
                             device=self.device, dtype=torch.float32) * 0.02
        return hidden, weight, labels, alpha, num_valid

    @pytest.mark.parametrize("mask_ratio", [0.5, 0.9])
    def test_skip_matches_dense(self, mask_ratio):
        hidden, weight, labels, alpha, num_valid = self._inputs(mask_ratio)
        # The fast path only fires for a single-sample, partially-masked, scalar-sum batch.
        # Assert the precondition so this test cannot silently go vacuous.
        judge_expression(0 < num_valid < self.seq_len)

        ld, ghd, gwd = _run_chunkloss(ChunkLoss.apply, hidden, weight, labels, alpha, self.chunk_size)
        ls, ghs, gws = _run_chunkloss(chunk_loss, hidden, weight, labels, alpha, self.chunk_size)

        # Loss agrees up to fp32 summation-order noise (~1e-6 on a loss of O(10)); the skip
        # path re-chunks the valid tokens so the reduction order differs from the dense path.
        judge_expression(torch.allclose(ls, ld, rtol=1e-4, atol=1e-4))
        # Gradients are otherwise bit-exact (dropped rows carry a genuine 0 gradient).
        judge_expression(torch.allclose(ghs, ghd, rtol=1e-4, atol=1e-5))
        judge_expression(torch.allclose(gws, gwd, rtol=1e-4, atol=1e-5))

    def test_no_mask_is_noop(self):
        # Nothing masked -> fast path must NOT fire; result is still identical to dense.
        hidden, weight, labels, alpha, num_valid = self._inputs(0.0)
        judge_expression(num_valid == self.seq_len)

        ld, ghd, gwd = _run_chunkloss(ChunkLoss.apply, hidden, weight, labels, alpha, self.chunk_size)
        ls, ghs, gws = _run_chunkloss(chunk_loss, hidden, weight, labels, alpha, self.chunk_size)

        judge_expression(torch.allclose(ls, ld, rtol=1e-4, atol=1e-4))
        judge_expression(torch.allclose(ghs, ghd, rtol=1e-4, atol=1e-5))
        judge_expression(torch.allclose(gws, gwd, rtol=1e-4, atol=1e-5))

    @pytest.mark.parametrize("mask_ratio", [0.5, 0.9])
    def test_skip_matches_dense_multibatch(self, mask_ratio):
        batch = 4
        torch.manual_seed(0)
        labels = torch.randint(0, self.vocab_size, (batch, self.seq_len), device=self.device)
        flat = labels.reshape(-1)
        n_mask = int(round(mask_ratio * flat.numel()))
        idx = torch.randperm(flat.numel(), device=self.device)[:n_mask]
        flat[idx] = self.ignore_index
        labels = flat.reshape(batch, self.seq_len)
        num_valid = int((labels != self.ignore_index).sum())
        # Assert both the multi-sample shape and the partial-mask precondition of the fast path.
        judge_expression(batch > 1 and 0 < num_valid < flat.numel())
        alpha = torch.tensor(float(max(num_valid, 1)), device=self.device)
        hidden = torch.randn(batch, self.seq_len, self.hidden_dim,
                             device=self.device, dtype=torch.float32)
        weight = torch.randn(self.vocab_size, self.hidden_dim,
                             device=self.device, dtype=torch.float32) * 0.02

        ld, ghd, gwd = _run_chunkloss(ChunkLoss.apply, hidden, weight, labels, alpha, self.chunk_size)
        ls, ghs, gws = _run_chunkloss(chunk_loss, hidden, weight, labels, alpha, self.chunk_size)

        judge_expression(torch.allclose(ls, ld, rtol=1e-4, atol=1e-4))
        judge_expression(torch.allclose(ghs, ghd, rtol=1e-4, atol=1e-5))
        judge_expression(torch.allclose(gws, gwd, rtol=1e-4, atol=1e-5))


class TestChunkLossCceSkipIgnored:
    """CCE route: pre-filter ignore_index tokens before the vocab-tile kernel == dense path.

    Requires Triton + NPU (the CCE kernels are NPU/Triton only); self-skips elsewhere.
    """

    seq_len = 2048
    hidden_dim = 512
    vocab_size = 4096
    vt = 2048
    ignore_index = -100

    def _run_dense(self, apply_fn, hidden0, weight0, labels):
        h = hidden0.clone().detach().requires_grad_(True)
        w = weight0.clone().detach().requires_grad_(True)
        loss = apply_fn(h.reshape(-1, self.hidden_dim), w,
                        labels.reshape(-1), self.vt, self.ignore_index)
        loss.backward()
        return loss.detach(), h.grad, w.grad

    def _run_skip(self, fused_fn, hidden0, weight0, labels):
        h = hidden0.clone().detach().requires_grad_(True)
        w = weight0.clone().detach().requires_grad_(True)
        loss = fused_fn(h, w, labels, vt=self.vt, ignore_index=self.ignore_index)
        loss.backward()
        return loss.detach(), h.grad, w.grad

    @pytest.mark.parametrize("mask_ratio", [0.5, 0.9])
    def test_cce_skip_matches_dense(self, mask_ratio):
        pytest.importorskip("triton")
        if get_device_type() != "npu":
            pytest.skip("CCE kernels require NPU")

        from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_fused import chunk_loss_cce_fused
        from mindspeed_mm.fsdp.features.memory.chunkloss.chunkloss_cce_kernels import ChunkLossCceFused

        device = torch.device("npu")
        torch.manual_seed(0)
        labels = torch.randint(0, self.vocab_size, (1, self.seq_len), device=device)
        n_mask = int(round(mask_ratio * self.seq_len))
        idx = torch.randperm(self.seq_len, device=device)[:n_mask]
        labels[0, idx] = self.ignore_index
        num_valid = int((labels != self.ignore_index).sum())
        judge_expression(0 < num_valid < self.seq_len)

        hidden = torch.randn(1, self.seq_len, self.hidden_dim, device=device, dtype=torch.bfloat16)
        weight = (torch.randn(self.vocab_size, self.hidden_dim, device=device,
                              dtype=torch.float32) * 0.02).bfloat16()

        ld, ghd, gwd = self._run_dense(ChunkLossCceFused.apply, hidden, weight, labels)
        ls, ghs, gws = self._run_skip(chunk_loss_cce_fused, hidden, weight, labels)

        # Filtering removes only masked tokens (0 loss/grad in-kernel), so results match up to
        # bf16 reduction-order noise.
        judge_expression(torch.allclose(ls, ld, rtol=1e-2, atol=1e-2))
        judge_expression(torch.allclose(ghs.float(), ghd.float(), rtol=1e-2, atol=1e-2))
        judge_expression(torch.allclose(gws.float(), gwd.float(), rtol=1e-2, atol=1e-2))
