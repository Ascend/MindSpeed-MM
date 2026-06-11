import pytest
import torch
from unittest.mock import patch, MagicMock

from mindspeed_mm.fsdp.utils.device import get_device_type
from mindspeed_mm.fsdp.loss.loss_func import (
    build_loss_func,
    get_loss_func_params,
)
from mindspeed_mm.fsdp.utils.constants import AVG_PER_STEP_TOKEN_NUM
from tests.ut_fsdp.utils.utils import judge_expression


@pytest.fixture(autouse=True)
def mock_parallel_state():
    with patch("mindspeed_mm.fsdp.loss.loss_func.get_parallel_state") as mock_ps_func:
        mock_ps = MagicMock()
        mock_ps.is_cp_enable.return_value = False
        mock_ps_func.return_value = mock_ps
        yield mock_ps_func


@pytest.fixture(autouse=True)
def mock_torch_distributed():
    """Mock torch.distributed """
    with patch("torch.distributed.all_reduce") as mock_all_reduce:
        def fake_all_reduce(tensor, op):
            if op == torch.distributed.ReduceOp.AVG:
                tensor.div_(1)
            return
        mock_all_reduce.side_effect = fake_all_reduce
        yield mock_all_reduce


class TestBuildLossFuncErrorHandling:
    """
    Test error handling in build_loss_func
    """
    device = torch.device(get_device_type())

    def test_missing_labels_error(self):
        """Test error when labels are not provided anywhere"""
        loss_func = build_loss_func(
            loss_type="default",
            chunk_size=256,
        )

        hidden_states = torch.randn(2, 512, 128, device=self.device)
        head_weight = torch.randn(1000, 128, device=self.device)

        with pytest.raises(ValueError, match="labels must be provided"):
            loss_func(hidden_states, head_weight, None)

    def test_invalid_loss_type(self):
        """Test error when invalid loss_type is provided"""
        labels = torch.randint(0, 1000, (2, 512), device=self.device)

        with pytest.raises(NotImplementedError, match="is not implemented"):
            get_loss_func_params(
                labels=labels,
                loss_type="invalid_type",
            )

    def test_per_token_loss_missing_avg_token_num(self):
        """Test error when per_token_loss is used without avg_per_step_token_num"""
        labels = torch.randint(0, 1000, (2, 512), device=self.device)

        with pytest.raises(KeyError, match="per_token_loss must use PrefetchGradAccDataLoader"):
            get_loss_func_params(
                labels=labels,
                loss_type="per_token_loss",
            )


class TestBuildLossFuncConsistency:
    """
    Test consistency between chunk and non-chunk loss results
    """
    device = torch.device(get_device_type())
    batch_size = 2
    seq_len = 512
    hidden_dim = 128
    vocab_size = 1000
    valid_lens = [300, 400]
    PAD_TOKEN = -100

    def _create_dummy_inputs(self):
        """Create dummy inputs for consistency testing"""
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, requires_grad=True
        )
        head_weight = torch.randn(
            self.vocab_size, self.hidden_dim,
            device=self.device, requires_grad=True
        )
        labels = torch.randint(0, self.vocab_size,
                              (self.batch_size, self.seq_len),
                              device=self.device)
        for b_idx in range(self.batch_size):
            valid_len = self.valid_lens[b_idx]
            labels[b_idx, valid_len:] = self.PAD_TOKEN
        return hidden_states, head_weight, labels

    def _transfer_data_to_packing_data(self, hidden_states, labels):
        packed_hidden_list = []
        packed_label_list = []
        for b_idx in range(self.batch_size):
            valid_len = self.valid_lens[b_idx]
            packed_hidden_list.append(hidden_states[b_idx, :valid_len, :])
            shift_label = labels[b_idx, :valid_len]
            shift_label[0] = -100
            packed_label_list.append(shift_label)

        packed_hidden_states = torch.cat(packed_hidden_list, dim=0)[None, :]
        packed_labels = torch.cat(packed_label_list, dim=0)[None, :]

        cu_seqlens = [0]
        cur = 0
        for length in self.valid_lens:
            cur += length
            cu_seqlens.append(cur)
        cu_seqlens = torch.tensor(cu_seqlens, device=self.device, dtype=torch.int32)[None, :]

        return packed_hidden_states, packed_labels, cu_seqlens

    def test_chunk_vs_non_chunk_default_loss(self):
        """Test that chunk and non-chunk loss produce similar results for default loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()

        # Compute chunk loss
        chunk_loss_func = build_loss_func(
            loss_type="default",
            chunk_size=self.seq_len,  # Use full seq_len as chunk size
        )
        chunk_loss = chunk_loss_func(hidden_states.clone().detach().requires_grad_(True),
                                     head_weight.clone().detach().requires_grad_(True),
                                     None, labels)

        # Compute non-chunk loss (using logits)
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight
        ).view(self.batch_size, self.seq_len, self.vocab_size)

        non_chunk_loss_func = build_loss_func(
            loss_type="default",
            chunk_size=None,
        )
        non_chunk_loss = non_chunk_loss_func(logits, labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(chunk_loss, torch.Tensor))
        judge_expression(isinstance(non_chunk_loss, torch.Tensor))
        judge_expression(chunk_loss.dim() == 0)
        judge_expression(non_chunk_loss.dim() == 0)
        judge_expression(torch.allclose(chunk_loss, non_chunk_loss, rtol=1e-5, atol=1e-6))

    def test_chunk_vs_non_chunk_per_sample_loss(self):
        """Test that chunk and non-chunk loss produce similar results for per sample loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()

        # Compute chunk loss
        chunk_loss_func = build_loss_func(
            loss_type="per_sample_loss",
            chunk_size=self.seq_len,  # Use full seq_len as chunk size
        )
        chunk_loss = chunk_loss_func(hidden_states.clone().detach().requires_grad_(True),
                                     head_weight.clone().detach().requires_grad_(True),
                                     None, labels)

        # Compute non-chunk loss (using logits)
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight
        ).view(self.batch_size, self.seq_len, self.vocab_size)

        non_chunk_loss_func = build_loss_func(
            loss_type="per_sample_loss",
            chunk_size=None,
        )
        non_chunk_loss = non_chunk_loss_func(logits, labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(chunk_loss, torch.Tensor))
        judge_expression(isinstance(non_chunk_loss, torch.Tensor))
        judge_expression(chunk_loss.dim() == 0)
        judge_expression(non_chunk_loss.dim() == 0)
        judge_expression(torch.allclose(chunk_loss, non_chunk_loss, rtol=1e-5, atol=1e-6))

    def test_chunk_vs_non_chunk_per_token_loss(self):
        """Test that chunk and non-chunk loss produce similar results for per sample loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()
        avg_per_step_token_num = torch.tensor(sum(self.valid_lens), device=self.device)
        # Compute chunk loss
        chunk_loss_func = build_loss_func(
            loss_type="per_token_loss",
            chunk_size=self.seq_len,  # Use full seq_len as chunk size
            **{AVG_PER_STEP_TOKEN_NUM: avg_per_step_token_num},
        )
        chunk_loss = chunk_loss_func(hidden_states.clone().detach().requires_grad_(True),
                                     head_weight.clone().detach().requires_grad_(True),
                                     None, labels)

        # Compute non-chunk loss (using logits)
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight
        ).view(self.batch_size, self.seq_len, self.vocab_size)

        non_chunk_loss_func = build_loss_func(
            loss_type="per_token_loss",
            chunk_size=None,
            **{AVG_PER_STEP_TOKEN_NUM: avg_per_step_token_num},
        )
        non_chunk_loss = non_chunk_loss_func(logits, labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(chunk_loss, torch.Tensor))
        judge_expression(isinstance(non_chunk_loss, torch.Tensor))
        judge_expression(chunk_loss.dim() == 0)
        judge_expression(non_chunk_loss.dim() == 0)
        judge_expression(torch.allclose(chunk_loss, non_chunk_loss, rtol=1e-5, atol=1e-6))

    def test_packing_vs_non_packing_default_loss(self):
        """Test that chunk and non-chunk loss produce similar results for default loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()
        packed_hidden_states, packed_labels, _ = self._transfer_data_to_packing_data(hidden_states, labels)

        non_chunk_loss_func = build_loss_func(
            loss_type="default",
            chunk_size=None,
        )
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight
        ).view(self.batch_size, self.seq_len, self.vocab_size)
        loss = non_chunk_loss_func(logits, labels)

        packed_logits = torch.nn.functional.linear(
            packed_hidden_states.view(-1, self.hidden_dim),
            head_weight
        ).view(1, -1, self.vocab_size)
        packing_loss = non_chunk_loss_func(packed_logits, packed_labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(loss, torch.Tensor))
        judge_expression(isinstance(packing_loss, torch.Tensor))
        judge_expression(loss.dim() == 0)
        judge_expression(packing_loss.dim() == 0)
        judge_expression(torch.allclose(loss, packing_loss, rtol=1e-5, atol=1e-6))

    def test_packing_vs_non_packing_per_token_loss(self):
        """Test that chunk and non-chunk loss produce similar results for per token loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()
        packed_hidden_states, packed_labels, _ = self._transfer_data_to_packing_data(hidden_states, labels)
        avg_per_step_token_num = (packed_labels > -1).sum()

        non_chunk_loss_func = build_loss_func(
            loss_type="per_token_loss",
            chunk_size=None,
            **{AVG_PER_STEP_TOKEN_NUM: avg_per_step_token_num},
        )
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight,
        ).view(self.batch_size, self.seq_len, self.vocab_size)
        loss = non_chunk_loss_func(logits, labels)

        packed_logits = torch.nn.functional.linear(
            packed_hidden_states.view(-1, self.hidden_dim),
            head_weight,
        ).view(1, -1, self.vocab_size)
        packing_loss = non_chunk_loss_func(packed_logits, packed_labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(loss, torch.Tensor))
        judge_expression(isinstance(packing_loss, torch.Tensor))
        judge_expression(loss.dim() == 0)
        judge_expression(packing_loss.dim() == 0)
        judge_expression(torch.allclose(loss, packing_loss, rtol=1e-5, atol=1e-6))

    def test_packing_vs_non_packing_per_sample_loss(self):
        """Test that chunk and non-chunk loss produce similar results for per sample loss type"""
        hidden_states, head_weight, labels = self._create_dummy_inputs()
        packed_hidden_states, packed_labels, cu_seqlens = self._transfer_data_to_packing_data(hidden_states, labels)

        non_chunk_loss_func = build_loss_func(
            loss_type="per_sample_loss",
            chunk_size=None,
        )
        logits = torch.nn.functional.linear(
            hidden_states.view(-1, self.hidden_dim),
            head_weight,
        ).view(self.batch_size, self.seq_len, self.vocab_size)
        loss = non_chunk_loss_func(logits, labels)

        non_chunk_packed_loss_func = build_loss_func(
            loss_type="per_sample_loss",
            chunk_size=None,
            cu_seqlens=cu_seqlens,
        )
        packed_logits = torch.nn.functional.linear(
            packed_hidden_states.view(-1, self.hidden_dim),
            head_weight,
        ).view(1, -1, self.vocab_size)
        packing_loss = non_chunk_packed_loss_func(packed_logits, packed_labels)

        # Both should produce similar scalar losses
        judge_expression(isinstance(loss, torch.Tensor))
        judge_expression(isinstance(packing_loss, torch.Tensor))
        judge_expression(loss.dim() == 0)
        judge_expression(packing_loss.dim() == 0)
        judge_expression(torch.allclose(loss, packing_loss, rtol=1e-5, atol=1e-6))
