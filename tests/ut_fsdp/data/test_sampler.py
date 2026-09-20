"""Unit tests for the ``BaseRandomBatchSampler`` small-dataset guard (issue #618).

``BaseRandomBatchSampler`` divides by ``active_total_samples`` (total samples
minus the dropped tail) when starting an epoch. On tiny datasets whose size is
smaller than the global batch (``micro_batch_size x num_replicas``) the active
total drops to zero and iteration crashed with ``ZeroDivisionError``. The
sampler now raises a descriptive ``ValueError`` instead. These tests pin the
guard and the unchanged iteration behaviour on small datasets.

Passing ``num_replicas`` / ``rank`` explicitly means these tests do not
require an initialized ``torch.distributed`` process group.
"""

import pytest

pytest.importorskip("torch")

from mindspeed_mm.fsdp.data.dataloader.sampler import BaseRandomBatchSampler  # noqa: E402


def test_too_small_dataset_raises_value_error():
    """total_samples < global batch must raise ValueError, not ZeroDivisionError."""
    # Each of these configurations hit ZeroDivisionError before the guard.
    cases = [
        (3, 4, 1),  # dataset smaller than micro_batch_size
        (1, 2, 1),  # single sample
        (5, 4, 2),  # dataset smaller than micro_batch_size x num_replicas
    ]
    for total_samples, micro_batch_size, num_replicas in cases:
        sampler = BaseRandomBatchSampler(
            list(range(total_samples)),
            batch_size=micro_batch_size,
            num_replicas=num_replicas,
            rank=0,
            shuffle=True,
            seed=0,
        )
        with pytest.raises(ValueError, match="smaller than the global batch"):
            next(iter(sampler))


def test_small_dataset_iteration_unchanged():
    """Small dataset above the global batch keeps its batch layout."""
    sampler = BaseRandomBatchSampler(
        list(range(20)), batch_size=2, num_replicas=1, rank=0, shuffle=False, seed=0
    )
    batches = list(iter(sampler))
    assert batches == [[i, i + 1] for i in range(0, 20, 2)]
    assert sampler.consumed_samples == 20


def test_small_dataset_sharded_iteration_unchanged():
    """Small dataset with data parallelism: ranks interleave, union covers all."""
    yielded = []
    for rank in range(2):
        sampler = BaseRandomBatchSampler(
            list(range(20)), batch_size=1, num_replicas=2, rank=rank,
            shuffle=False, seed=0,
        )
        rank_batches = list(iter(sampler))
        yielded.extend(batch[0] for batch in rank_batches)
        assert sampler.consumed_samples == 20
    assert yielded == [i for i in range(0, 20, 2)] + [i for i in range(1, 20, 2)]
