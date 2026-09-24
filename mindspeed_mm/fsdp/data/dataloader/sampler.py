from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler


class BaseRandomBatchSampler(StatefulDistributedSampler):
    """
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): Accepted for compatibility but ignored for shuffling.
            This sampler uses only the epoch as its random seed. To control the
            shuffle with a seed, use :class:`SeedRandomBatchSampler`. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Default: ``True``. (It is not implemented that the drop_last is false.)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        data_sharding: bool = False,
        infinite: bool = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.total_samples = len(dataset)
        self.micro_batch_size = batch_size
        self.consumed_samples = 0
        self.next_consumed_samples = None
        self.data_sharding = data_sharding
        self.infinite = infinite
        self.epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * self.num_replicas
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        if not drop_last:
            raise ValueError("It is not implemented that the drop_last is false.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        # resume sampler
        if self.next_consumed_samples is not None:
            self.consumed_samples = self.next_consumed_samples
            self.next_consumed_samples = None

        # infinite sampler keeps yielding across epoch boundaries to prevent data stall.
        while True:
            yield from self._iter_one_epoch()
            if not self.infinite:
                break

    def _iter_one_epoch(self):
        active_total_samples = self.total_samples - self.last_batch_size
        if active_total_samples <= 0:
            raise ValueError(
                f"Total samples ({self.total_samples}) is smaller than the global batch "
                f"size (micro_batch_size={self.micro_batch_size} x num_replicas="
                f"{self.num_replicas} = {self.micro_batch_times_data_parallel_size}); "
                f"the sampler has no full batch to iterate. Increase the dataset size "
                f"or reduce micro_batch_size/data_parallel_size."
            )
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self._get_epoch_seed())

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.num_replicas
            start_idx = self.rank * bucket_size
            if self.shuffle:
                idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
            else:
                idx_range_bucket = list(range(bucket_size))
            idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            if self.shuffle:
                idx_range_total = \
                    torch.randperm(full_bucket_size, generator=g).tolist()
            else:
                idx_range_total = list(range(full_bucket_size))
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.rank::self.num_replicas]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

    def _get_epoch_seed(self) -> int:
        """Preserve the legacy epoch-only shuffle order."""
        return self.epoch

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.consumed_samples}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_consumed_samples = state_dict[self._YIELDED]


class SeedRandomBatchSampler(BaseRandomBatchSampler):
    """Batch sampler whose shuffle depends on both seed and epoch.

    Enable in FSDP2 training via data.dataloader_param: dataloader_mode="sampler",
    sampler_type="SeedRandomBatchSampler", shuffle=True and drop_last=True.
    The trainer forwards training.seed; use 0 <= training.seed < 2**32 for NumPy
    worker/global seeding compatibility. Switching from BaseRandomBatchSampler
    changes the shuffle order, so keep the sampler type unchanged when resuming.

    Inherits batching, sharding, infinite iteration and checkpoint handling from
    BaseRandomBatchSampler. With shuffle=True, seed must be non-negative and
    identical across ranks. With shuffle=False, seed is ignored. Resuming requires
    the same seed and sampling configuration; state_dict stores only the position.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        data_sharding: bool = False,
        infinite: bool = False,
    ):
        if shuffle and seed < 0:
            raise ValueError(
                f"SeedRandomBatchSampler requires seed >= 0, got {seed}"
            )
        super().__init__(
            dataset, batch_size, num_replicas, rank, shuffle, seed,
            drop_last, data_sharding, infinite,
        )

    def _get_epoch_seed(self) -> int:
        # PyTorch's DistributedSampler uses seed + epoch for shuffling.
        # Different combinations can therefore produce the same shuffle order:
        #   epoch=1, seed=1 -> 2
        #   epoch=2, seed=0 -> 2
        # SeedSequence mixes both inputs deterministically to avoid this equal-sum pattern.
        # The CPU RNG uses 32 seed bits, so collisions remain possible after mixing.
        # Keep the seed identical across ranks so global shuffle partitions agree.
        return int(
            np.random.SeedSequence([self.epoch, self.seed])
            .generate_state(1, dtype=np.uint32)[0]
        )
