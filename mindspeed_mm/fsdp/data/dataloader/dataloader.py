# pylint: skip-file
# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import sys
from typing import Optional

import torch
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

from mindspeed_mm.fsdp.data.data_utils.utils import get_seed_worker
from mindspeed_mm.fsdp.data.dataloader.sampler import BaseRandomBatchSampler
from mindspeed_mm.fsdp.data.dataloader.data_collator import resolve_data_collator
from mindspeed_mm.fsdp.utils.constants import GLOBAL_STEP_TOKEN_NUM, AVG_PER_STEP_TOKEN_NUM
from mindspeed_mm.fsdp.utils.device import get_device_type, create_stream, get_current_stream, switch_to_specified_stream
from mindspeed_mm.fsdp.data.data_utils.utils import build_iterations
from mindspeed_mm.fsdp.utils.utils import move_to_device


def prepare_base_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=None,
    collate_param=None,
    dataset_param=None,
    model=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    collate_fn = None
    if collate_param:
        data_collator, collate_kwargs = resolve_data_collator(collate_param, dataset_param)
        collate_fn = data_collator(dataset_param=dataset_param, model=model, **collate_kwargs)
    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    return StatefulDataLoader(
        dataset,
        pin_memory=pin_memory,
        pin_memory_device=get_device_type(),
        collate_fn=collate_fn,
        worker_init_fn=get_seed_worker(seed),
        num_workers=num_workers,
        batch_sampler=None,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers
    )


def prepare_sampler_dataloader(
    dataset,
    batch_size=1,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    prefetch_factor=None,
    persistent_workers=None,
    process_group: Optional[ProcessGroup] = None,
    data_sharding=False,
    sampler_type="stateful_distributed_sampler",
    collate_param=None,
    dataset_param=None,
    model=None,
    **kwargs,
):
    """
    Prepare a dataloader for distributed training. The dataloader will be wrapped by
    `torch.utils.data.DataLoader` and `StatefulDistributedSampler`.

    Args:
        dataset (`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``

    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    if isinstance(dataset, torch.utils.data.dataset.IterableDataset):
        num_workers = 0

    if persistent_workers is None:
        persistent_workers = True if num_workers > 0 else False

    if sampler_type == "BaseRandomBatchSampler":
        batch_sampler = BaseRandomBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            drop_last=drop_last,
            data_sharding=data_sharding,
        )
        collate_fn = None

        if collate_param:
            if hasattr(dataset, 'collate_fn') and callable(getattr(dataset, 'collate_fn')):
                collate_fn = dataset.collate_fn
            else:
                data_collator, collate_kwargs = resolve_data_collator(collate_param, dataset_param)
                collate_fn = data_collator(**collate_kwargs, dataset_param=dataset_param, model=model)

        return StatefulDataLoader(
            dataset,
            pin_memory=pin_memory,
            pin_memory_device=get_device_type(),
            collate_fn=collate_fn,
            worker_init_fn=get_seed_worker(seed),
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers
        )
    else:
        raise NotImplementedError(f"Sampler type {sampler_type} is not implemented.")


class PrefetchGradAccDataLoader:
    """
    A DataLoader wrapper that prefetches a fixed number of batches (equal to gradient
    accumulation steps), computes total and average valid token counts across them,
    and injects these metrics into each batch before yielding.

    This is Used for calculate per-token-loss
    """

    def __init__(self, base_dataloader: StatefulDataLoader, grad_acc_step: int, cyclic: bool = True):
        """
        Args:
            base_dataloader: The underlying PyTorch DataLoader to wrap.
            grad_acc_step (int): Number of batches to accumulate gradients over.
            cyclic (bool): Whether to cycle over the underlying dataloader indefinitely.
        """
        if grad_acc_step <= 0:
            raise ValueError("grad_acc_step must be a positive integer.")
        self.grad_acc_step = grad_acc_step
        self.base_dataloader = base_dataloader
        self.cyclic = cyclic
        self.base_iter = self._build_base_iter()
        self._current_iterator = None  # Holds the active generator

    def _build_base_iter(self):
        if self.cyclic:
            base_iter, _, _ = build_iterations(self.base_dataloader)
            return base_iter
        return iter(self.base_dataloader)

    def __iter__(self):
        # Start a new iteration (reset state)
        self.base_iter = self._build_base_iter()
        self._current_iterator = self._generate_batches()
        return self  # Return self as the iterator

    def __len__(self):
        if hasattr(self.base_dataloader, "__len__"):
            return len(self.base_dataloader)
        else:
            return 0

    def __next__(self):
        if self._current_iterator is None:
            # If __next__ is called before __iter__, start iteration
            self._current_iterator = self._generate_batches()
        try:
            return next(self._current_iterator)
        except StopIteration:
            self._current_iterator = None  # Reset on exhaustion
            raise

    def _generate_batches(self):
        """Generator that yields batches with injected token counts."""
        try:
            while True:
                buffer = []
                total_tokens = 0
                fetched = 0

                # Prefetch grad_acc_step batches
                try:
                    for _ in range(self.grad_acc_step):
                        batch = next(self.base_iter)
                        valid_token_count = (batch["labels"] > -1).sum()  # tokens of shift labels
                        total_tokens += valid_token_count
                        buffer.append(batch)
                        fetched += 1
                except StopIteration:
                    if fetched == 0:
                        break  # No more data

                # 尾端不足时，平均tokens数应该用实际值做除法
                denom = fetched if fetched > 0 else self.grad_acc_step
                avg_tokens = total_tokens / denom
                for batch in buffer:
                    batch_dict = dict(batch)
                    batch_dict[GLOBAL_STEP_TOKEN_NUM] = total_tokens
                    batch_dict[AVG_PER_STEP_TOKEN_NUM] = avg_tokens
                    yield batch_dict
        finally:
            if hasattr(self.base_iter, 'close'):
                self.base_iter.close()

    def state_dict(self):
        return self.base_dataloader.state_dict()

    def load_state_dict(self, **kwargs):
        self.base_dataloader.load_state_dict(**kwargs)


class Preloader:
    """
    Async data preloader with decoupled CPU prefetch and H2D transfer.

    The CPU-side data loading (``next`` on the underlying iterator, collation, etc.)
    runs in a background thread and overlaps with forward/backward without consuming
    any device memory. The H2D transfer is launched on a dedicated stream only when
    :meth:`trigger_h2d` is called -- typically right after backward -- so that the
    next batch's device tensors are allocated after the current batch's activations have
    been freed, reducing peak device memory during the forward/backward pass.

    Expected usage pattern (one ``next`` / ``trigger_h2d`` pair per micro-batch)::

        batch = preloader.next()       # consume ready batch, start CPU prefetch
        forward(batch); backward(batch)
        preloader.trigger_h2d()        # launch H2D for next batch
    """

    def __init__(self, data_iterator, param_dtype=None):
        self.data_iterator = data_iterator
        self.param_dtype = param_dtype
        self.device = get_device_type()
        self.h2dstream = create_stream(self.device)
        # next_batch: device batch ready to be returned by next()
        # _cpu_batch: cpu batch fetched by the background thread, awaiting H2D
        self.next_batch = None
        self._cpu_batch = None
        self._cpu_thread = None
        self._h2d_thread = None
        self._fetch_error = None
        self._exhausted = False

        # Eagerly prepare the first batch so the first next() is immediately ready.
        self._fetch_cpu_sync()
        self._launch_h2d()

    def _fetch_cpu_sync(self):
        """Fetch the next CPU batch from the underlying iterator (thread target)."""
        try:
            self._cpu_batch = next(self.data_iterator)
        except StopIteration:
            self._cpu_batch = None
            self._exhausted = True
        except Exception:
            # Store exc_info to re-raise on the main thread after join,
            # preserving the original traceback from this background thread.
            self._fetch_error = sys.exc_info()

    def _start_cpu_prefetch(self):
        """Launch a background daemon thread to fetch the next CPU batch."""
        if self._exhausted:
            self._cpu_thread = None
            return
        self._cpu_thread = threading.Thread(target=self._fetch_cpu_sync, daemon=True)
        self._cpu_thread.start()

    def _launch_h2d(self):
        """Transfer the prefetched CPU batch to device on the dedicated stream."""
        # Make sure the CPU prefetch is done before launching H2D.
        if self._cpu_thread is not None:
            self._cpu_thread.join()
            self._cpu_thread = None
        # If CPU prefetch failed, mark next_batch as unavailable;
        # the stored error is re-raised on the main thread in next().
        if self._fetch_error is not None:
            self.next_batch = None
            return
        if self._cpu_batch is None:
            self.next_batch = None
            return
        batch_data = self._cpu_batch
        self._cpu_batch = None
        with switch_to_specified_stream(self.h2dstream):
            self.next_batch = move_to_device(
                batch_data, float_dtype=self.param_dtype, non_blocking=True
            )

    def trigger_h2d(self):
        """Launch H2D for the prefetched batch in a background thread.
        Call this after backward completes (per micro-batch) so that the next
        batch's device memory is allocated only after the current activations are freed.
        """
        if self._exhausted and self._cpu_thread is None:
            return
        self._h2d_thread = threading.Thread(target=self._launch_h2d, daemon=True)
        self._h2d_thread.start()

    def next(self):
        """Return the ready device batch, or None if the iterator is exhausted."""
        if self._h2d_thread is not None:
            self._h2d_thread.join()
            self._h2d_thread = None
        # Re-raise any exception captured in the CPU prefetch thread,
        # preserving the original traceback from where it originated.
        if self._fetch_error is not None:
            _, exc_value, exc_tb = self._fetch_error
            self._fetch_error = None
            raise exc_value.with_traceback(exc_tb)
        get_current_stream().wait_stream(self.h2dstream)
        if self.next_batch is None:
            raise StopIteration("Dataloader has been exhausted, no more data available.")
        batch = self.next_batch
        self.next_batch = None
        # Kick off CPU prefetch for the next batch so it overlaps with the upcoming forward&backward.
        self._start_cpu_prefetch()
        return batch

    def __iter__(self):
        yield self.next()

    def __next__(self):
        return self.next()
