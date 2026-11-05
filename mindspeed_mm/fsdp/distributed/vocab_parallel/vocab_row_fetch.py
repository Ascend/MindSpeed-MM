"""Row fetch backends for the vocabulary embedding.

The routing around the lookup (owner split, all-to-all of ids, return all-to-all) is the same for both
embedding placements; only the row fetch differs, so that is the single seam this package abstracts:

* :class:`HostRowFetcher` -- the embedding lives in host memory and each lookup ships the touched rows to
  the device on a side stream; the gradient is assembled from those rows (see VocabRowGrad).
* :class:`DeviceRowFetcher` -- the embedding is a vocabulary-parallel DTensor and the rows come straight
  from this rank's shard, so autograd owns the gradient and there is no host ledger.
"""
import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from .vocab_sparse_grad import host_row_lookup


class HostRowFetcher:
    """Gather the rows out of the host-resident embedding and hand them to the device."""

    def fetch(self, module, received_ids, plan, async_copy: bool = True):
        """Return ``(rows, event)``; ``event`` is None when the copy is blocking."""
        ready = []
        rows = host_row_lookup(
            module.weight,
            received_ids,
            module._vp_row_grad,
            received_ids.device,
            async_copy=async_copy,
            ready=ready,
        )
        return rows, (ready[0] if ready else None)


class DeviceRowFetcher:
    """Read the rows from the local shard of the vocabulary-parallel embedding."""

    def fetch(self, module, received_ids, plan, async_copy: bool = True):
        """Return ``(rows, None)``; the rows are already where the collectives consume them."""
        weight = module.weight
        local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
        # Same call the device-resident baseline made: the gradient of the selected rows flows back
        # into the local shard, and the window arithmetic already keeps every id inside it.
        rows = F.embedding(received_ids, local_weight)
        return rows, None
