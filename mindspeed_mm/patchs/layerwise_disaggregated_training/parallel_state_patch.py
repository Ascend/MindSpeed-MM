# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""VTP (Virtual Tensor Parallelism) parallel state management.

This module holds the global VTP state variables and their getter/setter
functions.  It is intentionally kept **free of any imports** from
``schedules_patch`` or ``p2p_communication_patch`` so that both of those
modules can safely import from here without creating a circular-import
chain.
"""

import torch


# ── Global VTP state ────────────────────────────────────────────
_VTP_ENABLED = False
_VTP_SIZE_LIST = None        # list[int] — TP size per PP stage
_VTP_STAGE_RANKS = None      # list[list[int]] — global ranks per stage
_VTP_INTRA_STAGE_GROUP = None  # ProcessGroup for intra-stage comm
_VTP_MY_STAGE_IDX = None     # int — index of this rank's stage


# ── Initialisation helpers ──────────────────────────────────────

def _init_vtp_state(vtp_enabled, vtp_size_list, stage_ranks):
    """Initialize VTP global state variables."""
    global _VTP_ENABLED, _VTP_SIZE_LIST, _VTP_STAGE_RANKS
    global _VTP_MY_STAGE_IDX

    _VTP_ENABLED = vtp_enabled
    _VTP_SIZE_LIST = vtp_size_list
    _VTP_STAGE_RANKS = stage_ranks

    rank = torch.distributed.get_rank()
    for stage_idx, stage in enumerate(stage_ranks):
        if rank in stage:
            _VTP_MY_STAGE_IDX = stage_idx
            break


def _create_vtp_groups(stage_ranks, timeout, backend):
    """Create VTP intra-stage communication group."""
    global _VTP_INTRA_STAGE_GROUP

    rank = torch.distributed.get_rank()

    for stage in stage_ranks:
        if len(stage) > 1:
            group = torch.distributed.new_group(
                ranks=stage, timeout=timeout, backend=backend
            )
            if rank in stage:
                _VTP_INTRA_STAGE_GROUP = group


# ── Getters ─────────────────────────────────────────────────────

def is_vtp_enabled():
    return _VTP_ENABLED


def get_vtp_size_list():
    return _VTP_SIZE_LIST


def get_vtp_stage_ranks():
    return _VTP_STAGE_RANKS


def get_vtp_intra_stage_group():
    return _VTP_INTRA_STAGE_GROUP


def get_vtp_my_stage_idx():
    return _VTP_MY_STAGE_IDX


def is_vtp_stage_rank0():
    if not _VTP_STAGE_RANKS or _VTP_MY_STAGE_IDX is None:
        return True
    return torch.distributed.get_rank() == _VTP_STAGE_RANKS[_VTP_MY_STAGE_IDX][0]
