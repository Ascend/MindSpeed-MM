# Copyright (c) 2024 Keller Jordan
# Copyright (c) 2025 Moonshot AI
from .coefficients import resolve_ns_coefficients
from .newton_schulz import (
    zeropower_via_newtonschulz5,
    zeropower_via_newtonschulz5_per_expert,
    zeropower_via_newtonschulz5_per_head,
)
from .optimizer import Muon, adjust_lr_wd_for_muon
from .qk_clip import qk_clip_, qk_clip_mla_


__all__ = [
    "Muon",
    "adjust_lr_wd_for_muon",
    "qk_clip_",
    "qk_clip_mla_",
    "resolve_ns_coefficients",
    "zeropower_via_newtonschulz5",
    "zeropower_via_newtonschulz5_per_expert",
    "zeropower_via_newtonschulz5_per_head",
]
