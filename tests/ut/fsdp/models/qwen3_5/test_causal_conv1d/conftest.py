# Copyright (c) 2026, Huawei Technologies Co., Ltd.

import pytest

from mindspeed_mm.fsdp.models.qwen3_5.triton.utils import get_available_device, is_arch35


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark a test as slow; deselected by default via `-m 'not slow'`",
    )


def pytest_collection_modifyitems(config, items):
    if is_arch35():
        skip = pytest.mark.skip(reason="causal_conv1d kernels are not supported on arch35")
        for item in items:
            item.add_marker(skip)


DEVICE = get_available_device()
