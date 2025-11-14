#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
@Desc    : MindSpore weight convert command-line entry
"""


def main():
    import mindspore as ms
    ms.set_context(device_target="CPU")
    import torch
    torch.configs.set_pyboost(False)

    from mindspeed.args_utils import get_mindspeed_args
    mindspeed_args = get_mindspeed_args()
    mindspeed_args.ai_framework = "mindspore"
    import mindspeed.megatron_adaptor

    import jsonargparse
    from checkpoint.common.converter import Commandable

    import os
    os.environ['JSONARGPARSE_DEPRECATION_WARNINGS'] = 'off'
    # Allow docstring (including field descriptions) to be parsed as the command-line help documentation.
    # When customizing a converter, you need to inherit from Converter and add it to __init__.py.
    jsonargparse.set_parsing_settings(docstring_parse_attribute_docstrings=True)
    jsonargparse.auto_cli(Commandable.subclasses)


if __name__ == "__main__":
    main()
