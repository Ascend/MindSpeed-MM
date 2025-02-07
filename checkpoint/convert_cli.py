#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : convert_cli.py
@Time    : 2025/01/12
@Desc    : 权重转换命令行入口
"""
import os
from abc import ABC, abstractmethod

import jsonargparse

from checkpoint import qwen2vl_hf_to_mm
from checkpoint.utils import ConvertHFConfig, ConvertResplitConfig, ConvertMMConfig

# 安全权限，当前用户读写权限，用户组内可读权限，其他用户无权限
SAFE_MODE = 0o640


class Converter(ABC):
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        """所有Converter的子类都会保存在Converter类属性subclasses中"""
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    @staticmethod
    @abstractmethod
    def hf_to_mm(cfg: ConvertMMConfig):
        pass

    @staticmethod
    @abstractmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        pass

    @staticmethod
    @abstractmethod
    def resplit(cfg: ConvertResplitConfig):
        pass


class Qwen2VLConverter(Converter):
    """Qwen2VL模型转换工具"""

    @staticmethod
    def hf_to_mm(cfg: ConvertMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        qwen2vl_hf_to_mm.main(cfg)
        # 安全管控权限
        os.chmod(cfg.mm_dir, SAFE_MODE)

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        # qwen2vl_mm_to_hf会用到mindspeed.megatron_adaptor模块，提前import会引入问题，故在此import
        from checkpoint import qwen2vl_mm_to_hf
        qwen2vl_mm_to_hf.main(cfg)
        # 安全管控权限
        os.chmod(cfg.save_hf_dir, SAFE_MODE)

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        # qwen2vl_resplit会用到mindspeed.megatron_adaptor模块，提前import会引入问题，故在此import
        from checkpoint import qwen2vl_resplit
        qwen2vl_resplit.main(cfg)
        # 安全管控权限
        os.chmod(cfg.target_dir, SAFE_MODE)


class InternVLConverter(Converter):
    """InternVL模型转换工具（仅做样例，当前尚未支持）"""

    @staticmethod
    def hf_to_mm(cfg: ConvertMMConfig):
        """huggingface模型转换mindspeed-mm模型权重"""
        pass

    @staticmethod
    def mm_to_hf(cfg: ConvertHFConfig):
        """mindspeed-mm模型转换huggingface模型权重"""
        pass

    @staticmethod
    def resplit(cfg: ConvertResplitConfig):
        """mindspeed-mm模型权重重新切分"""
        pass


def main():
    # 允许docstring（包含field的说明）解析为命令行的说明文档
    jsonargparse.set_docstring_parse_options(attribute_docstrings=True)
    jsonargparse.CLI(Converter.subclasses)


if __name__ == "__main__":
    main()
