"""
We can't use assert in our code for codecheck, so create this auxiliary function to wrap
the assert case in ut for ci.
"""
import megatron.core.parallel_state as ps


def judge_expression(expression):
    if not expression:
        raise AssertionError


class TestConfig(object):
    def __init__(self, entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = TestConfig(v)
            else:
                self.__dict__[k] = v

    def to_dict(self):
        ret = {}
        for k, v in self.__dict__.items():
            if isinstance(v, self.__class__):
                ret[k] = v.to_dict()
            else:
                ret[k] = v
        return ret


def initialize_model_parallel(
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    virtual_pipeline_model_parallel_size=None,
    pipeline_model_parallel_split_rank=None,
    context_parallel_size=1,
):
    ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=virtual_pipeline_model_parallel_size,
        pipeline_model_parallel_split_rank=pipeline_model_parallel_split_rank,
        context_parallel_size=context_parallel_size,
    )
