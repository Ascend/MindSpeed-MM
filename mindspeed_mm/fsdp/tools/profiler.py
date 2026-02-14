# Copyright 2025 Bytedance Ltd. and/or its affiliates
import datetime
import os
import logging

import torch

from mindspeed_mm.fsdp.params.training_args import Profiler
from mindspeed_mm.fsdp.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_torch_device
)


if IS_NPU_AVAILABLE:
    import torch_npu

logger = logging.getLogger(__name__)


class Profiler:
    def __init__(self, config: Profiler):
        self.config = config
        self.first_step = True  # flagging the first step for record memory history
        self.current_step = 0
        self.global_rank = int(os.getenv("RANK"))
        if self.config.enable:
            self._p = self._create_profiler()

    def _create_profiler(self):
        """
        Creates a profiler to record the CPU and CUDA activities. Default export to trace.json.
        Profile steps in [start_step, end_step).

        When is_npu_available = True, the profiler will be created as torch_npu.profiler.

        Args:
            start_step (int): The step to start recording.
            end_step (int): The step to end recording.
            save_path (str): The path to save the profiling result.
            record_shapes (bool): Whether to record the shapes of the tensors.
            with_memory (bool): Whether to profile the memory usage.
            with_stack (bool): Whether to include the stack trace.
        """

        def handler_fn(p):
            time = int(datetime.datetime.now().timestamp())

            trace_file_extention = "pt.trace.json.gz"

            os.makedirs(self.config.save_path, exist_ok=True)
            trace_file = os.path.join(self.config.save_path, f"rank{self.global_rank}_{time}.{trace_file_extention}")

            if IS_NPU_AVAILABLE:
                nonlocal npu_trace_handler
                npu_trace_handler(p)
                trace_file = p.prof_if.prof_path
            elif IS_CUDA_AVAILABLE:
                p.export_chrome_trace(trace_file)
            logger.info(f"Profiling result saved at {trace_file}.")

        if IS_NPU_AVAILABLE:
            profiler_module = torch_npu.profiler
            activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.NPU]
            npu_trace_handler = torch_npu.profiler.tensorboard_trace_handler(self.config.save_path)
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                data_simplification=False,
            )
        else:
            profiler_module = torch.profiler
            activities = [profiler_module.ProfilerActivity.CPU, profiler_module.ProfilerActivity.CUDA]
            experimental_config = None

        active = self.config.end_step - self.config.start_step

        skip_first = self.config.start_step - 1
        schedule = profiler_module.schedule(
            wait=0,
            warmup=0,
            active=active,
            repeat=1,
            skip_first=skip_first,
        )
        base_profiler = profiler_module.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=handler_fn,
            record_shapes=self.config.record_shapes,
            profile_memory=self.config.with_memory,
            with_modules=False,
            with_stack=self.config.with_stack,
            experimental_config=experimental_config,
        )

        return base_profiler

    def start(self):
        if not self.config.enable:
            return
        out = self._p.start()

    def stop(self):
        if not self.config.enable:
            return

        if self.config.end_step == self.current_step:
            out = self._p.stop()

        if self.config.with_memory and self.current_step == self.config.end_step:
            time = int(datetime.datetime.now().timestamp())
            memory_file_extension = "pkl"
            memory_file = os.path.join(self.config.save_path, f"rank{self.global_rank}_{time}.{memory_file_extension}")
            get_torch_device().memory._dump_snapshot(memory_file)
            logger.info(f"Profiling memory visualization saved at {memory_file}.")
            get_torch_device().memory._record_memory_history(enabled=None)  # step recording memory snapshot

    def step(self, *a, **kw):
        if not self.config.enable:
            return

        out = self._p.step(*a, **kw)
        self.current_step += 1
    
    def memory_record(self):
        if not self.config.enable:
            return
        if self.current_step >= self.config.start_step and self.current_step < self.config.end_step:
            if self.config.with_memory and self.first_step:
                get_torch_device().memory._record_memory_history()
                self.first_step = False
