"""UTs for prime_cpu_affinity_binding (fsdp.utils.device).

Cardless semantics: MM_PRIME_CPU_AFFINITY=auto (default) + CPU_AFFINITY_CONF
unset/'0' -> strict no-op (torch.ones never touched); =0 -> no-op even with
CPU_AFFINITY_CONF set; =1 -> primes even with CPU_AFFINITY_CONF unset.
Priming executes a tiny RNG-free backward (NPU available in the UT env).
"""
import os

import pytest
import torch

from mindspeed_mm.fsdp.utils.device import prime_cpu_affinity_binding


def _forbidden(*a, **kw):
    raise AssertionError("must not allocate when priming is disabled")


class TestPrimeCpuAffinityBinding:
    def test_noop_when_unset(self, monkeypatch):
        monkeypatch.delenv("CPU_AFFINITY_CONF", raising=False)
        monkeypatch.delenv("MM_PRIME_CPU_AFFINITY", raising=False)
        monkeypatch.setattr(torch, "ones", _forbidden)
        prime_cpu_affinity_binding()  # must be a strict no-op

    def test_noop_when_zero(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "0")
        monkeypatch.delenv("MM_PRIME_CPU_AFFINITY", raising=False)
        monkeypatch.setattr(torch, "ones", _forbidden)
        prime_cpu_affinity_binding()

    def test_noop_when_prime_disabled(self, monkeypatch):
        # Kill-switch: affinity stays configured, priming is off.
        monkeypatch.setenv("CPU_AFFINITY_CONF", "1")
        monkeypatch.setenv("MM_PRIME_CPU_AFFINITY", "0")
        monkeypatch.setattr(torch, "ones", _forbidden)
        prime_cpu_affinity_binding()

    @pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="needs an NPU")
    def test_runs_when_set(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "1")
        monkeypatch.delenv("MM_PRIME_CPU_AFFINITY", raising=False)
        prime_cpu_affinity_binding()  # executes the dummy backward; no exception

    @pytest.mark.skipif(
        not (hasattr(torch, "npu") and torch.npu.is_available()),
        reason="needs an NPU")
    def test_force_prime_without_conf(self, monkeypatch):
        monkeypatch.delenv("CPU_AFFINITY_CONF", raising=False)
        monkeypatch.setenv("MM_PRIME_CPU_AFFINITY", "1")
        prime_cpu_affinity_binding()  # forced priming without affinity conf

    def test_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("CPU_AFFINITY_CONF", "1")
        monkeypatch.delenv("MM_PRIME_CPU_AFFINITY", raising=False)

        def _boom(*a, **kw):
            raise RuntimeError("device gone")

        monkeypatch.setattr(torch, "ones", _boom)
        prime_cpu_affinity_binding()  # warns and falls back, never raises
