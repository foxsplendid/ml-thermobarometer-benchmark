# -*- coding: utf-8 -*-
"""Tests for src.runtime (hardware probe + n_jobs allocator)."""

import os
import pytest

from src.runtime import (
    Runtime,
    get_runtime,
    runtime_summary_str,
    suggest_n_jobs,
)


def test_runtime_fields():
    r = get_runtime()
    assert isinstance(r, Runtime)
    assert r.n_physical_cores >= 1
    assert r.n_logical_cores >= r.n_physical_cores
    assert isinstance(r.has_cuda_gpu, bool)
    assert r.n_gpu_devices >= 0
    assert r.platform


def test_runtime_is_cached():
    assert get_runtime() is get_runtime()


def test_runtime_summary_str():
    s = runtime_summary_str()
    assert "Runtime:" in s
    assert "cores" in s


def test_gpu_probe_failure_returns_no_gpu(monkeypatch):
    """SPEC §3: GPU probe failures must degrade to no-GPU, not raise."""
    import sys
    import types
    import src.runtime as rt

    class _BrokenCuda:
        def is_available(self):
            raise RuntimeError("driver exploded")

    broken_torch = types.ModuleType("torch")
    broken_torch.cuda = _BrokenCuda()

    def _boom():
        raise RuntimeError("no driver")

    broken_cb_utils = types.ModuleType("catboost.utils")
    broken_cb_utils.get_gpu_device_count = _boom
    broken_cb = types.ModuleType("catboost")
    broken_cb.utils = broken_cb_utils

    monkeypatch.setitem(sys.modules, "torch", broken_torch)
    monkeypatch.setitem(sys.modules, "catboost", broken_cb)
    monkeypatch.setitem(sys.modules, "catboost.utils", broken_cb_utils)

    assert rt._probe_gpu() == (False, 0, None)


def _with_env(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, str(v))


def test_suggest_n_jobs_default_positive(monkeypatch):
    _with_env(monkeypatch, ML_N_JOBS=None, ML_RESERVE_CORES=None, ML_OUTER_PROCS=None)
    assert suggest_n_jobs("model") >= 1


def test_suggest_n_jobs_respects_reserve(monkeypatch):
    r = get_runtime()
    _with_env(monkeypatch, ML_N_JOBS=None, ML_OUTER_PROCS=None,
              ML_RESERVE_CORES=r.n_physical_cores)
    # Reserve everything -> still clamped to 1
    assert suggest_n_jobs("model") == 1


def test_suggest_n_jobs_respects_cap(monkeypatch):
    _with_env(monkeypatch, ML_RESERVE_CORES=0, ML_OUTER_PROCS=None, ML_N_JOBS=2)
    assert suggest_n_jobs("model") <= 2


def test_suggest_n_jobs_cap_negative_means_all(monkeypatch):
    """V7 convention: ML_N_JOBS=-1 means 'use all cores'."""
    r = get_runtime()
    _with_env(monkeypatch, ML_RESERVE_CORES=0, ML_OUTER_PROCS=None, ML_N_JOBS=-1)
    assert suggest_n_jobs("model") == r.n_physical_cores


def test_suggest_n_jobs_cross_proc_divides(monkeypatch):
    r = get_runtime()
    if r.n_physical_cores < 4:
        pytest.skip("need >=4 physical cores")
    _with_env(monkeypatch, ML_RESERVE_CORES=0, ML_N_JOBS=None, ML_OUTER_PROCS=4)
    assert suggest_n_jobs("cross_proc") == max(1, r.n_physical_cores // 4)


def test_suggest_n_jobs_explicit_override(monkeypatch):
    _with_env(monkeypatch, ML_N_JOBS=None, ML_OUTER_PROCS=None, ML_RESERVE_CORES=999)
    # explicit reserve_cores overrides env
    assert suggest_n_jobs("model", reserve_cores=0) >= 1
