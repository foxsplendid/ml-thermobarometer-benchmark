# -*- coding: utf-8 -*-
"""Tests for H3: CatBoost auto device + dataset-size gate."""

import os
import pytest

from src.model_modules import _get_catboost_task_type, CatBoostModel


def _has_gpu():
    try:
        from catboost.utils import get_gpu_device_count
        return get_gpu_device_count() >= 1
    except Exception:
        return False


def test_explicit_cpu_always_cpu():
    assert _get_catboost_task_type('CPU', n_samples=1_000_000) == {}


def test_explicit_gpu_always_gpu():
    out = _get_catboost_task_type('GPU', n_samples=10)
    assert out.get('task_type') == 'GPU'


@pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU on this machine")
def test_auto_picks_cpu_for_small_data(monkeypatch):
    monkeypatch.setenv('ML_CATBOOST_GPU_MIN_SAMPLES', '5000')
    out = _get_catboost_task_type('auto', n_samples=100)
    assert out == {}, f"small data should pick CPU even with GPU available, got {out}"


@pytest.mark.skipif(not _has_gpu(), reason="no CUDA GPU on this machine")
def test_auto_picks_gpu_for_large_data(monkeypatch):
    monkeypatch.setenv('ML_CATBOOST_GPU_MIN_SAMPLES', '5000')
    out = _get_catboost_task_type('auto', n_samples=50_000)
    assert out.get('task_type') == 'GPU'


def test_auto_no_gpu_falls_back_to_cpu():
    # Even without a GPU machine, the logic should be safe
    out = _get_catboost_task_type('auto', n_samples=10_000_000)
    # If no GPU available -> {}; if GPU available -> GPU. Either is valid.
    assert out == {} or out.get('task_type') == 'GPU'


def test_threshold_env_override(monkeypatch):
    if not _has_gpu():
        pytest.skip("no GPU")
    monkeypatch.setenv('ML_CATBOOST_GPU_MIN_SAMPLES', '100')
    out = _get_catboost_task_type('auto', n_samples=500)
    assert out.get('task_type') == 'GPU'

    monkeypatch.setenv('ML_CATBOOST_GPU_MIN_SAMPLES', '10000')
    out = _get_catboost_task_type('auto', n_samples=500)
    assert out == {}


def test_n_samples_none_skips_gate():
    """Back-compat: V7-style callers without n_samples should not be gated."""
    out = _get_catboost_task_type('auto', n_samples=None)
    # Should follow V7 behaviour: GPU if available, else CPU
    if _has_gpu():
        assert out.get('task_type') == 'GPU'
    else:
        assert out == {}


def test_catboost_model_init_does_not_lock_device():
    """H3: device decision is deferred to fit(); init is cheap."""
    m = CatBoostModel(iterations=10, depth=3, task_type='auto')
    # No task_type should be locked into params at init time
    assert 'task_type' not in m.params
    # But the preference is stored
    assert m._task_type_pref == 'auto'


def test_resolve_runtime_params_small_data_cpu(monkeypatch):
    monkeypatch.setenv('ML_CATBOOST_GPU_MIN_SAMPLES', '10000')
    m = CatBoostModel(iterations=10, depth=3, task_type='auto')
    resolved = m._resolve_runtime_params(n_samples=200)
    assert 'task_type' not in resolved  # CPU path
    assert 'thread_count' in resolved   # CPU gets thread_count
