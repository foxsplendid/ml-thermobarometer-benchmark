# -*- coding: utf-8 -*-
"""Tests for H7: optional parallel inner loop of StrictOOFStacking.

No catboost / GPU involvement: budget injection and numeric identity are
exercised with the forest models only, so the suite stays fast and runs on
any machine.
"""

import numpy as np
import pytest

from src.model_modules import StrictOOFStacking


def _base_params():
    # n_jobs pinned to 1: makes sequential vs parallel runs bit-comparable
    # (only the task scheduling differs) and bypasses budget injection.
    return {
        "ert": {"n_estimators": 30, "max_depth": 6, "n_jobs": 1},
        "rf": {"n_estimators": 30, "max_depth": 6, "n_jobs": 1},
    }


@pytest.fixture
def synth_data():
    rng = np.random.RandomState(7)
    X = rng.rand(300, 6)
    y = X @ (rng.rand(6) * 10.0) + rng.rand(300)
    return X, y


def test_default_off(monkeypatch):
    monkeypatch.delenv("ML_STACKING_PARALLEL", raising=False)
    st = StrictOOFStacking(base_model_params=_base_params())
    assert st.inner_parallel == 1


def test_env_enables(monkeypatch):
    monkeypatch.setenv("ML_N_JOBS", "8")  # budget must allow 3 workers
    monkeypatch.setenv("ML_STACKING_PARALLEL", "3")
    st = StrictOOFStacking(base_model_params=_base_params())
    assert st.inner_parallel == 3


def test_workers_clamped_to_budget(monkeypatch):
    monkeypatch.setenv("ML_N_JOBS", "2")
    monkeypatch.setenv("ML_RESERVE_CORES", "0")
    st = StrictOOFStacking(base_model_params=_base_params(), inner_parallel=16)
    assert st.inner_parallel == 2


def test_invalid_env_falls_back(monkeypatch):
    monkeypatch.setenv("ML_STACKING_PARALLEL", "abc")
    st = StrictOOFStacking(base_model_params=_base_params())
    assert st.inner_parallel == 1


def test_parallel_oof_identical(synth_data, monkeypatch):
    """Parallel scheduling must not change a single bit of the OOF matrix
    or the stacked predictions (fixed seeds, per-task estimators)."""
    monkeypatch.delenv("ML_STACKING_PARALLEL", raising=False)
    monkeypatch.setenv("ML_N_JOBS", "8")  # budget must allow 3 workers
    X, y = synth_data

    seq = StrictOOFStacking(base_model_params=_base_params(),
                            inner_cv=3, random_seed=42, inner_parallel=1)
    par = StrictOOFStacking(base_model_params=_base_params(),
                            inner_cv=3, random_seed=42, inner_parallel=3)

    fitted_seq = seq.fit(X, y)
    fitted_par = par.fit(X, y)

    assert np.array_equal(seq._oof_meta_features, par._oof_meta_features)

    X_holdout = np.random.RandomState(11).rand(50, 6)
    assert np.array_equal(seq.predict(fitted_seq, X_holdout),
                          par.predict(fitted_par, X_holdout))


def test_budget_injection(monkeypatch):
    """workers x per-fit threads must not exceed the inner-loop budget:
    budget capped at ML_N_JOBS=2, 2 workers -> 1 thread per fit."""
    monkeypatch.setenv("ML_N_JOBS", "2")
    monkeypatch.setenv("ML_RESERVE_CORES", "0")
    monkeypatch.setenv("ML_STACKING_PARALLEL", "2")
    st = StrictOOFStacking(base_model_params={"ert": {"n_estimators": 10}})
    assert st.base_models[0].params["n_jobs"] == 1


def test_explicit_njobs_respected(monkeypatch):
    monkeypatch.delenv("ML_STACKING_PARALLEL", raising=False)
    st = StrictOOFStacking(
        base_model_params={"ert": {"n_estimators": 10, "n_jobs": 3}},
        inner_parallel=2,
    )
    assert st.base_models[0].params["n_jobs"] == 3


def test_parallel_pins_catboost_auto_to_cpu(monkeypatch):
    """Parallel workers must not auto-select one shared GPU; an explicit
    'GPU' stays (caller's responsibility, warned)."""
    monkeypatch.setenv("ML_N_JOBS", "2")
    monkeypatch.setenv("ML_RESERVE_CORES", "0")

    st = StrictOOFStacking(
        base_model_params={"catboost": {"iterations": 10}}, inner_parallel=2
    )
    cb = st.base_models[0]
    assert cb._task_type_pref == "CPU"
    assert cb._user_kwargs["thread_count"] == 1  # budget 2 // 2 workers

    st_gpu = StrictOOFStacking(
        base_model_params={"catboost": {"iterations": 10, "task_type": "GPU"}},
        inner_parallel=2,
    )
    assert st_gpu.base_models[0]._task_type_pref == "GPU"
