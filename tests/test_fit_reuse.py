# -*- coding: utf-8 -*-
"""Tests for H6: opt-in reuse of outer-CV fits between experiments that
differ only in correction module (E10-E12 consume E07-E09 in the real
matrix; here a synthetic ert raw none/segmented pair)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.protocol import ExperimentConfig, ExperimentMatrix, StratifiedCVProtocol

N_FEATURES = 4
MODEL_PARAMS = {"n_estimators": 10, "max_depth": 5, "n_jobs": 1}


def _make_data():
    rng = np.random.RandomState(5)
    X = rng.rand(150, N_FEATURES) * 10.0
    y_T = X @ (rng.rand(N_FEATURES) * 50.0) + 800.0 + rng.randn(150) * 5.0
    y_P = X @ (rng.rand(N_FEATURES) * 2.0) + rng.randn(150) * 0.5
    labels = np.repeat(np.arange(4), 30)  # 4 bins x 30 train samples
    train = slice(0, 120)
    test = slice(120, 150)
    return (X[train], y_T[train], y_P[train], labels[:120],
            X[test], y_T[test], y_P[test])


def _configs(corr_pair=("none", "segmented"), model_params=None):
    return [
        ExperimentConfig(
            exp_id=f"EX_{corr}",
            data_module_name="raw",
            model_module_name="ert",
            corr_module_name=corr,
            model_params=dict(model_params or MODEL_PARAMS),
        )
        for corr in corr_pair
    ]


def _run(out_dir, reuse, nested=True, configs=None):
    X, y_T, y_P, labels, X_test, yT_test, yP_test = _make_data()
    matrix = ExperimentMatrix(X=X, y_T=y_T, y_P=y_P, output_dir=str(out_dir))
    return matrix.run_experiments(
        configs=configs if configs is not None else _configs(),
        n_splits=3,
        stratify_labels=labels,
        X_test=X_test,
        y_T_test=yT_test,
        y_P_test=yP_test,
        random_seed=42,
        nested_correction=nested,
        verbose=False,
        reuse_identical_fits=reuse,
    )


def test_reuse_identical_metrics(tmp_path):
    """With deterministic estimators, reuse on/off must agree on every
    numeric output except the training_time columns."""
    s_off = _run(tmp_path / "off", reuse=False)
    s_on = _run(tmp_path / "on", reuse=True)

    assert list(s_off.columns) == list(s_on.columns)
    for col in s_off.columns:
        if "training_time" in col:
            continue
        a, b = s_off[col].values, s_on[col].values
        if s_off[col].dtype.kind in "fciu":
            np.testing.assert_array_equal(a, b, err_msg=f"column {col}")
        else:
            assert (a == b).all(), f"column {col}"

    for exp in ("EX_none", "EX_segmented"):
        for tgt in ("T", "P"):
            a = pd.read_csv(tmp_path / "off" / f"{exp}_{tgt}_fold_metrics.csv")
            b = pd.read_csv(tmp_path / "on" / f"{exp}_{tgt}_fold_metrics.csv")
            cols = [c for c in a.columns if "training_time" not in c]
            pd.testing.assert_frame_equal(a[cols], b[cols])


def test_reuse_skips_outer_fits(tmp_path, monkeypatch):
    """Fit accounting (nested off so only outer fits count): producer does
    2 targets x (3 folds + 1 full fit) = 8 fits; a reused consumer adds 0."""
    from src import model_modules

    calls = {"n": 0}
    orig_fit = model_modules.ExtraTreesModel.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(model_modules.ExtraTreesModel, "fit", counting_fit)

    _run(tmp_path / "on", reuse=True, nested=False)
    assert calls["n"] == 8

    calls["n"] = 0
    _run(tmp_path / "off", reuse=False, nested=False)
    assert calls["n"] == 16


def test_no_reuse_when_params_differ(tmp_path, monkeypatch):
    """Differing model_params must miss the cache: both experiments fit fully."""
    from src import model_modules

    calls = {"n": 0}
    orig_fit = model_modules.ExtraTreesModel.fit

    def counting_fit(self, *args, **kwargs):
        calls["n"] += 1
        return orig_fit(self, *args, **kwargs)

    monkeypatch.setattr(model_modules.ExtraTreesModel, "fit", counting_fit)

    configs = _configs(corr_pair=("none", "none"))
    configs[0].exp_id = "EX_a"
    configs[1].exp_id = "EX_b"
    configs[1].model_params = dict(MODEL_PARAMS, n_estimators=20)

    _run(tmp_path, reuse=True, nested=False, configs=configs)
    assert calls["n"] == 16


def test_consumer_joblib_artifact(tmp_path):
    """The consumer artifact replays the producer's model bitwise but owns
    its corr_model and config."""
    _run(tmp_path, reuse=True)

    prod = joblib.load(tmp_path / "models" / "EX_none_T_model.joblib")
    cons = joblib.load(tmp_path / "models" / "EX_segmented_T_model.joblib")

    assert prod["config"]["exp_id"] == "EX_none"
    assert cons["config"]["exp_id"] == "EX_segmented"
    assert cons["config"]["corr_module"] == "segmented"

    assert prod["corr_model"] is None  # NoCorrection.fit returns None
    assert isinstance(cons["corr_model"], dict)
    assert "boundaries" in cons["corr_model"]

    X_probe = np.random.RandomState(3).rand(20, N_FEATURES) * 10.0
    np.testing.assert_array_equal(
        prod["model"].predict(X_probe), cons["model"].predict(X_probe)
    )


def test_cache_key_type_distinct():
    """1 == 1.0 == True under Python equality, but e.g. sklearn
    max_features=1 and 1.0 are different fits — keys must not collide."""
    from src.protocol import _fit_cache_key

    def cfg(mf):
        return ExperimentConfig(
            exp_id="EX", data_module_name="raw", model_module_name="ert",
            corr_module_name="none", model_params={"max_features": mf},
        )

    keys = {_fit_cache_key(cfg(mf), "T") for mf in (1, 1.0, True)}
    assert len(keys) == 3


def test_precomputed_with_uncertainty_raises():
    protocol = StratifiedCVProtocol(n_splits=3, random_seed=0)
    with pytest.raises(ValueError, match="uncertainty"):
        protocol.run(
            np.zeros((10, 2)), np.zeros(10),
            pipeline_factory=lambda seed=None: None,
            uncertainty_module=object(),
            precomputed_folds=[],
        )
