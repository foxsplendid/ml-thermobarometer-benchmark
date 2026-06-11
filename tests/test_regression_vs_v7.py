# -*- coding: utf-8 -*-
"""Slow regression test: V8 in V7-equivalent configuration must reproduce
the V7 golden numbers (tests/golden_v7_metrics_summary.csv, a frozen mirror
of the V7-era results/metrics_summary.csv — results/ itself is overwritten
by the V8 baseline rerun, so the live file can no longer serve as reference).

Run explicitly with:  pytest -m slow tests/test_regression_vs_v7.py

Scope: E01 ERT raw none, Liquid feature set, full 10-fold CV (~2-3 min).
With nested_correction=False and merge_sparse_bins=False the V8 code path is
line-identical to V7; the tolerance is 1e-4 relative rather than bit-exact
because tree construction can dither at machine precision under different
thread counts (SPEC §7).
"""

import os
import numpy as np
import pandas as pd
import pytest

from config import get_config_dict

GOLDEN_EXP_ID = "E01_ert_raw_none_liq"
METRIC_COLS = ["T_rmse_mean", "P_rmse_mean", "T_r2_mean", "P_r2_mean"]
REL_TOL = 1e-4


def _golden_path():
    return os.path.join(os.path.dirname(__file__), "golden_v7_metrics_summary.csv")


@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.exists(_golden_path()), reason="V7 golden metrics_summary.csv missing"
)
def test_e01_liquid_matches_v7(tmp_path, monkeypatch):
    # Stable thread count: numeric dither across thread counts is the reason
    # for the 1e-4 tolerance; pin it anyway to minimise variance (SPEC §7).
    monkeypatch.setenv("ML_N_JOBS", "2")
    monkeypatch.setenv("ML_RESERVE_CORES", "0")

    from main import load_data, prepare_splits, get_experiment_configs
    from src.protocol import ExperimentMatrix

    cfg = get_config_dict()
    if not os.path.exists(cfg["data_path"]):
        pytest.skip("input.csv not present")

    golden = pd.read_csv(_golden_path())
    golden_row = golden[golden["exp_id"] == GOLDEN_EXP_ID]
    assert len(golden_row) == 1, f"golden row {GOLDEN_EXP_ID} not found"

    X, y_T, y_P = load_data(cfg, feature_set="Liquid")
    split_data = prepare_splits(X, y_T, y_P, cfg)
    train_idx = split_data["train_idx"]
    test_idx = split_data["test_idx"]

    e01_liq = [
        c for c in get_experiment_configs()
        if c.exp_id == GOLDEN_EXP_ID and c.feature_set == "Liquid"
    ]
    assert len(e01_liq) == 1

    matrix = ExperimentMatrix(
        X=X[train_idx], y_T=y_T[train_idx], y_P=y_P[train_idx],
        output_dir=str(tmp_path),
    )
    summary_df = matrix.run_experiments(
        configs=e01_liq,
        n_splits=cfg["n_splits"],
        stratify_labels=split_data["tp_bins_train"],
        X_test=X[test_idx],
        y_T_test=y_T[test_idx],
        y_P_test=y_P[test_idx],
        random_seed=cfg["random_seed"],
        # V7-equivalent configuration:
        nested_correction=False,
        merge_sparse_bins=False,
        verbose=False,
    )

    row = summary_df[summary_df["exp_id"] == GOLDEN_EXP_ID].iloc[0]
    for col in METRIC_COLS:
        got = float(row[col])
        want = float(golden_row.iloc[0][col])
        assert got == pytest.approx(want, rel=REL_TOL), (
            f"{col}: V8={got!r} vs V7 golden={want!r} (rel tol {REL_TOL})"
        )
