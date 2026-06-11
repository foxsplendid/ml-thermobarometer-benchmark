# -*- coding: utf-8 -*-
"""Tests for S4: load_data missing-value guard (no silent zero-fill)."""

import os
import numpy as np
import pytest

from main import load_data
from config import get_config_dict


def _mini_config(tmp_path, rows):
    csv_path = tmp_path / "mini.csv"
    csv_path.write_text("SiO2.cpx,TiO2.cpx,T,P\n" + "\n".join(rows), encoding="utf-8")
    return {
        'data_path': str(csv_path),
        'data_encoding': 'utf-8',
        'feature_sets': {'Mini': ['SiO2.cpx', 'TiO2.cpx']},
        'target_T': 'T',
        'target_P': 'P',
    }


@pytest.mark.skipif(
    not os.path.exists(get_config_dict()['data_path']),
    reason="input.csv not present",
)
def test_input_csv_features_finite():
    """Guards against silent dataset drift: current data must be NaN-free."""
    cfg = get_config_dict()
    for feature_set in cfg['feature_sets']:
        X, y_T, y_P = load_data(cfg, feature_set=feature_set)
        assert np.isfinite(X).all()
        assert np.isfinite(y_T).all()
        assert np.isfinite(y_P).all()


def test_load_data_raises_on_nan_feature(tmp_path):
    cfg = _mini_config(tmp_path, ["52.1,0.4,1100,5", "51.0,,1150,6"])
    with pytest.raises(ValueError, match="TiO2.cpx"):
        load_data(cfg, feature_set='Mini')


def test_load_data_raises_on_nan_target(tmp_path):
    cfg = _mini_config(tmp_path, ["52.1,0.4,1100,5", "51.0,0.3,,6"])
    with pytest.raises(ValueError, match="NaN found in targets"):
        load_data(cfg, feature_set='Mini')


def test_load_data_keeps_zeros(tmp_path):
    """Exact zeros are the upstream below-detection convention — kept as-is."""
    cfg = _mini_config(tmp_path, ["52.1,0.0,1100,5", "51.0,0.3,1150,6"])
    X, y_T, y_P = load_data(cfg, feature_set='Mini')
    assert X[0, 1] == 0.0
    assert np.isfinite(X).all()
