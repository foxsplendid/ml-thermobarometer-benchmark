# -*- coding: utf-8 -*-
"""Smoke tests for tool scripts (marked slow; run with pytest --runslow).

These tests invoke the tool entry points directly via their main() functions to
verify that imports, argparse, and data-loading wiring are correct. They do NOT
run full experiments — they rely on main.py --test having already produced
output in results_test/.
"""

import os
import sys
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.mark.slow
def test_tools_common_imports():
    """tools/_common.py must import without error and expose required names."""
    from tools._common import (
        BASE_CONFIG,
        add_common_args,
        build_experiment_config,
        init_tool_logging,
    )
    assert isinstance(BASE_CONFIG, dict)
    assert callable(add_common_args)
    assert callable(build_experiment_config)
    assert callable(init_tool_logging)


@pytest.mark.slow
def test_build_experiment_config_ert():
    """build_experiment_config should return an ExperimentConfig with correct fields."""
    from tools._common import BASE_CONFIG, build_experiment_config
    from src.protocol import ExperimentConfig

    cfg = build_experiment_config(
        exp_id="test_ert",
        data_module="raw",
        model_module="ert",
        corr_module="none",
        feature_set="Liquid",
        random_seed=0,
    )
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.exp_id == "test_ert"
    assert cfg.model_module_name == "ert"
    assert cfg.data_module_name == "raw"
    assert "random_seed" in cfg.model_params or "n_estimators" in cfg.model_params


@pytest.mark.slow
def test_add_common_args_parses_defaults():
    """add_common_args defaults must parse without error."""
    import argparse
    from tools._common import add_common_args

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args([])

    assert args.feature_set == "Liquid"
    assert args.data_module == "augmented"
    assert args.model_module == "ert"
    assert args.corr_module == "none"
    assert args.n_splits == 10
    assert args.random_seed == 42


@pytest.mark.slow
def test_fold_parallel_determinism():
    """Sequential and 2-worker-parallel folds must produce identical OOF predictions."""
    import numpy as np
    import os
    from unittest.mock import patch

    from src.protocol import Pipeline, StratifiedCVProtocol
    from src.data_modules import RawDataModule
    from src.model_modules import ExtraTreesModel
    from src.correction_modules import NoCorrection

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 9))
    y = rng.standard_normal(120) * 100 + 1000
    stratify = (y > np.median(y)).astype(int)

    def factory(seed=42):
        return Pipeline(
            RawDataModule(random_seed=seed),
            ExtraTreesModel(n_estimators=10, random_seed=seed, n_jobs=1),
            NoCorrection(),
        )

    # Sequential run.
    with patch.dict(os.environ, {"ML_FOLD_WORKERS": "1"}):
        proto_seq = StratifiedCVProtocol(n_splits=3, random_seed=42)
        res_seq = proto_seq.run(X, y, factory, stratify_labels=stratify, verbose=False)

    # Parallel run (threading backend, 2 workers).
    with patch.dict(os.environ, {"ML_FOLD_WORKERS": "2", "ML_FOLD_BACKEND": "threading"}):
        proto_par = StratifiedCVProtocol(n_splits=3, random_seed=42)
        res_par = proto_par.run(X, y, factory, stratify_labels=stratify, verbose=False)

    pred_seq = res_seq["predictions"].sort_values("sample_idx")["y_pred_corr"].values
    pred_par = res_par["predictions"].sort_values("sample_idx")["y_pred_corr"].values

    np.testing.assert_allclose(
        pred_seq, pred_par, rtol=1e-10,
        err_msg="Parallel and sequential folds disagree — seed or ordering issue.",
    )
