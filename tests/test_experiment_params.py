# -*- coding: utf-8 -*-
"""Tests for experiment_params: H4 fast tier and matrix leak protection."""

import numpy as np
import pytest

from config import get_config_dict
from src.experiment_params import FAST_TIER_OVERRIDES, build_model_params


@pytest.fixture
def cfg():
    return get_config_dict()


class TestFastTier:

    def test_default_tier_identical(self, cfg):
        """Omitting tier must equal tier='full' and stay pinned to the
        scientific values — the fast tier must never drift into defaults."""
        for m in ["ert", "catboost", "rf", "stacking"]:
            assert build_model_params(cfg, m) == build_model_params(cfg, m, tier="full")

        ert = build_model_params(cfg, "ert")
        assert ert["n_estimators"] == 200
        assert ert["max_depth"] == 15
        cb = build_model_params(cfg, "catboost")
        assert cb["iterations"] == 1000
        assert cb["depth"] == 6
        assert cb["learning_rate"] == 0.03
        assert build_model_params(cfg, "stacking")["inner_cv"] == 5

    def test_fast_tier_values(self, cfg):
        ert = build_model_params(cfg, "ert", tier="fast")
        assert ert["n_estimators"] == 50
        assert ert["max_depth"] == 10
        assert ert["min_samples_split"] == 5  # un-overridden key keeps default

        cb = build_model_params(cfg, "catboost", tier="fast")
        assert cb["iterations"] == 200
        assert cb["depth"] == 4
        assert cb["learning_rate"] == 0.1
        assert cb["loss_function"] == "RMSE"  # un-overridden key keeps default

        st = build_model_params(cfg, "stacking", tier="fast")
        assert st["inner_cv"] == 3
        assert st["base_model_params"]["ert"]["n_estimators"] == 50
        assert st["base_model_params"]["catboost"]["iterations"] == 200
        assert st["base_model_params"]["rf"]["n_estimators"] == 50

    def test_unknown_tier_raises(self, cfg):
        with pytest.raises(ValueError, match="tier"):
            build_model_params(cfg, "ert", tier="turbo")

    def test_unknown_module_fast_stays_empty(self, cfg):
        """Pinned: unknown modules have no fast overrides; the {} fallback
        stays untouched, same as tier='full'."""
        assert build_model_params(cfg, "ridge", tier="fast") == {}
        assert build_model_params(cfg, "ridge", tier="full") == {}


class TestMatrixLeakProtection:

    def test_main_matrix_never_fast(self):
        """Leak anchor: the 24-config benchmark matrix is pinned to the
        scientific tier; a future caller passing tier='fast' into the
        benchmark path would trip this test."""
        import main as main_mod

        configs = main_mod.get_experiment_configs()
        assert len(configs) == 24
        for c in configs:
            mp = c.model_params
            if c.model_module_name == "ert":
                assert mp["n_estimators"] == 200
                assert mp["max_depth"] == 15
            elif c.model_module_name == "catboost":
                assert mp["iterations"] == 1000
                assert mp["depth"] == 6
                assert mp["learning_rate"] == 0.03
            elif c.model_module_name == "stacking":
                assert mp["inner_cv"] == 5
                assert mp["base_model_params"]["ert"]["n_estimators"] == 200
                assert mp["base_model_params"]["catboost"]["iterations"] == 1000

    def test_quick_configs_fast_tier(self):
        import main as main_mod

        configs = main_mod.get_experiment_configs(tier="fast")
        e01 = next(c for c in configs if c.exp_id.startswith("E01"))
        e02 = next(c for c in configs if c.exp_id.startswith("E02"))
        assert e01.model_params["n_estimators"] == 50
        assert e02.model_params["iterations"] == 200


class TestFastParamsConstructAndFit:

    def test_fast_params_construct_and_fit(self, cfg):
        """Guards against FAST_TIER_OVERRIDES key names drifting from the
        model constructor signatures."""
        from src.model_modules import ExtraTreesModel, CatBoostModel

        rng = np.random.RandomState(0)
        X = rng.rand(100, 10)
        y = rng.rand(100) * 100.0

        ert = ExtraTreesModel(random_seed=42, **build_model_params(cfg, "ert", tier="fast"))
        fitted = ert.fit(X, y)
        assert np.isfinite(ert.predict(fitted, X)).all()

        cb = CatBoostModel(random_seed=42, **build_model_params(cfg, "catboost", tier="fast"))
        fitted = cb.fit(X, y)
        assert np.isfinite(cb.predict(fitted, X)).all()
