# -*- coding: utf-8 -*-
"""Unit tests for model modules and stacking components."""

import pytest
import numpy as np
from src.model_modules import (
    ExtraTreesModel,
    CatBoostModel,
    RandomForestModel,
    StrictOOFStacking,
    RidgeModel,
    get_model_module
)


class TestExtraTreesModel:
    """TestExtraTreesModel class."""

    def test_fit_and_predict(self, train_val_split):
        """test_fit_and_predict function."""
        module = ExtraTreesModel(n_estimators=10, max_depth=5, random_seed=42)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_with_sample_weights(self, train_val_split):
        """test_with_sample_weights function."""
        module = ExtraTreesModel(n_estimators=10, random_seed=42)
        weights = np.ones(len(train_val_split['y_train']))

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train'],
            sample_weights=weights
        )

        y_pred = module.predict(model, train_val_split['X_val'])
        assert len(y_pred) == len(train_val_split['X_val'])

    def test_feature_importance(self, train_val_split):
        """test_feature_importance function."""
        module = ExtraTreesModel(n_estimators=10, random_seed=42)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        importance = module.get_feature_importance(model)

        assert len(importance) == train_val_split['X_train'].shape[1]
        assert np.isclose(importance.sum(), 1.0, rtol=0.01)


class TestCatBoostModel:
    """TestCatBoostModel class."""

    def test_fit_and_predict(self, train_val_split):
        """test_fit_and_predict function."""
        module = CatBoostModel(iterations=10, depth=3, random_seed=42, silent=True)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_training_time_recorded(self, train_val_split):
        """test_training_time_recorded function."""
        module = CatBoostModel(iterations=10, random_seed=42, silent=True)

        module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert module.get_training_time() > 0


class TestStrictOOFStacking:
    """TestStrictOOFStacking class."""

    def test_fit_and_predict(self, train_val_split):
        """test_fit_and_predict function."""
        base_models = [
            ExtraTreesModel(n_estimators=5, max_depth=3, random_seed=42),
            RandomForestModel(n_estimators=5, max_depth=3, random_seed=42),
        ]

        module = StrictOOFStacking(
            base_models=base_models,
            inner_cv=2,
            random_seed=42
        )

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_oof_predictions(self, train_val_split):
        """test_oof_predictions function."""
        base_models = [
            ExtraTreesModel(n_estimators=5, max_depth=3, random_seed=42),
        ]

        module = StrictOOFStacking(
            base_models=base_models,
            inner_cv=2,
            random_seed=42
        )

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_oof = module.get_oof_predictions(
            model,
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert len(y_oof) == len(train_val_split['y_train'])

    def test_base_correlations(self, train_val_split):
        """test_base_correlations function."""
        base_models = [
            ExtraTreesModel(n_estimators=5, random_seed=42),
            RandomForestModel(n_estimators=5, random_seed=42),
        ]

        module = StrictOOFStacking(
            base_models=base_models,
            inner_cv=2,
            random_seed=42
        )

        module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        corr = module.get_base_correlations()

        assert corr is not None
        assert corr.shape == (len(base_models), len(base_models))


class TestRidgeModel:
    """TestRidgeModel class."""

    def test_fit_and_predict(self, train_val_split):
        """test_fit_and_predict function."""
        module = RidgeModel(alpha=1.0)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])

    def test_get_weights(self, train_val_split):
        """test_get_weights function."""
        module = RidgeModel(alpha=1.0)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        weights = module.get_weights(model)

        assert len(weights) == train_val_split['X_train'].shape[1]


class TestGetModelModule:
    """TestGetModelModule class."""

    def test_get_ert(self):
        """test_get_ert function."""
        module = get_model_module('ert')
        assert isinstance(module, ExtraTreesModel)

    def test_get_catboost(self):
        """test_get_catboost function."""
        module = get_model_module('catboost')
        assert isinstance(module, CatBoostModel)

    def test_get_stacking(self):
        """test_get_stacking function."""
        module = get_model_module('stacking')
        assert isinstance(module, StrictOOFStacking)

    def test_aliases(self):
        """test_aliases function."""
        m1 = get_model_module('ert')
        m2 = get_model_module('extratrees')
        assert type(m1) == type(m2)

        m3 = get_model_module('rf')
        m4 = get_model_module('randomforest')
        assert type(m3) == type(m4)

    def test_invalid_name(self):
        """test_invalid_name function."""
        with pytest.raises(ValueError):
            get_model_module('invalid_model')

