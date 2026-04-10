# -*- coding: utf-8 -*-
"""Unit tests for pipeline and cross-validation protocols."""

import pytest
import numpy as np
from src.protocol import Pipeline, StratifiedCVProtocol
from src.metrics import compute_all_metrics
from src.data_modules import RawDataModule
from src.model_modules import ExtraTreesModel
from src.correction_modules import NoCorrection


class TestPipeline:
    """TestPipeline class."""

    def test_fit_and_predict(self, train_val_split):
        """test_fit_and_predict function."""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10, random_seed=42),
            corr_module=NoCorrection()
        )

        pipeline.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        X_val_scaled, _ = pipeline.data_module.transform(
            train_val_split['X_val'],
            pipeline._state
        )
        y_pred = pipeline.predict(X_val_scaled)

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_predict_from_raw_input(self, train_val_split):
        """test_predict_from_raw_input function."""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10, random_seed=42),
            corr_module=NoCorrection()
        )

        pipeline.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = pipeline.predict_from_raw_input(train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])

    def test_not_fitted_raises(self, train_val_split):
        """test_not_fitted_raises function."""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10),
            corr_module=NoCorrection()
        )

        with pytest.raises(RuntimeError):
            pipeline.predict_from_raw_input(train_val_split['X_val'])

    def test_get_name(self):
        """test_get_name function."""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(),
            corr_module=NoCorrection()
        )

        name = pipeline.get_name()

        assert 'RawDataModule' in name
        assert 'ExtraTreesModel' in name


class TestStratifiedCVProtocol:
    """TestStratifiedCVProtocol class."""

    def test_basic_run(self, sample_data):
        """test_basic_run function."""
        X, y = sample_data

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(
            X, y,
            pipeline_factory,
            verbose=False
        )

        assert 'fold_metrics' in results
        assert 'predictions' in results
        assert 'summary' in results
        assert len(results['fold_metrics']) == 3

    def test_with_stratify_labels(self, sample_data, pt_data):
        """test_with_stratify_labels function."""
        X, y = sample_data
        y_T, _ = pt_data

        bins = np.digitize(y, np.percentile(y, [25, 50, 75]))

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(
            X, y,
            pipeline_factory,
            stratify_labels=bins,
            verbose=False
        )

        assert len(results['fold_metrics']) == 3

    def test_predictions_cover_all_samples(self, sample_data):
        """test_predictions_cover_all_samples function."""
        X, y = sample_data

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(X, y, pipeline_factory, verbose=False)

        pred_indices = results['predictions']['sample_idx'].values
        assert len(np.unique(pred_indices)) == len(X)


class TestComputeAllMetrics:
    """TestComputeAllMetrics class."""

    def test_basic_metrics(self):
        """test_basic_metrics function."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 20

        metrics = compute_all_metrics(y_true, y_pred)

        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'mbe' in metrics
        assert 'slope' in metrics
        assert 'intercept' in metrics

    def test_with_raw_predictions(self):
        """test_with_raw_predictions function."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 20
        y_pred_raw = y_pred + 50

        metrics = compute_all_metrics(y_true, y_pred, y_pred_raw)

        assert 'rmse_raw' in metrics
        assert 'mae_raw' in metrics

    def test_r2_bounds(self):
        """test_r2_bounds function."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 10

        metrics = compute_all_metrics(y_true, y_pred)

        assert 0 <= metrics['r2'] <= 1

