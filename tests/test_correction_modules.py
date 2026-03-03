# -*- coding: utf-8 -*-
"""Unit tests for prediction-correction modules."""

import pytest
import numpy as np
from src.correction_modules import (
    NoCorrection,
    ResidualRegressionCorrector,
    SegmentedLinearCorrector,
    get_correction_module
)


@pytest.fixture
def biased_predictions():
    """biased_predictions function."""
    np.random.seed(42)
    n_samples = 100

    y_true = np.linspace(800, 1400, n_samples)
    y_pred = y_true * 0.85 + 150 + np.random.randn(n_samples) * 15

    return y_true, y_pred


class TestNoCorrection:
    """TestNoCorrection class."""

    def test_fit_returns_none(self, biased_predictions):
        """test_fit_returns_none function."""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)

        assert corr_model is None

    def test_apply_returns_copy(self, biased_predictions):
        """test_apply_returns_copy function."""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        assert np.allclose(y_corr, y_pred)
        y_corr[0] = -999
        assert y_pred[0] != -999

    def test_get_correction_params(self, biased_predictions):
        """test_get_correction_params function."""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)
        params = module.get_correction_params(corr_model)

        assert params['method'] == 'none'


class TestResidualRegressionCorrector:
    """TestResidualRegressionCorrector class."""

    def test_reduces_rmse(self, biased_predictions):
        """test_reduces_rmse function."""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        rmse_before = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_after = np.sqrt(np.mean((y_true - y_corr) ** 2))

        assert rmse_after < rmse_before

    def test_slope_adjustment(self, biased_predictions):
        """test_slope_adjustment function."""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        from scipy.stats import linregress
        slope_before = linregress(y_pred, y_true).slope
        slope_after = linregress(y_corr, y_true).slope

        assert abs(slope_after - 1.0) < abs(slope_before - 1.0)

    def test_get_correction_params(self, biased_predictions):
        """test_get_correction_params function."""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        params = module.get_correction_params(corr_model)

        assert 'slope_before' in params
        assert 'intercept_before' in params


class TestSegmentedLinearCorrector:
    """TestSegmentedLinearCorrector class."""

    def test_fit_creates_segments(self, biased_predictions):
        """test_fit_creates_segments function."""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(n_segments=3)

        corr_model = module.fit(y_true, y_pred)

        assert 'boundaries' in corr_model
        assert 'segment_models' in corr_model
        assert len(corr_model['boundaries']) == 4  # 3 segments = 4 boundaries

    def test_apply_within_bounds(self, biased_predictions):
        """test_apply_within_bounds function."""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(n_segments=3)

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        assert len(y_corr) == len(y_pred)
        assert not np.any(np.isnan(y_corr))

    def test_clip_to_train_range(self, biased_predictions):
        """test_clip_to_train_range function."""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(clip_to_train_range=True)

        corr_model = module.fit(y_true, y_pred)

        y_extreme = np.array([500, 2000])
        y_corr = module.apply(corr_model, y_extreme)

        assert y_corr[0] >= corr_model['y_min']
        assert y_corr[1] <= corr_model['y_max']


class TestGetCorrectionModule:
    """TestGetCorrectionModule class."""

    def test_get_none(self):
        """test_get_none function."""
        module = get_correction_module('none')
        assert isinstance(module, NoCorrection)

    def test_get_residual(self):
        """test_get_residual function."""
        module = get_correction_module('residual')
        assert isinstance(module, ResidualRegressionCorrector)

    def test_get_segmented(self):
        """test_get_segmented function."""
        module = get_correction_module('segmented')
        assert isinstance(module, SegmentedLinearCorrector)

    def test_invalid_name(self):
        """test_invalid_name function."""
        with pytest.raises(ValueError):
            get_correction_module('invalid_correction')

