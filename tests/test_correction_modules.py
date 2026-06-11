# -*- coding: utf-8 -*-
"""Unit tests for prediction-correction modules."""

import pytest
import numpy as np
from src.correction_modules import (
    NoCorrection,
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


class TestEdgeExtrapolation:
    """S5 (V8): behaviour outside the fitted boundary range."""

    def test_in_range_identical_to_raw_mode(self, biased_predictions):
        """Edge handling must not perturb in-range corrections."""
        y_true, y_pred = biased_predictions
        offset_corr = SegmentedLinearCorrector(edge_mode='offset').fit(y_true, y_pred)
        raw_corr = SegmentedLinearCorrector(edge_mode='raw').fit(y_true, y_pred)

        module = SegmentedLinearCorrector()
        np.testing.assert_allclose(
            module.apply(offset_corr, y_pred), module.apply(raw_corr, y_pred)
        )

    def test_boundary_continuity(self, biased_predictions):
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(edge_mode='offset', clip_to_train_range=False)
        corr_model = module.fit(y_true, y_pred)

        lo, hi = corr_model['boundaries'][0], corr_model['boundaries'][-1]
        eps = 1e-9
        f = lambda v: module.apply(corr_model, np.array([v]))[0]
        assert abs(f(lo - eps) - f(lo)) < 1e-6
        assert abs(f(hi + eps) - f(hi)) < 1e-6

    def test_slope_one_continuation(self, biased_predictions):
        """Beyond the data range: constant additive offset, order-preserving."""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(edge_mode='offset', clip_to_train_range=False)
        corr_model = module.fit(y_true, y_pred)

        lo, hi = corr_model['boundaries'][0], corr_model['boundaries'][-1]
        below = np.array([lo - 200, lo - 100, lo - 1])
        above = np.array([hi + 1, hi + 100, hi + 200])
        f_below = module.apply(corr_model, below)
        f_above = module.apply(corr_model, above)

        np.testing.assert_allclose(f_below - below, (f_below - below)[0])
        np.testing.assert_allclose(f_above - above, (f_above - above)[0])
        assert np.all(np.diff(f_below) > 0) and np.all(np.diff(f_above) > 0)

    def test_raw_mode_reproduces_v7_passthrough(self, biased_predictions):
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(edge_mode='raw', clip_to_train_range=False)
        corr_model = module.fit(y_true, y_pred)

        out_of_range = np.array([corr_model['boundaries'][0] - 50,
                                 corr_model['boundaries'][-1] + 50])
        np.testing.assert_array_equal(module.apply(corr_model, out_of_range), out_of_range)

    def test_v7_persisted_model_without_key_behaves_raw(self, biased_predictions):
        """A corr_model dict lacking 'edge_mode' (V7 joblib) must replay raw."""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(clip_to_train_range=False)
        corr_model = module.fit(y_true, y_pred)
        del corr_model['edge_mode']

        out_of_range = np.array([corr_model['boundaries'][0] - 50])
        np.testing.assert_array_equal(module.apply(corr_model, out_of_range), out_of_range)

    def test_offset_respects_train_range_clip(self, biased_predictions):
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(edge_mode='offset', clip_to_train_range=True)
        corr_model = module.fit(y_true, y_pred)

        y_corr = module.apply(corr_model, np.array([500.0, 2000.0]))
        assert y_corr[0] >= corr_model['y_min']
        assert y_corr[1] <= corr_model['y_max']

    def test_invalid_edge_mode_raises(self):
        with pytest.raises(ValueError, match="edge_mode"):
            SegmentedLinearCorrector(edge_mode='extrapolate')


class TestGetCorrectionModule:
    """TestGetCorrectionModule class."""

    def test_get_none(self):
        """test_get_none function."""
        module = get_correction_module('none')
        assert isinstance(module, NoCorrection)

    def test_get_segmented(self):
        """test_get_segmented function."""
        module = get_correction_module('segmented')
        assert isinstance(module, SegmentedLinearCorrector)

    def test_invalid_name(self):
        """test_invalid_name function."""
        with pytest.raises(ValueError):
            get_correction_module('invalid_correction')
