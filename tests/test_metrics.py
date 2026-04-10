# -*- coding: utf-8 -*-
"""Unit tests for evaluation metric utilities."""

import pytest
import numpy as np
from src.metrics import (
    rmse, mae, r2, bias,
    compute_slope_intercept,
    compute_bias_stats,
    summarize_folds
)


class TestBasicMetrics:
    """TestBasicMetrics class."""

    def test_rmse_perfect(self):
        """test_rmse_perfect function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_positive(self):
        """test_rmse_positive function."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.2, 3.3])

        assert rmse(y_true, y_pred) > 0

    def test_mae_perfect(self):
        """test_mae_perfect function."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert mae(y_true, y_pred) == 0.0

    def test_mae_symmetric(self):
        """test_mae_symmetric function."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])

        assert mae(y_true, y_pred) == 1.0

    def test_r2_perfect(self):
        """test_r2_perfect function."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert r2(y_true, y_pred) == 1.0

    def test_r2_bounds(self):
        """test_r2_bounds function."""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 10

        r2_val = r2(y_true, y_pred)
        assert 0 <= r2_val <= 1

    def test_bias_positive_means_underestimate(self):
        """test_bias_positive_means_underestimate function."""
        y_true = np.array([2, 3, 4])
        y_pred = np.array([1, 2, 3])

        assert bias(y_true, y_pred) == 1.0

    def test_bias_negative_means_overestimate(self):
        """test_bias_negative_means_overestimate function."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])

        assert bias(y_true, y_pred) == -1.0


class TestSlopeIntercept:
    """TestSlopeIntercept class."""

    def test_perfect_prediction(self):
        """test_perfect_prediction function."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        slope, intercept = compute_slope_intercept(y_true, y_pred)

        assert np.isclose(slope, 1.0, atol=1e-10)
        assert np.isclose(intercept, 0.0, atol=1e-10)

    def test_linear_bias(self):
        """test_linear_bias function."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([5.0, 15.0, 25.0, 35.0, 45.0])

        slope, intercept = compute_slope_intercept(y_true, y_pred)

        assert np.isclose(slope, 1.0, atol=1e-10)
        assert np.isclose(intercept, 5.0, atol=1e-10)


class TestBiasStats:
    """TestBiasStats class."""

    def test_basic_stats(self):
        """test_basic_stats function."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        stats = compute_bias_stats(y_true, y_pred)

        assert 'bias_mean' in stats
        assert 'resid_std' in stats

    def test_zero_bias(self):
        """test_zero_bias function."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.9, 2.1, 3.0])

        stats = compute_bias_stats(y_true, y_pred)

        assert np.isclose(stats['bias_mean'], 0.0, atol=1e-10)


class TestSummarizeFolds:
    """TestSummarizeFolds class."""

    def test_basic_summary(self):
        """test_basic_summary function."""
        fold_metrics = [
            {'rmse': 30.0, 'r2': 0.93},
            {'rmse': 31.0, 'r2': 0.92},
            {'rmse': 29.0, 'r2': 0.94},
        ]

        summary = summarize_folds(fold_metrics)

        assert 'rmse_mean' in summary
        assert 'rmse_std' in summary
        assert 'r2_mean' in summary
        assert 'r2_std' in summary

    def test_mean_calculation(self):
        """test_mean_calculation function."""
        fold_metrics = [
            {'rmse': 30.0},
            {'rmse': 32.0},
            {'rmse': 31.0},
        ]

        summary = summarize_folds(fold_metrics)

        assert np.isclose(summary['rmse_mean'], 31.0)

    def test_with_ci(self):
        """test_with_ci function."""
        fold_metrics = [
            {'rmse': 30.0 + i * 0.5} for i in range(10)
        ]

        summary = summarize_folds(fold_metrics, compute_ci=True)

        assert 'rmse_ci_lower' in summary
        assert 'rmse_ci_upper' in summary
        assert summary['rmse_ci_lower'] < summary['rmse_mean']
        assert summary['rmse_ci_upper'] > summary['rmse_mean']

    def test_excludes_non_numeric(self):
        """test_excludes_non_numeric function."""
        fold_metrics = [
            {'fold_id': 0, 'rmse': 30.0},
            {'fold_id': 1, 'rmse': 31.0},
        ]

        summary = summarize_folds(fold_metrics)

        assert 'fold_id_mean' not in summary
        assert 'rmse_mean' in summary

