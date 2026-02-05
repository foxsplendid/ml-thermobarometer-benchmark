# -*- coding: utf-8 -*-
"""
指标计算测试

测试 metrics.py 中的各种指标函数
"""

import pytest
import numpy as np
from src.metrics import (
    rmse, mae, r2, mape, bias,
    compute_slope_intercept,
    compute_bias_stats,
    compute_metrics,
    summarize_folds
)


class TestBasicMetrics:
    """基础指标函数测试"""

    def test_rmse_perfect(self):
        """完美预测 RMSE 为 0"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert rmse(y_true, y_pred) == 0.0

    def test_rmse_positive(self):
        """RMSE 应为正数"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.2, 3.3])

        assert rmse(y_true, y_pred) > 0

    def test_mae_perfect(self):
        """完美预测 MAE 为 0"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])

        assert mae(y_true, y_pred) == 0.0

    def test_mae_symmetric(self):
        """MAE 对称性"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # 全部高估 1

        assert mae(y_true, y_pred) == 1.0

    def test_r2_perfect(self):
        """完美预测 R² 为 1"""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        assert r2(y_true, y_pred) == 1.0

    def test_r2_bounds(self):
        """R² 通常在 [0, 1] 范围（好的预测）"""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 10  # 小噪声

        r2_val = r2(y_true, y_pred)
        assert 0 <= r2_val <= 1

    def test_bias_positive_means_underestimate(self):
        """正偏差表示低估"""
        y_true = np.array([2, 3, 4])
        y_pred = np.array([1, 2, 3])  # 低估 1

        assert bias(y_true, y_pred) == 1.0

    def test_bias_negative_means_overestimate(self):
        """负偏差表示高估"""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # 高估 1

        assert bias(y_true, y_pred) == -1.0


class TestSlopeIntercept:
    """compute_slope_intercept 测试"""

    def test_perfect_prediction(self):
        """完美预测斜率为1，截距为0"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        slope, intercept = compute_slope_intercept(y_true, y_pred)

        assert np.isclose(slope, 1.0, atol=1e-10)
        assert np.isclose(intercept, 0.0, atol=1e-10)

    def test_linear_bias(self):
        """线性偏差检测"""
        y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        y_pred = np.array([5.0, 15.0, 25.0, 35.0, 45.0])  # 全部低估 5

        slope, intercept = compute_slope_intercept(y_true, y_pred)

        # 斜率仍为 1，截距为 5
        assert np.isclose(slope, 1.0, atol=1e-10)
        assert np.isclose(intercept, 5.0, atol=1e-10)


class TestBiasStats:
    """compute_bias_stats 测试"""

    def test_basic_stats(self):
        """基本偏差统计"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 1.9, 3.2])

        stats = compute_bias_stats(y_true, y_pred)

        assert 'bias_mean' in stats
        assert 'resid_std' in stats

    def test_zero_bias(self):
        """无偏情况"""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([0.9, 2.1, 3.0])  # 均值偏差为 0

        stats = compute_bias_stats(y_true, y_pred)

        assert np.isclose(stats['bias_mean'], 0.0, atol=1e-10)


class TestComputeMetrics:
    """compute_metrics 测试"""

    def test_returns_all_metrics(self):
        """返回所有指标"""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        metrics = compute_metrics(y_true, y_pred)

        expected_keys = ['rmse', 'mae', 'r2', 'slope', 'intercept', 'bias_mean', 'resid_std']
        for key in expected_keys:
            assert key in metrics

    def test_with_prefix(self):
        """带前缀"""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1

        metrics = compute_metrics(y_true, y_pred, prefix='T_')

        assert 'T_rmse' in metrics
        assert 'T_r2' in metrics


class TestSummarizeFolds:
    """summarize_folds 测试"""

    def test_basic_summary(self):
        """基本汇总"""
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
        """均值计算正确"""
        fold_metrics = [
            {'rmse': 30.0},
            {'rmse': 32.0},
            {'rmse': 31.0},
        ]

        summary = summarize_folds(fold_metrics)

        assert np.isclose(summary['rmse_mean'], 31.0)

    def test_with_ci(self):
        """带置信区间"""
        fold_metrics = [
            {'rmse': 30.0 + i * 0.5} for i in range(10)
        ]

        summary = summarize_folds(fold_metrics, compute_ci=True)

        assert 'rmse_ci_lower' in summary
        assert 'rmse_ci_upper' in summary
        assert summary['rmse_ci_lower'] < summary['rmse_mean']
        assert summary['rmse_ci_upper'] > summary['rmse_mean']

    def test_excludes_non_numeric(self):
        """排除非数值列"""
        fold_metrics = [
            {'fold_id': 0, 'rmse': 30.0},
            {'fold_id': 1, 'rmse': 31.0},
        ]

        summary = summarize_folds(fold_metrics)

        assert 'fold_id_mean' not in summary
        assert 'rmse_mean' in summary
