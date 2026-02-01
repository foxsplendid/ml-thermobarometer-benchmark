# -*- coding: utf-8 -*-
"""
M3 校正模块测试

测试 NoCorrection, ResidualRegressionCorrector, SegmentedLinearCorrector
"""

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
    """
    生成带系统性偏差的预测数据

    模拟：预测值 = 真实值 * 0.85 + 150 + 噪声
    """
    np.random.seed(42)
    n_samples = 100

    y_true = np.linspace(800, 1400, n_samples)
    y_pred = y_true * 0.85 + 150 + np.random.randn(n_samples) * 15

    return y_true, y_pred


class TestNoCorrection:
    """NoCorrection 测试"""

    def test_fit_returns_none(self, biased_predictions):
        """fit 返回 None"""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)

        assert corr_model is None

    def test_apply_returns_copy(self, biased_predictions):
        """apply 返回输入的副本"""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        assert np.allclose(y_corr, y_pred)
        # 确保是副本而非引用
        y_corr[0] = -999
        assert y_pred[0] != -999

    def test_get_correction_params(self, biased_predictions):
        """获取校正参数"""
        y_true, y_pred = biased_predictions
        module = NoCorrection()

        corr_model = module.fit(y_true, y_pred)
        params = module.get_correction_params(corr_model)

        assert params['method'] == 'none'


class TestResidualRegressionCorrector:
    """ResidualRegressionCorrector 测试"""

    def test_reduces_rmse(self, biased_predictions):
        """校正后 RMSE 应降低"""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        rmse_before = np.sqrt(np.mean((y_true - y_pred) ** 2))
        rmse_after = np.sqrt(np.mean((y_true - y_corr) ** 2))

        assert rmse_after < rmse_before

    def test_slope_adjustment(self, biased_predictions):
        """校正后斜率应更接近 1"""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        # 校正前斜率
        from scipy.stats import linregress
        slope_before = linregress(y_pred, y_true).slope
        slope_after = linregress(y_corr, y_true).slope

        # 校正后斜率更接近 1
        assert abs(slope_after - 1.0) < abs(slope_before - 1.0)

    def test_get_correction_params(self, biased_predictions):
        """获取校正参数"""
        y_true, y_pred = biased_predictions
        module = ResidualRegressionCorrector()

        corr_model = module.fit(y_true, y_pred)
        params = module.get_correction_params(corr_model)

        assert 'slope_before' in params
        assert 'intercept_before' in params


class TestSegmentedLinearCorrector:
    """SegmentedLinearCorrector 测试"""

    def test_fit_creates_segments(self, biased_predictions):
        """fit 创建分段模型"""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(n_segments=3)

        corr_model = module.fit(y_true, y_pred)

        assert 'boundaries' in corr_model
        assert 'segment_models' in corr_model
        assert len(corr_model['boundaries']) == 4  # 3 segments = 4 boundaries

    def test_apply_within_bounds(self, biased_predictions):
        """apply 在训练范围内正常工作"""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(n_segments=3)

        corr_model = module.fit(y_true, y_pred)
        y_corr = module.apply(corr_model, y_pred)

        assert len(y_corr) == len(y_pred)
        assert not np.any(np.isnan(y_corr))

    def test_clip_to_train_range(self, biased_predictions):
        """clip_to_train_range 选项"""
        y_true, y_pred = biased_predictions
        module = SegmentedLinearCorrector(clip_to_train_range=True)

        corr_model = module.fit(y_true, y_pred)

        # 测试超出训练范围的值
        y_extreme = np.array([500, 2000])  # 远超训练范围
        y_corr = module.apply(corr_model, y_extreme)

        # 应该被裁剪到训练范围
        assert y_corr[0] >= corr_model['y_min']
        assert y_corr[1] <= corr_model['y_max']


class TestGetCorrectionModule:
    """工厂函数测试"""

    def test_get_none(self):
        """获取 none 模块"""
        module = get_correction_module('none')
        assert isinstance(module, NoCorrection)

    def test_get_residual(self):
        """获取 residual 模块"""
        module = get_correction_module('residual')
        assert isinstance(module, ResidualRegressionCorrector)

    def test_get_segmented(self):
        """获取 segmented 模块"""
        module = get_correction_module('segmented')
        assert isinstance(module, SegmentedLinearCorrector)

    def test_invalid_name(self):
        """无效名称抛出异常"""
        with pytest.raises(ValueError):
            get_correction_module('invalid_correction')
