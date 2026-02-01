# -*- coding: utf-8 -*-
"""
协议测试

测试 Pipeline, StratifiedCVProtocol
"""

import pytest
import numpy as np
from src.protocol import Pipeline, StratifiedCVProtocol, compute_all_metrics
from src.data_modules import RawDataModule
from src.model_modules import ExtraTreesModel
from src.correction_modules import NoCorrection


class TestPipeline:
    """Pipeline 测试"""

    def test_fit_and_predict(self, train_val_split):
        """基本训练和预测流程"""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10, random_seed=42),
            corr_module=NoCorrection()
        )

        pipeline.fit(
            train_val_split['X_train'],
            train_val_split['y_train'],
            train_val_split['groups_train']
        )

        # 使用 transform 后的数据预测
        X_val_scaled, _ = pipeline.data_module.transform(
            train_val_split['X_val'],
            pipeline._state
        )
        y_pred = pipeline.predict(X_val_scaled)

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_predict_raw(self, train_val_split):
        """predict_raw 自动处理标准化"""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10, random_seed=42),
            corr_module=NoCorrection()
        )

        pipeline.fit(
            train_val_split['X_train'],
            train_val_split['y_train'],
            train_val_split['groups_train']
        )

        # 直接使用原始数据
        y_pred = pipeline.predict_raw(train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])

    def test_not_fitted_raises(self, train_val_split):
        """未训练时预测应抛出异常"""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(n_estimators=10),
            corr_module=NoCorrection()
        )

        with pytest.raises(RuntimeError):
            pipeline.predict_raw(train_val_split['X_val'])

    def test_get_name(self):
        """获取管道名称"""
        pipeline = Pipeline(
            data_module=RawDataModule(),
            model_module=ExtraTreesModel(),
            corr_module=NoCorrection()
        )

        name = pipeline.get_name()

        assert 'RawDataModule' in name
        assert 'ExtraTreesModel' in name


class TestStratifiedCVProtocol:
    """StratifiedCVProtocol 测试"""

    def test_basic_run(self, sample_data):
        """基本 CV 运行"""
        X, y, groups = sample_data

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(
            X, y, groups,
            pipeline_factory,
            verbose=False
        )

        assert 'fold_metrics' in results
        assert 'predictions' in results
        assert 'summary' in results
        assert len(results['fold_metrics']) == 3

    def test_with_stratify_labels(self, sample_data, pt_data):
        """带分层标签的 CV"""
        X, y, groups = sample_data
        y_T, _ = pt_data

        # 创建分层标签
        bins = np.digitize(y, np.percentile(y, [25, 50, 75]))

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(
            X, y, groups,
            pipeline_factory,
            stratify_labels=bins,
            verbose=False
        )

        assert len(results['fold_metrics']) == 3

    def test_predictions_cover_all_samples(self, sample_data):
        """预测应覆盖所有样本"""
        X, y, groups = sample_data

        protocol = StratifiedCVProtocol(n_splits=3, random_seed=42)

        def pipeline_factory(seed=None):
            return Pipeline(
                data_module=RawDataModule(),
                model_module=ExtraTreesModel(n_estimators=10, random_seed=seed or 42),
                corr_module=NoCorrection()
            )

        results = protocol.run(X, y, groups, pipeline_factory, verbose=False)

        # 所有样本都应该有预测
        pred_indices = results['predictions']['sample_idx'].values
        assert len(np.unique(pred_indices)) == len(X)


class TestComputeAllMetrics:
    """compute_all_metrics 测试"""

    def test_basic_metrics(self):
        """基本指标计算"""
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
        """带原始预测的指标计算"""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 20
        y_pred_raw = y_pred + 50  # 模拟校正前

        metrics = compute_all_metrics(y_true, y_pred, y_pred_raw)

        assert 'rmse_raw' in metrics
        assert 'mae_raw' in metrics

    def test_r2_bounds(self):
        """R² 应在合理范围内"""
        np.random.seed(42)
        y_true = np.random.randn(100) * 100 + 1000
        y_pred = y_true + np.random.randn(100) * 10  # 小噪声

        metrics = compute_all_metrics(y_true, y_pred)

        assert 0 <= metrics['r2'] <= 1
