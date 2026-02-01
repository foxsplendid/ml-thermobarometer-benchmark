# -*- coding: utf-8 -*-
"""
M2 模型模块测试

测试 ExtraTreesModel, CatBoostModel, StrictOOFStacking
"""

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
    """ExtraTreesModel 测试"""

    def test_fit_and_predict(self, train_val_split):
        """基本训练和预测"""
        module = ExtraTreesModel(n_estimators=10, max_depth=5, random_seed=42)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_with_sample_weights(self, train_val_split):
        """带样本权重训练"""
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
        """特征重要性获取"""
        module = ExtraTreesModel(n_estimators=10, random_seed=42)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        importance = module.get_feature_importance(model)

        assert len(importance) == train_val_split['X_train'].shape[1]
        assert np.isclose(importance.sum(), 1.0, rtol=0.01)  # 重要性和为1


class TestCatBoostModel:
    """CatBoostModel 测试"""

    def test_fit_and_predict(self, train_val_split):
        """基本训练和预测"""
        module = CatBoostModel(iterations=10, depth=3, random_seed=42, silent=True)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])
        assert not np.any(np.isnan(y_pred))

    def test_training_time_recorded(self, train_val_split):
        """训练时间记录"""
        module = CatBoostModel(iterations=10, random_seed=42, silent=True)

        module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert module.get_training_time() > 0


class TestStrictOOFStacking:
    """StrictOOFStacking 测试"""

    def test_fit_and_predict(self, train_val_split):
        """基本训练和预测"""
        # 使用简化的基模型加速测试
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
        """OOF 预测生成"""
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

        # 获取 OOF 预测
        y_oof = module.get_oof_predictions(
            model,
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert len(y_oof) == len(train_val_split['y_train'])

    def test_base_correlations(self, train_val_split):
        """基模型相关性矩阵"""
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
    """RidgeModel（元模型）测试"""

    def test_fit_and_predict(self, train_val_split):
        """基本训练和预测"""
        module = RidgeModel(alpha=1.0)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        y_pred = module.predict(model, train_val_split['X_val'])

        assert len(y_pred) == len(train_val_split['X_val'])

    def test_get_weights(self, train_val_split):
        """获取回归系数"""
        module = RidgeModel(alpha=1.0)

        model = module.fit(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        weights = module.get_weights(model)

        assert len(weights) == train_val_split['X_train'].shape[1]


class TestGetModelModule:
    """工厂函数测试"""

    def test_get_ert(self):
        """获取 ert 模块"""
        module = get_model_module('ert')
        assert isinstance(module, ExtraTreesModel)

    def test_get_catboost(self):
        """获取 catboost 模块"""
        module = get_model_module('catboost')
        assert isinstance(module, CatBoostModel)

    def test_get_stacking(self):
        """获取 stacking 模块"""
        module = get_model_module('stacking')
        assert isinstance(module, StrictOOFStacking)

    def test_aliases(self):
        """别名支持"""
        m1 = get_model_module('ert')
        m2 = get_model_module('extratrees')
        assert type(m1) == type(m2)

        m3 = get_model_module('rf')
        m4 = get_model_module('randomforest')
        assert type(m3) == type(m4)

    def test_invalid_name(self):
        """无效名称抛出异常"""
        with pytest.raises(ValueError):
            get_model_module('invalid_model')
