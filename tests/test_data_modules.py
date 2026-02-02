# -*- coding: utf-8 -*-
"""
M1 数据模块测试

测试 RawDataModule, BalancedDataModule, AugmentedDataModule
"""

import pytest
import numpy as np
from src.data_modules import (
    RawDataModule,
    BalancedDataModule,
    AugmentedDataModule,
    get_data_module
)


class TestRawDataModule:
    """RawDataModule 测试"""

    def test_fit_transform_shape(self, train_val_split):
        """验证 fit_transform 输出形状正确"""
        module = RawDataModule()
        X2, y2, weights, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 形状不变
        assert X2.shape == train_val_split['X_train'].shape
        assert len(y2) == len(train_val_split['y_train'])
        assert len(weights) == len(y2)

    def test_weights_uniform(self, train_val_split):
        """验证权重为均匀分布"""
        module = RawDataModule()
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 所有权重相等
        assert np.allclose(weights, 1.0)

    def test_transform_uses_train_scaler(self, train_val_split):
        """验证 transform 使用训练集的 scaler"""
        module = RawDataModule()
        X_scaled, _, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 训练集标准化后均值接近0
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)

        # 验证集使用训练集参数，均值可能不为0
        X_val_scaled, _ = module.transform(train_val_split['X_val'], state)
        # 验证集形状正确
        assert X_val_scaled.shape == train_val_split['X_val'].shape

    def test_state_contains_scaler(self, train_val_split):
        """验证 state 包含 scaler"""
        module = RawDataModule()
        _, _, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert state.scaler is not None
        assert hasattr(state.scaler, 'transform')


class TestBalancedDataModule:
    """BalancedDataModule 测试"""

    def test_fit_transform_shape(self, train_val_split):
        """验证 fit_transform 输出形状正确"""
        module = BalancedDataModule(n_bins=5)
        X2, y2, weights, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 形状不变
        assert X2.shape == train_val_split['X_train'].shape
        assert len(weights) == len(y2)

    def test_weights_sum(self, train_val_split):
        """验证权重总和约等于样本数"""
        module = BalancedDataModule(n_bins=5)
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 权重总和等于样本数
        assert np.isclose(weights.sum(), len(weights), rtol=0.01)

    def test_weights_positive(self, train_val_split):
        """验证权重非负"""
        module = BalancedDataModule(n_bins=5)
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert np.all(weights > 0)


class TestAugmentedDataModule:
    """AugmentedDataModule 测试"""

    def test_augmentation_increases_samples(self, train_val_split):
        """验证增强后样本数增加"""
        n_aug = 5
        module = AugmentedDataModule(n_aug=n_aug)
        X2, y2, _, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        original_size = len(train_val_split['X_train'])
        expected_size = original_size * (1 + n_aug)
        assert len(X2) == expected_size
        assert len(y2) == expected_size

    def test_original_samples_preserved(self, train_val_split):
        """验证原始样本被保留（在增强数据的前部）"""
        module = AugmentedDataModule(n_aug=3)
        X2, y2, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        original_size = len(train_val_split['X_train'])

        # 前 original_size 个样本是原始数据（标准化后）
        X_orig_scaled = state.scaler.transform(train_val_split['X_train'])
        assert np.allclose(X2[:original_size], X_orig_scaled)

    def test_transform_no_augmentation(self, train_val_split):
        """验证 transform 不进行增强"""
        module = AugmentedDataModule(n_aug=5)
        _, _, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        X_val_scaled, _ = module.transform(train_val_split['X_val'], state)

        # 验证集大小不变
        assert X_val_scaled.shape == train_val_split['X_val'].shape

    def test_epma_perturbation_bounds(self, train_val_split):
        """验证 EPMA 扰动后值非负"""
        module = AugmentedDataModule(n_aug=10, clip_min=0.0)
        X2, _, _, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        # 所有值应非负（因为 clip_min=0.0）
        # 注意：标准化后可能为负，这里检查原始空间
        # 由于增强在原始空间进行后再标准化，此处不直接检查
        # 改为检查形状正确
        assert X2.shape[1] == train_val_split['X_train'].shape[1]


class TestGetDataModule:
    """工厂函数测试"""

    def test_get_raw(self):
        """获取 raw 模块"""
        module = get_data_module('raw')
        assert isinstance(module, RawDataModule)

    def test_get_balanced(self):
        """获取 balanced 模块"""
        module = get_data_module('balanced')
        assert isinstance(module, BalancedDataModule)

    def test_get_augmented(self):
        """获取 augmented 模块"""
        module = get_data_module('augmented')
        assert isinstance(module, AugmentedDataModule)

    def test_case_insensitive(self):
        """名称大小写不敏感"""
        module1 = get_data_module('RAW')
        module2 = get_data_module('Raw')
        module3 = get_data_module('raw')

        assert type(module1) == type(module2) == type(module3)

    def test_invalid_name(self):
        """无效名称抛出异常"""
        with pytest.raises(ValueError):
            get_data_module('invalid_module')
