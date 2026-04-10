# -*- coding: utf-8 -*-
"""Unit tests for data preprocessing modules."""

import pytest
import numpy as np
from src.data_modules import (
    RawDataModule,
    BalancedDataModule,
    AugmentedDataModule,
    get_data_module
)


class TestRawDataModule:
    """TestRawDataModule class."""

    def test_fit_transform_shape(self, train_val_split):
        """test_fit_transform_shape function."""
        module = RawDataModule()
        X2, y2, weights, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert X2.shape == train_val_split['X_train'].shape
        assert len(y2) == len(train_val_split['y_train'])
        assert len(weights) == len(y2)

    def test_weights_uniform(self, train_val_split):
        """test_weights_uniform function."""
        module = RawDataModule()
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert np.allclose(weights, 1.0)

    def test_transform_uses_train_scaler(self, train_val_split):
        """test_transform_uses_train_scaler function."""
        module = RawDataModule()
        X_scaled, _, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)

        X_val_scaled, _ = module.transform(train_val_split['X_val'], state)
        assert X_val_scaled.shape == train_val_split['X_val'].shape

    def test_state_contains_scaler(self, train_val_split):
        """test_state_contains_scaler function."""
        module = RawDataModule()
        _, _, _, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert state.scaler is not None
        assert hasattr(state.scaler, 'transform')


class TestBalancedDataModule:
    """TestBalancedDataModule class."""

    def test_fit_transform_shape(self, train_val_split):
        """test_fit_transform_shape function."""
        module = BalancedDataModule(n_bins=5)
        X2, y2, weights, state = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert X2.shape == train_val_split['X_train'].shape
        assert len(weights) == len(y2)

    def test_weights_sum(self, train_val_split):
        """test_weights_sum function."""
        module = BalancedDataModule(n_bins=5)
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert np.isclose(weights.sum(), len(weights), rtol=0.01)

    def test_weights_positive(self, train_val_split):
        """test_weights_positive function."""
        module = BalancedDataModule(n_bins=5)
        _, _, weights, _ = module.fit_transform(
            train_val_split['X_train'],
            train_val_split['y_train']
        )

        assert np.all(weights > 0)


class TestAugmentedDataModule:
    """TestAugmentedDataModule class."""

    def test_augmentation_increases_samples(self, sample_data_large):
        """test_augmentation_increases_samples function."""
        from config import DataConfig
        X, y = sample_data_large
        n_aug = 5
        feature_names = DataConfig().feature_sets['Liquid']
        module = AugmentedDataModule(n_aug=n_aug, feature_names=feature_names)
        X2, y2, _, _ = module.fit_transform(X, y)

        expected_size = len(X) * (1 + n_aug)
        assert len(X2) == expected_size
        assert len(y2) == expected_size

    def test_original_samples_preserved(self, sample_data_large):
        """test_original_samples_preserved function."""
        from config import DataConfig
        X, y = sample_data_large
        feature_names = DataConfig().feature_sets['Liquid']
        module = AugmentedDataModule(n_aug=3, feature_names=feature_names)
        X2, y2, _, state = module.fit_transform(X, y)

        original_size = len(X)
        X_orig_scaled = state.scaler.transform(X)
        assert np.allclose(X2[:original_size], X_orig_scaled)
        assert len(y2) == original_size * 4

    def test_transform_no_augmentation(self, sample_data_large):
        """test_transform_no_augmentation function."""
        from config import DataConfig
        X, y = sample_data_large
        X_train, X_val = X[:400], X[400:]
        y_train = y[:400]
        feature_names = DataConfig().feature_sets['Liquid']
        module = AugmentedDataModule(n_aug=5, feature_names=feature_names)
        _, _, _, state = module.fit_transform(X_train, y_train)

        X_val_scaled, _ = module.transform(X_val, state)
        assert X_val_scaled.shape == X_val.shape

    def test_missing_feature_names_raises(self, sample_data_large):
        """test_missing_feature_names_raises function."""
        X, y = sample_data_large
        module = AugmentedDataModule(n_aug=2)
        with pytest.raises(ValueError, match="feature_names must be provided explicitly"):
            module.fit_transform(X, y)

    def test_explicit_feature_names_stored(self, sample_data_large):
        """test_explicit_feature_names_stored function."""
        from config import DataConfig
        X, y = sample_data_large
        feature_names = DataConfig().feature_sets['Liquid']
        module = AugmentedDataModule(n_aug=2, feature_names=feature_names)
        _, _, _, state = module.fit_transform(X, y)

        assert state.feature_names == feature_names

    def test_fold_seed_determinism(self, sample_data_large):
        """test_fold_seed_determinism function."""
        from config import DataConfig
        X, y = sample_data_large
        feature_names = DataConfig().feature_sets['Liquid']

        module_a = AugmentedDataModule(n_aug=2, feature_names=feature_names)
        X2_a, _, _, _ = module_a.fit_transform(X, y, fold_seed=77)

        module_b = AugmentedDataModule(n_aug=2, feature_names=feature_names)
        X2_b, _, _, _ = module_b.fit_transform(X, y, fold_seed=77)

        assert np.allclose(X2_a, X2_b), "Same fold_seed must produce identical augmented data"

        module_c = AugmentedDataModule(n_aug=2, feature_names=feature_names)
        X2_c, _, _, _ = module_c.fit_transform(X, y, fold_seed=99)

        assert not np.allclose(X2_a, X2_c), "Different fold_seeds should produce different augmented data"


class TestGetDataModule:
    """TestGetDataModule class."""

    def test_get_raw(self):
        """test_get_raw function."""
        module = get_data_module('raw')
        assert isinstance(module, RawDataModule)

    def test_get_balanced(self):
        """test_get_balanced function."""
        module = get_data_module('balanced')
        assert isinstance(module, BalancedDataModule)

    def test_get_augmented(self):
        """test_get_augmented function."""
        module = get_data_module('augmented')
        assert isinstance(module, AugmentedDataModule)

    def test_case_insensitive(self):
        """test_case_insensitive function."""
        module1 = get_data_module('RAW')
        module2 = get_data_module('Raw')
        module3 = get_data_module('raw')

        assert type(module1) == type(module2) == type(module3)

    def test_invalid_name(self):
        """test_invalid_name function."""
        with pytest.raises(ValueError):
            get_data_module('invalid_module')

