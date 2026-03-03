# -*- coding: utf-8 -*-
"""Shared pytest fixtures for reproducible synthetic benchmark data."""

import pytest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# ============================================================

@pytest.fixture
def sample_data():
    """sample_data function."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features) * 10 + 50
    X = np.clip(X, 0, 100)

    y = np.random.randn(n_samples) * 100 + 1100

    return X, y


@pytest.fixture
def sample_data_large():
    """sample_data_large function."""
    np.random.seed(42)
    n_samples = 500
    n_features = 18

    X = np.random.randn(n_samples, n_features) * 10 + 50
    X = np.clip(X, 0, 100)
    y = np.random.randn(n_samples) * 100 + 1100

    return X, y


@pytest.fixture
def train_val_split(sample_data):
    """train_val_split function."""
    X, y = sample_data
    train_idx = np.arange(80)
    val_idx = np.arange(80, 100)

    return {
        'X_train': X[train_idx],
        'X_val': X[val_idx],
        'y_train': y[train_idx],
        'y_val': y[val_idx],
    }


# ============================================================
# ============================================================

@pytest.fixture
def pt_data():
    """pt_data function."""
    np.random.seed(42)
    n_samples = 200

    y_T = np.random.uniform(800, 1400, n_samples)

    y_P = np.random.uniform(0, 15, n_samples)

    return y_T, y_P


# ============================================================
# ============================================================

@pytest.fixture
def trained_raw_module(train_val_split):
    """trained_raw_module function."""
    from src.data_modules import RawDataModule

    module = RawDataModule(random_seed=42)
    X_scaled, y, weights, state = module.fit_transform(
        train_val_split['X_train'],
        train_val_split['y_train']
    )

    return module, state, X_scaled, weights


# ============================================================
# ============================================================

@pytest.fixture
def test_config():
    """test_config function."""
    return {
        'n_splits': 3,
        'random_seed': 42,
        'n_aug': 2,
    }

