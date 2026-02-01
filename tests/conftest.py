# -*- coding: utf-8 -*-
"""
pytest 共享 fixtures

提供测试数据和通用配置，避免各测试文件重复生成
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 基础测试数据
# ============================================================

@pytest.fixture
def sample_data():
    """
    生成标准测试数据

    Returns
    -------
    tuple
        (X, y, groups) - 特征矩阵、目标值、分组标签
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # 模拟 wt% 氧化物数据（范围 0-60%）
    X = np.random.randn(n_samples, n_features) * 10 + 50
    X = np.clip(X, 0, 100)  # 确保非负

    # 模拟温度目标（800-1400°C）
    y = np.random.randn(n_samples) * 100 + 1100

    # 模拟文献分组
    groups = np.random.choice(['RefA', 'RefB', 'RefC', 'RefD'], n_samples)

    return X, y, groups


@pytest.fixture
def sample_data_large():
    """
    生成较大规模测试数据（用于性能测试）

    Returns
    -------
    tuple
        (X, y, groups)
    """
    np.random.seed(42)
    n_samples = 500
    n_features = 18  # Liquid 特征集

    X = np.random.randn(n_samples, n_features) * 10 + 50
    X = np.clip(X, 0, 100)
    y = np.random.randn(n_samples) * 100 + 1100
    groups = np.random.choice([f'Ref{i}' for i in range(20)], n_samples)

    return X, y, groups


@pytest.fixture
def train_val_split(sample_data):
    """
    训练/验证划分（80/20）

    Returns
    -------
    dict
        包含 X_train, X_val, y_train, y_val, groups_train
    """
    X, y, groups = sample_data
    train_idx = np.arange(80)
    val_idx = np.arange(80, 100)

    return {
        'X_train': X[train_idx],
        'X_val': X[val_idx],
        'y_train': y[train_idx],
        'y_val': y[val_idx],
        'groups_train': groups[train_idx],
        'groups_val': groups[val_idx],
    }


# ============================================================
# P-T 相关测试数据
# ============================================================

@pytest.fixture
def pt_data():
    """
    生成 P-T 测试数据

    Returns
    -------
    tuple
        (y_T, y_P) - 温度和压力数组
    """
    np.random.seed(42)
    n_samples = 200

    # 温度：800-1400°C
    y_T = np.random.uniform(800, 1400, n_samples)

    # 压力：0-15 kbar
    y_P = np.random.uniform(0, 15, n_samples)

    return y_T, y_P


# ============================================================
# 模型相关 fixtures
# ============================================================

@pytest.fixture
def trained_raw_module(train_val_split):
    """
    已拟合的 RawDataModule

    Returns
    -------
    tuple
        (module, state, X_scaled, weights)
    """
    from src.data_modules import RawDataModule

    module = RawDataModule(random_seed=42)
    X_scaled, y, weights, state = module.fit_transform(
        train_val_split['X_train'],
        train_val_split['y_train'],
        train_val_split['groups_train']
    )

    return module, state, X_scaled, weights


# ============================================================
# 配置相关 fixtures
# ============================================================

@pytest.fixture
def test_config():
    """
    测试用配置

    Returns
    -------
    dict
        简化的测试配置
    """
    return {
        'n_splits': 3,  # 测试用少折数
        'random_seed': 42,
        'n_aug': 2,  # 测试用少增强
    }
