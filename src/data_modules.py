# -*- coding: utf-8 -*-
"""
M1 数据模块 - RawDataModule, BalancedDataModule, AugmentedDataModule

约束：拟合操作仅在训练折进行，验证折仅应用变换
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from .interfaces import DataModule, DataModuleState


# ============================================================
# Raw 数据模块
# ============================================================

class RawDataModule(DataModule):
    """原始数据模块 - 仅标准化，作为基线"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
    
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      groups_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """标准化训练数据"""
        # 拟合标准化器
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # 所有样本权重相等
        sample_weights = np.ones(len(y_train), dtype=np.float64)
        
        # 保存状态
        state = DataModuleState(
            scaler=scaler,
            feature_std=np.std(X_train, axis=0)  # 原始空间标准差（用于MC）
        )
        
        return X_scaled, y_train.copy(), sample_weights, state
    
    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """变换验证数据（使用训练集的scaler）"""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None


# ============================================================
# Balanced 数据模块（分箱重加权）
# ============================================================

class BalancedDataModule(DataModule):
    """
    平衡数据模块 - 分箱重加权
    
    策略：
    1. 对目标变量进行分箱（等频或等宽）
    2. 计算每个箱的样本数
    3. 样本权重 = 1 / bin_count（逆频率加权）
    4. 归一化权重使总和 = 样本数
    
    这样可以让模型更关注稀疏区域的样本
    """
    
    def __init__(self, 
                 n_bins: int = 10, 
                 strategy: str = 'quantile',
                 random_seed: int = 42):
        """
        Parameters
        ----------
        n_bins : int
            分箱数量
        strategy : str
            分箱策略: 'quantile'（等频）| 'uniform'（等宽）| 'kmeans'
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_seed = random_seed
    
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      groups_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """标准化并计算样本权重"""
        # 1. 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # 2. 分箱
        kbd = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy=self.strategy,
            subsample=None  # 使用全部数据
        )
        bins = kbd.fit_transform(y_train.reshape(-1, 1)).flatten().astype(int)
        
        # 3. 计算逆频率权重
        bin_counts = np.bincount(bins, minlength=self.n_bins)
        # 避免除零（对于空箱）
        bin_weights = 1.0 / (bin_counts + 1e-8)
        # 归一化：使权重总和 = 样本数
        raw_weights = bin_weights[bins]
        sample_weights = raw_weights / raw_weights.sum() * len(y_train)
        
        # 4. 保存状态
        state = DataModuleState(
            scaler=scaler,
            bin_edges=kbd.bin_edges_,
            feature_std=np.std(X_train, axis=0),
            extra={'kbd': kbd, 'bin_counts': bin_counts}
        )
        
        return X_scaled, y_train.copy(), sample_weights, state
    
    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """变换验证数据"""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None


# ============================================================
# Augmented 数据模块（MC-based 增强）
# ============================================================

class AugmentedDataModule(DataModule):
    """
    Augmented data module with EPMA-style perturbations.

    Strategy:
    1. Generate n_aug perturbations per sample.
    2. EPMA relative error model (>1 wt%: 3%, <=1 wt%: 8%).
    3. Use the same targets for perturbed samples.

    Notes:
    - Augmentation only happens within training folds.
    - Augmented size = (1 + n_aug) * original size.
    """

    def __init__(
                 self, 
                 n_aug: int = 15,
                 noise_level: float = 0.02,
                 random_seed: int = 42,
                 error_model: str = 'epma',
                 rel_err_high: float = 0.03,
                 rel_err_low: float = 0.08,
                 error_threshold: float = 1.0,
                 clip_min: Optional[float] = 0.0):
        """
        Parameters
        ----------
        n_aug : int
            Number of augmented samples per original sample.
        noise_level : float
            Gaussian noise level (fallback when error_model != "epma").
        error_model : str
            "epma" for EPMA-style relative error, otherwise Gaussian.
        rel_err_high : float
            Relative error for values > error_threshold.
        rel_err_low : float
            Relative error for values <= error_threshold.
        error_threshold : float
            Threshold in wt% to switch relative error.
        clip_min : float or None
            Minimum value after perturbation; None disables clipping.
        """
        self.n_aug = n_aug
        self.noise_level = noise_level
        self.random_seed = random_seed
        self.error_model = error_model.lower().strip() if isinstance(error_model, str) else "epma"
        self.rel_err_high = rel_err_high
        self.rel_err_low = rel_err_low
        self.error_threshold = error_threshold
        self.clip_min = clip_min

    def _epma_perturb(self, X_raw: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        rel_err = np.where(
            np.abs(X_raw) > self.error_threshold,
            self.rel_err_high,
            self.rel_err_low
        )
        scale = rel_err * np.abs(X_raw)
        noise = rng.normal(0.0, scale, size=X_raw.shape)
        X_augmented = X_raw + noise
        if self.clip_min is not None:
            X_augmented = np.maximum(X_augmented, self.clip_min)
        return X_augmented

    def _gaussian_perturb(self, X_raw: np.ndarray, feature_std: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        noise = rng.normal(0.0, self.noise_level, X_raw.shape) * feature_std
        X_augmented = X_raw + noise
        if self.clip_min is not None:
            X_augmented = np.maximum(X_augmented, self.clip_min)
        return X_augmented

    def fit_transform(
                      self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      groups_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """Standardize and generate augmented samples."""
        # 使用隔离的 RandomState，避免污染全局随机状态
        rng = np.random.RandomState(self.random_seed)

        # 1. Feature std for Gaussian fallback
        feature_std = np.std(X_train, axis=0)

        # 2. Fit scaler on raw data
        scaler = StandardScaler()
        scaler.fit(X_train)

        # 3. Generate augmented samples
        X_list = [X_train]
        y_list = [y_train]

        for _ in range(self.n_aug):
            if self.error_model == "epma":
                X_augmented = self._epma_perturb(X_train, rng)
            else:
                X_augmented = self._gaussian_perturb(X_train, feature_std, rng)
            X_list.append(X_augmented)
            y_list.append(y_train)

        # 4. Merge and scale
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        X_scaled = scaler.transform(X_all)

        # 5. Uniform sample weights
        sample_weights = np.ones(len(y_all), dtype=np.float64)

        # 6. Save state
        state = DataModuleState(
            scaler=scaler,
            feature_std=feature_std,
            extra={
                'n_aug': self.n_aug,
                'noise_level': self.noise_level,
                'error_model': self.error_model,
                'rel_err_high': self.rel_err_high,
                'rel_err_low': self.rel_err_low,
                'error_threshold': self.error_threshold,
                'clip_min': self.clip_min,
                'original_size': len(y_train)
            }
        )

        return X_scaled, y_all, sample_weights, state

    def transform(
                  self,
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform validation data (no augmentation)."""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None

# ============================================================
# 便捷工厂函数
# ============================================================

def get_data_module(name: str, **kwargs) -> DataModule:
    """
    数据模块工厂函数
    
    Parameters
    ----------
    name : str
        模块名称: 'raw' | 'balanced' | 'augmented'
    **kwargs
        模块参数
        
    Returns
    -------
    DataModule
        数据模块实例
    """
    modules = {
        'raw': RawDataModule,
        'balanced': BalancedDataModule,
        'augmented': AugmentedDataModule,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"未知数据模块: {name}，支持 {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    print("=== 数据模块测试 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features) * 10 + 50
    y = np.random.randn(n_samples) * 100 + 1000
    groups = np.random.choice(['A', 'B', 'C'], n_samples)
    
    # 划分训练/验证
    train_idx = np.arange(80)
    val_idx = np.arange(80, 100)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train = groups[train_idx]
    
    # 测试各模块
    for name in ['raw', 'balanced', 'augmented']:
        print(f"--- {name.upper()} ---")
        module = get_data_module(name)
        
        X2, y2, weights, state = module.fit_transform(X_train, y_train, groups_train)
        print(f"训练后: X2.shape={X2.shape}, weights.sum()={weights.sum():.2f}")
        
        X_val2, _ = module.transform(X_val, state)
        print(f"验证后: X_val2.shape={X_val2.shape}")
        print()
    
    print("✅ 所有数据模块测试通过！")
