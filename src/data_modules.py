# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - M1 数据模块实现
Data Modules: RawDataModule, BalancedDataModule, AugmentedDataModule

核心约束：
1. 所有拟合操作（scaler, binning）只在训练折进行
2. 验证折仅应用变换，禁止重新拟合
3. 样本权重只用于训练，验证时返回 None
"""

import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from .interfaces import DataModule, DataModuleState


# ============================================================
# Raw 数据模块（仅标准化）
# ============================================================

class RawDataModule(DataModule):
    """
    原始数据模块 - 仅进行标准化
    
    这是最基础的数据处理，作为其他模块的对照基线
    """
    
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
    增强数据模块 - 基于蒙特卡洛的数据增强
    
    策略：
    1. 对每个原始样本生成 n_aug 个扰动副本
    2. 扰动方式：在原始空间添加高斯噪声（相对于特征标准差）
    3. 扰动后的样本与原始样本一起用于训练
    
    注意：
    - 增强只在训练折进行，验证折不增强
    - 增强后样本数 = (1 + n_aug) × 原始样本数
    - 增强样本与原始样本使用相同的目标值
    """
    
    def __init__(self, 
                 n_aug: int = 5, 
                 noise_level: float = 0.02,
                 random_seed: int = 42):
        """
        Parameters
        ----------
        n_aug : int
            每个样本生成的增强副本数
        noise_level : float
            噪声水平（相对于特征标准差的比例）
        """
        self.n_aug = n_aug
        self.noise_level = noise_level
        self.random_seed = random_seed
    
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      groups_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """标准化并生成增强样本"""
        np.random.seed(self.random_seed)
        
        # 1. 计算原始空间的特征标准差（用于生成噪声）
        feature_std = np.std(X_train, axis=0)
        
        # 2. 标准化器（在原始数据上拟合）
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # 3. 生成增强样本
        X_list = [X_train]
        y_list = [y_train]
        
        for i in range(self.n_aug):
            # 在原始空间添加噪声
            noise = np.random.normal(0, self.noise_level, X_train.shape) * feature_std
            X_augmented = X_train + noise
            X_list.append(X_augmented)
            y_list.append(y_train)  # 目标值不变
        
        # 4. 合并并标准化
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        X_scaled = scaler.transform(X_all)
        
        # 5. 所有样本权重相等
        sample_weights = np.ones(len(y_all), dtype=np.float64)
        
        # 6. 保存状态
        state = DataModuleState(
            scaler=scaler,
            feature_std=feature_std,
            extra={
                'n_aug': self.n_aug,
                'noise_level': self.noise_level,
                'original_size': len(y_train)
            }
        )
        
        return X_scaled, y_all, sample_weights, state
    
    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """变换验证数据（不增强，仅标准化）"""
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
