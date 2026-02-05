# -*- coding: utf-8 -*-
"""
M1 数据模块 - RawDataModule, BalancedDataModule, AugmentedDataModule

约束：拟合操作仅在训练折进行，验证折仅应用变换
"""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from .interfaces import DataModule, DataModuleState


# ============================================================
# Raw 数据模块
# ============================================================

class RawDataModule(DataModule):
    """原始数据模块 - 仅标准化，作为基线"""

    def __init__(self, random_seed: int = 42, feature_names: Optional[List[str]] = None):
        """Parameters
        ----------
        random_seed : int
            随机种子（保留用于接口一致性）
        feature_names : List[str], optional
            特征列名列表（用于 MC 不确定性模块）
        """
        self.random_seed = random_seed
        self.feature_names = feature_names
    
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray
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
            feature_std=np.std(X_train, axis=0),  # 原始空间标准差（用于MC）
            feature_names=self.feature_names
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
                 random_seed: int = 42,
                 feature_names: Optional[List[str]] = None):
        """
        Parameters
        ----------
        n_bins : int
            分箱数量
        strategy : str
            分箱策略: 'quantile'（等频）| 'uniform'（等宽）| 'kmeans'
        random_seed : int
            随机种子（用于 kmeans 策略）
        feature_names : List[str], optional
            特征列名列表（用于 MC 不确定性模块）
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_seed = random_seed
        self.feature_names = feature_names
    
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """标准化训练数据并按目标分箱计算样本权重。"""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # 1. 按目标分箱，估计各箱样本数量
        kbd = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy=self.strategy,
            subsample=None,  # 禁用子采样，保证分箱稳定
            random_state=self.random_seed if self.strategy == 'kmeans' else None
        )
        bins = kbd.fit_transform(y_train.reshape(-1, 1)).flatten().astype(int)
        
        # 2. 反频率加权，避免高频区主导
        bin_counts = np.bincount(bins, minlength=self.n_bins)
        bin_weights = 1.0 / (bin_counts + 1e-8)
        raw_weights = bin_weights[bins]
        sample_weights = raw_weights / raw_weights.sum() * len(y_train)
        
        # 3. 保存状态
        state = DataModuleState(
            scaler=scaler,
            bin_edges=kbd.bin_edges_,
            feature_std=np.std(X_train, axis=0),
            feature_names=self.feature_names,
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
    数据增强模块 - 基于 EPMA 误差模型的扰动增强

    策略：
    1. 为每个样本生成 n_aug 个扰动副本
    2. EPMA 误差模型：按氧化物列名映射相对误差（主量 3%，低含量 8%）
    3. 不做负值裁剪，保留完整正态分布
    4. 与 M4 不确定性模块 / perturbation.py 使用相同的误差模型

    注意：
    - 增强仅在训练折内进行，验证折不增强
    - 增强后样本量 = (1 + n_aug) × 原样本量
    - 每次调用 fit_transform 使用不同随机种子（基于调用计数器）
    """

    def __init__(
                 self, 
                 n_aug: int = 15,
                 feature_names: Optional[List[str]] = None,
                 random_seed: int = 42):
        """
        Parameters
        ----------
        n_aug : int
            每个原始样本生成的增强副本数
        feature_names : List[str], optional
            特征列名列表；若为 None，将按特征数自动推断（9/18），否则抛出异常
        random_seed : int
            基础随机种子
        """
        self.n_aug = n_aug
        self.feature_names = feature_names
        self.random_seed = random_seed
        self._fit_count = 0  # 调用计数器，用于派生不同的随机种子

    def fit_transform(
                      self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """标准化训练数据并执行 EPMA 扰动增强。"""
        from .perturbation import get_rel_err_vector, epma_perturb

        # 为每次 fit_transform 派生不同随机种子，避免增强样本重复
        effective_seed = self.random_seed + self._fit_count
        self._fit_count += 1
        rng = np.random.RandomState(effective_seed)

        # 1. 统计原始特征标准差（用于 MC）
        feature_std = np.std(X_train, axis=0)

        # 2. 解析/校验特征名，保证误差模型精确匹配
        if self.feature_names is None:
            self.feature_names = self._infer_feature_names(X_train.shape[1])
        if len(self.feature_names) != X_train.shape[1]:
            raise ValueError("feature_names 长度必须与 X_train 的特征维度一致")
        rel_err_vec = get_rel_err_vector(self.feature_names, strict=True)

        # 3. 拟合 scaler（仅在训练折）
        scaler = StandardScaler()
        scaler.fit(X_train)

        # 4. 执行扰动增强，与 perturbation.py 误差模型保持一致
        X_list = [X_train]
        y_list = [y_train]

        for _ in range(self.n_aug):
            X_augmented = epma_perturb(X_train, rel_err_vec, rng)
            X_list.append(X_augmented)
            y_list.append(y_train)

        # 5. 拼接并标准化
        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        X_scaled = scaler.transform(X_all)

        # 6. 权重保持一致
        sample_weights = np.ones(len(y_all), dtype=np.float64)

        # 7. 保存状态
        state = DataModuleState(
            scaler=scaler,
            feature_std=feature_std,
            feature_names=self.feature_names,
            extra={
                'n_aug': self.n_aug,
                'original_size': len(y_train)
            }
        )

        return X_scaled, y_all, sample_weights, state

    def transform(
                  self,
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """变换验证数据（不进行增强）"""
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

