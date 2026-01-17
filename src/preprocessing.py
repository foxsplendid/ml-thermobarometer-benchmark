# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 预处理模块
Preprocessing Module: 数据加载、特征选择、数据增强

核心约束：
1. 数据增强仅在训练集上执行
2. 标准化器只在训练折 fit
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler


# ============================================================
# 特征列定义
# ============================================================

# CPX 氧化物特征（12列）
CPX_OXIDE_COLS = [
    'SiO2.cpx', 'Al2O3.cpx', 'TiO2.cpx', 'CaO.cpx', 'Na2O.cpx', 'K2O.cpx',
    'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'Cr2O3.cpx', 'NiO.cpx', 'P2O5.cpx'
]

# LIQ 氧化物特征（12列）
LIQ_OXIDE_COLS = [
    'SiO2.liq', 'Al2O3.liq', 'TiO2.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq',
    'FeO.liq', 'MgO.liq', 'MnO.liq', 'Cr2O3.liq', 'NiO.liq', 'P2O5.liq'
]

# CPX 阳离子特征（12列，6氧基归算）
CPX_CATION_COLS = [
    'Si.cpx', 'Al.cpx', 'Ti.cpx', 'Ca.cpx', 'Na.cpx', 'K.cpx',
    'Fe.cpx', 'Mg.cpx', 'Mn.cpx', 'Cr.cpx', 'Ni.cpx', 'P.cpx'
]

# 目标列
TARGET_COLS = ['T', 'P']

# 分组列
GROUP_COL = 'Ref'

# 预定义特征集合
FEATURE_SETS = {
    'cpx_oxide': CPX_OXIDE_COLS,
    'liq_oxide': LIQ_OXIDE_COLS,
    'cpx_cation': CPX_CATION_COLS,
    'cpx_only': CPX_OXIDE_COLS + CPX_CATION_COLS,  # 24列
    'cpx_liq': CPX_OXIDE_COLS + LIQ_OXIDE_COLS + CPX_CATION_COLS,  # 36列
    'all_oxide': CPX_OXIDE_COLS + LIQ_OXIDE_COLS,  # 24列
}


# ============================================================
# 数据加载
# ============================================================

def load_data(filepath: str, encoding: str = 'latin-1', drop_dirty_cols: bool = True) -> pd.DataFrame:
    """
    加载数据文件
    
    Parameters
    ----------
    filepath : str
        数据文件路径
    encoding : str, default='latin-1'
        文件编码
    drop_dirty_cols : bool, default=True
        是否删除脏列（Unnamed: 0, ï..Index 等）
    
    Returns
    -------
    df : pd.DataFrame
        清洗后的数据框
    """
    df = pd.read_csv(filepath, encoding=encoding)
    
    if drop_dirty_cols:
        # 删除常见的脏列
        dirty_patterns = ['Unnamed:', 'ï..']
        cols_to_drop = [col for col in df.columns if any(p in col for p in dirty_patterns)]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
    
    # 确保数值列为 float
    numeric_cols = CPX_OXIDE_COLS + LIQ_OXIDE_COLS + CPX_CATION_COLS + TARGET_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 填充缺失值
    df = df.fillna(0)
    
    return df


def get_feature_cols(mode: str = 'cpx_liq') -> List[str]:
    """
    获取特征列名列表
    
    Parameters
    ----------
    mode : str, default='cpx_liq'
        特征集合模式：'cpx_only', 'cpx_liq', 'cpx_oxide', 'liq_oxide', 'cpx_cation', 'all_oxide'
    
    Returns
    -------
    feature_cols : List[str]
        特征列名列表
    """
    if mode not in FEATURE_SETS:
        raise ValueError(f"未知特征集合: {mode}，支持 {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[mode]


def prepare_data(df: pd.DataFrame, feature_mode: str = 'cpx_liq') -> Dict[str, np.ndarray]:
    """
    准备实验数据
    
    Parameters
    ----------
    df : pd.DataFrame
        原始数据框
    feature_mode : str, default='cpx_liq'
        特征集合模式
    
    Returns
    -------
    data : dict
        包含 X, y_T, y_P, groups, row_ids, refs 的字典
    """
    feature_cols = get_feature_cols(feature_mode)
    
    # 检查列是否存在
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失特征列: {missing_cols}")
    
    # 提取数据
    X = df[feature_cols].values.astype(np.float64)
    y_T = df['T'].values.astype(np.float64)
    y_P = df['P'].values.astype(np.float64)
    groups = df[GROUP_COL].values
    row_ids = np.arange(len(df))
    refs = df[GROUP_COL].values
    
    return {
        'X': X,
        'y_T': y_T,
        'y_P': y_P,
        'groups': groups,
        'row_ids': row_ids,
        'refs': refs,
        'feature_cols': feature_cols
    }


# ============================================================
# 数据增强
# ============================================================

def augment_noise(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  n_aug: int = 1, noise_level: float = 0.02,
                  random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    高斯噪声数据增强
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        特征矩阵
    y : np.ndarray, shape (n_samples,)
        目标值
    groups : np.ndarray, shape (n_samples,)
        分组标签
    n_aug : int, default=1
        增强倍数（最终样本数 = (1 + n_aug) * n_samples）
    noise_level : float, default=0.02
        噪声水平（相对于各特征标准差的比例）
    random_seed : int, optional
        随机种子
    
    Returns
    -------
    X_aug, y_aug, groups_aug : np.ndarray
        增强后的数据
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X_list = [X]
    y_list = [y]
    groups_list = [groups]
    
    feature_std = np.std(X, axis=0)
    
    for _ in range(n_aug):
        noise = np.random.normal(0, noise_level, X.shape) * feature_std
        X_list.append(X + noise)
        y_list.append(y)
        groups_list.append(groups)
    
    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list)
    )


def augment_jitter(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_aug: int = 1, jitter_range: float = 0.01,
                   random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    抖动数据增强（均匀分布）
    
    Parameters
    ----------
    jitter_range : float, default=0.01
        抖动范围（相对于各特征标准差的比例）
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    X_list = [X]
    y_list = [y]
    groups_list = [groups]
    
    feature_std = np.std(X, axis=0)
    
    for _ in range(n_aug):
        jitter = np.random.uniform(-jitter_range, jitter_range, X.shape) * feature_std
        X_list.append(X + jitter)
        y_list.append(y)
        groups_list.append(groups)
    
    return (
        np.vstack(X_list),
        np.concatenate(y_list),
        np.concatenate(groups_list)
    )


def augment_data(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                 method: str = 'noise', n_aug: int = 1, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    数据增强工厂函数
    
    Parameters
    ----------
    method : str, default='noise'
        增强方法：'noise', 'jitter', 'none'
    """
    method = method.lower().strip()
    
    if method == 'noise':
        return augment_noise(X, y, groups, n_aug=n_aug, **kwargs)
    elif method == 'jitter':
        return augment_jitter(X, y, groups, n_aug=n_aug, **kwargs)
    elif method == 'none':
        return X, y, groups
    else:
        raise ValueError(f"未知增强方法: {method}，支持 'noise', 'jitter', 'none'")


# ============================================================
# 标准化器封装
# ============================================================

class FoldScaler:
    """
    折叠标准化器封装
    
    记录拟合状态，确保每折使用独立的标准化器
    """
    
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted = False
    
    def fit(self, X: np.ndarray) -> 'FoldScaler':
        """在训练集上拟合"""
        self._scaler = StandardScaler()
        self._scaler.fit(X)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换数据"""
        if not self._is_fitted:
            raise RuntimeError("标准化器未拟合，请先调用 fit()")
        return self._scaler.transform(X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        self.fit(X)
        return self.transform(X)
    
    @property
    def mean_(self) -> np.ndarray:
        """获取均值"""
        if not self._is_fitted:
            raise RuntimeError("标准化器未拟合")
        return self._scaler.mean_
    
    @property
    def scale_(self) -> np.ndarray:
        """获取标准差"""
        if not self._is_fitted:
            raise RuntimeError("标准化器未拟合")
        return self._scaler.scale_


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    import os
    
    print("=== 数据预处理示例 ===")
    
    # 假设数据文件存在
    data_path = 'input.csv'
    if os.path.exists(data_path):
        # 加载数据
        df = load_data(data_path)
        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)[:10]}...")
        
        # 准备数据
        data = prepare_data(df, feature_mode='cpx_liq')
        print(f"\n特征矩阵形状: {data['X'].shape}")
        print(f"温度范围: {data['y_T'].min():.0f} - {data['y_T'].max():.0f} ℃")
        print(f"压力范围: {data['y_P'].min():.2f} - {data['y_P'].max():.2f} kbar")
        print(f"文献来源数量: {len(np.unique(data['groups']))}")
    else:
        print(f"数据文件 {data_path} 不存在")
