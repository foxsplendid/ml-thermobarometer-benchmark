# -*- coding: utf-8 -*-
"""
共用扰动模块 - EPMA 误差模型扰动函数

设计依据：Ágreda-López et al. (2024) ML_PT_Pyworkflow

功能：
    1. 按氧化物列名映射相对误差（而非按数值阈值）
    2. 提供训练数据增强和误差传播的统一扰动接口

使用方法："""

import numpy as np
from typing import List, Optional, Tuple

# 默认的按氧化物列名映射的相对误差
# 来源：Ágreda-López et al. (2024) - 主量 3%, 低含量 8%
DEFAULT_OXIDE_REL_ERR = {
    # CPX 氧化物（9列）
    'SiO2.cpx': 0.03,   # 主量
    'TiO2.cpx': 0.08,   # 低含量
    'Al2O3.cpx': 0.03,  # 主量
    'Cr2O3.cpx': 0.08,  # 低含量
    'FeO.cpx': 0.03,    # 主量
    'MgO.cpx': 0.03,    # 主量
    'MnO.cpx': 0.08,    # 低含量
    'CaO.cpx': 0.03,    # 主量
    'Na2O.cpx': 0.08,   # 低含量
    # LIQ 氧化物（9列）
    'SiO2.liq': 0.03,   # 主量
    'TiO2.liq': 0.08,   # 低含量
    'Al2O3.liq': 0.03,  # 主量
    'FeO.liq': 0.03,    # 主量
    'MgO.liq': 0.03,    # 主量
    'MnO.liq': 0.08,    # 低含量
    'CaO.liq': 0.03,    # 主量
    'Na2O.liq': 0.08,   # 低含量
    'K2O.liq': 0.08,    # 低含量
}

def get_rel_err_vector(
    feature_names: List[str],
    oxide_rel_err: Optional[dict] = None,
    default_rel_err: Optional[float] = None,
    strict: bool = True,
) -> np.ndarray:
    """
    根据特征名生成相对误差向量。

    Parameters
    ----------
    feature_names : List[str]
        特征列名列表（与 X 的列顺序一致）
    oxide_rel_err : dict, optional
        列名到相对误差的映射，默认使用 DEFAULT_OXIDE_REL_ERR
    default_rel_err : float, optional
        显式指定未知列名的兜底值（仅 strict=False 时生效）
    strict : bool
        是否严格校验列名；True 时发现未知列名直接报错

    Returns
    -------
    np.ndarray
        相对误差向量，shape = (n_features,)

    Example
    -------
    >>> feature_names = ['SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx']
    >>> rel_err = get_rel_err_vector(feature_names)
    >>> print(rel_err)  # [0.03, 0.08, 0.03]
    """
    if oxide_rel_err is None:
        oxide_rel_err = DEFAULT_OXIDE_REL_ERR

    missing = [name for name in feature_names if name not in oxide_rel_err]
    if missing and (strict or default_rel_err is None):
        raise ValueError(f"未找到以下特征的 EPMA 误差映射: {missing}")

    return np.array([
        oxide_rel_err.get(name, default_rel_err)
        for name in feature_names
    ])


def epma_perturb(
    X: np.ndarray,
    rel_err_vec: np.ndarray,
    rng: np.random.RandomState
) -> np.ndarray:
    """
    EPMA 误差模型扰动 - 核心函数

    对输入 X 的每列按对应的相对误差添加高斯噪声：
        X_perturbed = X + Normal(0, rel_err * |X|)

    Parameters
    ----------
    X : np.ndarray
        输入数据，shape = (n_samples, n_features)
    rel_err_vec : np.ndarray
        每列的相对误差，shape = (n_features,)
    rng : np.random.RandomState
        随机数生成器

    Returns
    -------
    np.ndarray
        扰动后的数据，shape = (n_samples, n_features)

    Notes
    -----
    - 不做负值截断（clip），保留完整正态分布
    - 不做闭合约束（成分总和归一化）
    - 与 Ágreda-López et al. (2024) 的 perturbation() 函数保持一致
    """
    # 计算每个元素的标准差：std = rel_err * |value|
    # rel_err_vec 广播到 X 的每一行
    scale = rel_err_vec * np.abs(X)

    # 生成高斯噪声
    noise = rng.normal(0.0, scale, size=X.shape)

    # 返回扰动后的数据（不做 clip）
    return X + noise


def perturbation_with_repeats(
    X: np.ndarray,
    y: np.ndarray,
    rel_err_vec: np.ndarray,
    n_perturbations: int = 15,
    rng: Optional[np.random.RandomState] = None,
    random_seed: int = 42,
    include_original: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    数据增强扰动 - 生成扰动副本
    仅返回增强后的 X/y，不再返回分组标记
    """
    if rng is None:
        rng = np.random.RandomState(random_seed)

    n_original = len(X)

    if include_original:
        # 保留原始样本，避免引入额外偏差
        X_list = [X]
        y_list = [y]

        for _ in range(n_perturbations):
            X_perturbed = epma_perturb(X, rel_err_vec, rng)
            X_list.append(X_perturbed)
            y_list.append(y)

        X_aug = np.vstack(X_list)
        y_aug = np.concatenate(y_list)
    else:
        X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
        y_aug = np.repeat(y, repeats=n_perturbations, axis=0)

        scale = rel_err_vec * np.abs(X_rep)
        X_aug = rng.normal(X_rep, scale)

    return X_aug, y_aug

