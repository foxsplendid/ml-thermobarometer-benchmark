# -*- coding: utf-8 -*-
"""
数据集划分工具 - P-T 网格采样与分层

主要函数：
- compute_pt_edges: 计算 P-T 网格边界
- assign_pt_bins: 分配样本到 P-T 格子
- select_test_indices: 每个非空格子随机选 1 个样本作为测试集
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class PTBins:
    p_edges: np.ndarray
    t_edges: np.ndarray

    @property
    def n_p_bins(self) -> int:
        return max(len(self.p_edges) - 1, 0)

    @property
    def n_t_bins(self) -> int:
        return max(len(self.t_edges) - 1, 0)


def compute_pt_edges(y_t: np.ndarray, y_p: np.ndarray) -> PTBins:
    """
    计算 P-T 网格边界

    策略（遵循文献惯例）：
    - k = ceil(sqrt(n))，即网格数约等于样本数的平方根
    - P 边界四舍五入到 0.1 kbar
    - T 边界四舍五入到 1 °C
    """
    n_samples = len(y_t)
    if n_samples == 0:
        raise ValueError("Empty input for P-T binning.")

    k = int(np.ceil(np.sqrt(n_samples)))

    p_min = float(np.min(y_p)) - 0.1
    p_max = float(np.max(y_p)) + 0.1
    p_edges = np.linspace(p_min, p_max, k)
    p_edges = np.round(p_edges, 1)

    t_min = float(np.min(y_t)) - 1.0
    t_max = float(np.max(y_t)) + 1.0
    t_edges = np.linspace(t_min, t_max, k)
    t_edges = np.round(t_edges, 0)

    return PTBins(p_edges=p_edges, t_edges=t_edges)


def assign_pt_bins(y_t: np.ndarray, y_p: np.ndarray, bins: PTBins) -> np.ndarray:
    """
    将每个样本分配到 P-T 网格单元（返回整数标签）
    """
    if bins.n_p_bins <= 0 or bins.n_t_bins <= 0:
        raise ValueError("Invalid P-T bin edges.")

    p_bins = np.digitize(y_p, bins.p_edges[1:-1], right=False)
    t_bins = np.digitize(y_t, bins.t_edges[1:-1], right=False)

    return p_bins * bins.n_t_bins + t_bins




def select_test_indices(
    tp_bins: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    从每个非空P-T bin中随机选择一个样本作为测试集

    Parameters
    ----------
    tp_bins : np.ndarray
        样本的P-T bin标签
    random_state : int, optional
        随机种子

    Returns
    -------
    test_indices : np.ndarray
        测试集索引

    Notes
    -----
    """
    rng = np.random.RandomState(random_state)

    # 构建bin到样本索引的映射
    bin_to_indices: Dict[int, np.ndarray] = {}
    for bin_id in np.unique(tp_bins):
        idxs = np.where(tp_bins == bin_id)[0]
        if idxs.size > 0:
            bin_to_indices[int(bin_id)] = idxs

    # 从每个bin中随机选择一个样本
    test_idx_list = []
    for bin_id in sorted(bin_to_indices.keys()):
        idxs = bin_to_indices[bin_id]
        picked = rng.choice(idxs)
        test_idx_list.append(picked)

    return np.array(test_idx_list, dtype=int)


