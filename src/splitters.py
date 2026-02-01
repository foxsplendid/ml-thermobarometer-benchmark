# -*- coding: utf-8 -*-
"""
数据集划分工具 - P-T 网格采样与分层

主要函数：
- compute_pt_edges: 计算 P-T 网格边界
- assign_pt_bins: 分配样本到 P-T 格子
- select_test_indices: 每个非空格子随机选 1 个样本作为测试集
- stratified_subsample_indices: 分层子采样（用于学习曲线实验）
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
    不再考虑Ref分组约束，优先保证P-T分布平衡
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


def stratified_subsample_indices(
    indices: np.ndarray,
    strat_labels: np.ndarray,
    fraction: float,
    seed: int = 42
) -> np.ndarray:
    """
    对给定索引进行分层子采样，保持每个分层（bin）的比例

    用于学习曲线实验：在训练集内部按比例抽取子集，同时保证P-T分布平衡

    Parameters
    ----------
    indices : np.ndarray
        原始索引数组（例如 train_full_indices）
    strat_labels : np.ndarray
        与 indices 等长的分层标签（例如 P-T bin 标签）
    fraction : float
        采样比例，范围 (0, 1]
    seed : int
        随机种子

    Returns
    -------
    subsampled_indices : np.ndarray
        采样后的索引（原始索引空间）

    Notes
    -----
    - 每个非空 bin 至少保留 1 个样本（如果 fraction > 0 且该 bin 非空）
    - 如果 fraction = 1.0，返回原始 indices
    - 尽量保持每个 bin 的比例，使用 ceil 确保小 bin 不丢失
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction 必须在 (0, 1] 范围内，当前值: {fraction}")

    if fraction == 1.0:
        return indices.copy()

    rng = np.random.RandomState(seed)

    # 构建 bin -> 局部索引（在 indices 数组中的位置）的映射
    bin_to_local_indices: Dict[int, np.ndarray] = {}
    unique_bins = np.unique(strat_labels)
    for bin_id in unique_bins:
        local_idxs = np.where(strat_labels == bin_id)[0]
        if local_idxs.size > 0:
            bin_to_local_indices[int(bin_id)] = local_idxs

    # 分层采样
    sampled_local_indices = []
    for bin_id in sorted(bin_to_local_indices.keys()):
        local_idxs = bin_to_local_indices[bin_id]
        n_bin = len(local_idxs)
        # 计算该 bin 应采样数量：ceil 保证至少 1 个（如果 bin 非空）
        n_sample = max(1, int(np.ceil(n_bin * fraction)))
        n_sample = min(n_sample, n_bin)  # 不能超过 bin 大小
        # 随机选择
        chosen = rng.choice(local_idxs, size=n_sample, replace=False)
        sampled_local_indices.extend(chosen.tolist())

    # 转换回原始索引空间
    sampled_local_indices = np.array(sampled_local_indices, dtype=int)
    return indices[sampled_local_indices]

