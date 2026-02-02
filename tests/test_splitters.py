# -*- coding: utf-8 -*-
"""
划分工具测试

测试 compute_pt_edges, assign_pt_bins, select_test_indices
"""

import pytest
import numpy as np
from src.splitters import (
    compute_pt_edges,
    assign_pt_bins,
    select_test_indices,
    PTBins
)


class TestComputePTEdges:
    """compute_pt_edges 测试"""

    def test_basic_edges(self, pt_data):
        """基本边界计算"""
        y_T, y_P = pt_data

        bins = compute_pt_edges(y_T, y_P)

        assert isinstance(bins, PTBins)
        assert len(bins.p_edges) > 0
        assert len(bins.t_edges) > 0

    def test_edges_cover_data(self, pt_data):
        """边界应覆盖所有数据"""
        y_T, y_P = pt_data

        bins = compute_pt_edges(y_T, y_P)

        # P 边界覆盖
        assert bins.p_edges[0] <= y_P.min()
        assert bins.p_edges[-1] >= y_P.max()

        # T 边界覆盖
        assert bins.t_edges[0] <= y_T.min()
        assert bins.t_edges[-1] >= y_T.max()

    def test_edge_count_formula(self, pt_data):
        """边界数量符合公式 k = ceil(sqrt(n))"""
        y_T, y_P = pt_data
        n_samples = len(y_T)
        expected_k = int(np.ceil(np.sqrt(n_samples)))

        bins = compute_pt_edges(y_T, y_P)

        assert len(bins.p_edges) == expected_k
        assert len(bins.t_edges) == expected_k

    def test_empty_input_raises(self):
        """空输入应抛出异常"""
        with pytest.raises(ValueError):
            compute_pt_edges(np.array([]), np.array([]))


class TestAssignPTBins:
    """assign_pt_bins 测试"""

    def test_basic_assignment(self, pt_data):
        """基本分箱分配"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins = assign_pt_bins(y_T, y_P, bins)

        assert len(tp_bins) == len(y_T)
        assert np.issubdtype(tp_bins.dtype, np.integer)

    def test_bin_labels_non_negative(self, pt_data):
        """bin 标签应非负"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins = assign_pt_bins(y_T, y_P, bins)

        assert np.all(tp_bins >= 0)

    def test_deterministic(self, pt_data):
        """结果应确定性"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins1 = assign_pt_bins(y_T, y_P, bins)
        tp_bins2 = assign_pt_bins(y_T, y_P, bins)

        assert np.array_equal(tp_bins1, tp_bins2)


class TestSelectTestIndices:
    """select_test_indices 测试"""

    def test_basic_selection(self, pt_data):
        """基本测试集选择"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx = select_test_indices(tp_bins, random_state=42)

        assert len(test_idx) > 0
        assert len(test_idx) <= len(np.unique(tp_bins))  # 每个 bin 最多 1 个

    def test_unique_indices(self, pt_data):
        """测试索引应唯一"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx = select_test_indices(tp_bins, random_state=42)

        assert len(test_idx) == len(np.unique(test_idx))

    def test_reproducible_with_seed(self, pt_data):
        """相同种子应产生相同结果"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx1 = select_test_indices(tp_bins, random_state=42)
        test_idx2 = select_test_indices(tp_bins, random_state=42)

        assert np.array_equal(test_idx1, test_idx2)

    def test_different_seeds_differ(self, pt_data):
        """不同种子应产生不同结果"""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx1 = select_test_indices(tp_bins, random_state=42)
        test_idx2 = select_test_indices(tp_bins, random_state=123)

        # 不完全相同（虽然可能有重叠）
        assert not np.array_equal(test_idx1, test_idx2)


