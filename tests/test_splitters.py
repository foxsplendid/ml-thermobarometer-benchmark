# -*- coding: utf-8 -*-
"""Unit tests for P-T splitter utilities."""

import pytest
import numpy as np
from src.utils import (
    compute_pt_edges,
    assign_pt_bins,
    select_test_indices,
    PTBins
)


class TestComputePTEdges:
    """TestComputePTEdges class."""

    def test_basic_edges(self, pt_data):
        """test_basic_edges function."""
        y_T, y_P = pt_data

        bins = compute_pt_edges(y_T, y_P)

        assert isinstance(bins, PTBins)
        assert len(bins.p_edges) > 0
        assert len(bins.t_edges) > 0

    def test_edges_cover_data(self, pt_data):
        """test_edges_cover_data function."""
        y_T, y_P = pt_data

        bins = compute_pt_edges(y_T, y_P)

        assert bins.p_edges[0] <= y_P.min()
        assert bins.p_edges[-1] >= y_P.max()

        assert bins.t_edges[0] <= y_T.min()
        assert bins.t_edges[-1] >= y_T.max()

    def test_edge_count_formula(self, pt_data):
        """test_edge_count_formula function."""
        y_T, y_P = pt_data
        n_samples = len(y_T)
        expected_k = int(np.ceil(np.sqrt(n_samples)))

        bins = compute_pt_edges(y_T, y_P)

        assert len(bins.p_edges) == expected_k
        assert len(bins.t_edges) == expected_k

    def test_empty_input_raises(self):
        """test_empty_input_raises function."""
        with pytest.raises(ValueError):
            compute_pt_edges(np.array([]), np.array([]))


class TestAssignPTBins:
    """TestAssignPTBins class."""

    def test_basic_assignment(self, pt_data):
        """test_basic_assignment function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins = assign_pt_bins(y_T, y_P, bins)

        assert len(tp_bins) == len(y_T)
        assert np.issubdtype(tp_bins.dtype, np.integer)

    def test_bin_labels_non_negative(self, pt_data):
        """test_bin_labels_non_negative function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins = assign_pt_bins(y_T, y_P, bins)

        assert np.all(tp_bins >= 0)

    def test_deterministic(self, pt_data):
        """test_deterministic function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)

        tp_bins1 = assign_pt_bins(y_T, y_P, bins)
        tp_bins2 = assign_pt_bins(y_T, y_P, bins)

        assert np.array_equal(tp_bins1, tp_bins2)


class TestSelectTestIndices:
    """TestSelectTestIndices class."""

    def test_basic_selection(self, pt_data):
        """test_basic_selection function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx = select_test_indices(tp_bins, random_state=42)

        assert len(test_idx) > 0
        assert len(test_idx) <= len(np.unique(tp_bins))

    def test_unique_indices(self, pt_data):
        """test_unique_indices function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx = select_test_indices(tp_bins, random_state=42)

        assert len(test_idx) == len(np.unique(test_idx))

    def test_reproducible_with_seed(self, pt_data):
        """test_reproducible_with_seed function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx1 = select_test_indices(tp_bins, random_state=42)
        test_idx2 = select_test_indices(tp_bins, random_state=42)

        assert np.array_equal(test_idx1, test_idx2)

    def test_different_seeds_differ(self, pt_data):
        """test_different_seeds_differ function."""
        y_T, y_P = pt_data
        bins = compute_pt_edges(y_T, y_P)
        tp_bins = assign_pt_bins(y_T, y_P, bins)

        test_idx1 = select_test_indices(tp_bins, random_state=42)
        test_idx2 = select_test_indices(tp_bins, random_state=123)

        assert not np.array_equal(test_idx1, test_idx2)



