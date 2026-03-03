# -*- coding: utf-8 -*-
"""P-T binning and hold-out split helpers for stratified sampling."""

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
    """compute_pt_edges function."""
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
    """assign_pt_bins function."""
    if bins.n_p_bins <= 0 or bins.n_t_bins <= 0:
        raise ValueError("Invalid P-T bin edges.")

    p_bins = np.digitize(y_p, bins.p_edges[1:-1], right=False)
    t_bins = np.digitize(y_t, bins.t_edges[1:-1], right=False)

    return p_bins * bins.n_t_bins + t_bins




def select_test_indices(
    tp_bins: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """select_test_indices function."""
    rng = np.random.RandomState(random_state)

    bin_to_indices: Dict[int, np.ndarray] = {}
    for bin_id in np.unique(tp_bins):
        idxs = np.where(tp_bins == bin_id)[0]
        if idxs.size > 0:
            bin_to_indices[int(bin_id)] = idxs

    test_idx_list = []
    for bin_id in sorted(bin_to_indices.keys()):
        idxs = bin_to_indices[bin_id]
        picked = rng.choice(idxs)
        test_idx_list.append(picked)

    return np.array(test_idx_list, dtype=int)



