# -*- coding: utf-8 -*-
"""Protocol package: cross-validation pipeline and experiment matrix."""

from .pipeline import (
    P_SEED_OFFSET,
    Pipeline,
    StratifiedCVProtocol,
    apply_seed,
    call_pipeline_factory,
    derive_target_seed,
    get_effective_n_splits,
    merge_sparse_bins,
)
from .matrix import ExperimentConfig, ExperimentMatrix

__all__ = [
    "P_SEED_OFFSET",
    "Pipeline",
    "StratifiedCVProtocol",
    "apply_seed",
    "call_pipeline_factory",
    "derive_target_seed",
    "get_effective_n_splits",
    "merge_sparse_bins",
    "ExperimentConfig",
    "ExperimentMatrix",
]
