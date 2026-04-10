# -*- coding: utf-8 -*-
"""Public exports for the thermobarometer benchmark package."""

from .utils import (
    get_logger,
    setup_logging,
    get_rel_err_vector,
    epma_perturb,
    perturbation_with_repeats,
    DEFAULT_OXIDE_REL_ERR,
    compute_pt_edges,
    assign_pt_bins,
    select_test_indices,
    PTBins,
    build_model_params,
    build_data_params,
)

from .interfaces import (
    DataModule,
    ModelModule,
    CorrectionModule,
    UncertaintyModule,
    DataModuleState,
)

from .data_modules import (
    RawDataModule,
    BalancedDataModule,
    AugmentedDataModule,
    get_data_module,
)

from .model_modules import (
    ExtraTreesModel,
    CatBoostModel,
    RandomForestModel,
    StrictOOFStacking,
    RidgeModel,
    get_model_module,
)

from .correction_modules import (
    NoCorrection,
    SegmentedLinearCorrector,
    get_correction_module,
)

from .uncertainty_modules import (
    MCUncertaintyEstimator,
    get_uncertainty_module,
)

from .protocol import (
    Pipeline,
    StratifiedCVProtocol,
    ExperimentConfig,
    ExperimentMatrix,
)

from .runtime import get_n_jobs, get_fold_workers, get_fold_backend, log_runtime_info

from .metrics import (
    rmse,
    mae,
    r2,
    compute_all_metrics,
    compute_slope_intercept,
    compute_bias_stats,
    summarize_folds,
)

__version__ = '1.0.0'
__author__ = 'ML Thermobarometer Benchmark Protocol'
