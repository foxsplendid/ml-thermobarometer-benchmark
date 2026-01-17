# -*- coding: utf-8 -*-
"""
ѧϰѹ - src ģʼ
ģĺͺ
"""

# ģģ
from .models import (
    BaseThermoModel,
    CatBoostWrapper,
    GroupAwareStacker,
    get_model,
    create_default_stacker
)

# ģ
from .runner import (
    ExperimentConfig,
    SingleTargetRunner,
    ExperimentRunner,
    run_single_experiment,
    run_experiment_matrix
)

# ƫУģ
from .correction import (
    BiasCorrector,
    LinearBiasCorrector,
    IdentityCorrector,
    PolynomialBiasCorrector,
    get_corrector
)

# Ԥģ
from .preprocessing import (
    load_data,
    get_feature_cols,
    prepare_data,
    augment_data,
    augment_noise,
    FoldScaler,
    CPX_OXIDE_COLS,
    LIQ_OXIDE_COLS,
    CPX_CATION_COLS,
    FEATURE_SETS
)

# ָģ
from .metrics import (
    rmse,
    mae,
    r2,
    mape,
    bias,
    compute_metrics,
    compute_metrics_by_target,
    summarize_folds,
    print_summary,
    compare_experiments
)

# ӻģ
from .viz import (
    plot_pred_vs_true,
    plot_residuals,
    plot_fold_comparison,
    plot_experiment_summary,
    plot_full_report,
    save_figure
)

__all__ = [
    # ģ
    'BaseThermoModel', 'CatBoostWrapper', 'GroupAwareStacker', 
    'get_model', 'create_default_stacker',
    # 
    'ExperimentConfig', 'SingleTargetRunner', 'ExperimentRunner',
    'run_single_experiment', 'run_experiment_matrix',
    # У
    'BiasCorrector', 'LinearBiasCorrector', 'IdentityCorrector', 
    'PolynomialBiasCorrector', 'get_corrector',
    # Ԥ
    'load_data', 'get_feature_cols', 'prepare_data', 'augment_data', 
    'augment_noise', 'FoldScaler',
    'CPX_OXIDE_COLS', 'LIQ_OXIDE_COLS', 'CPX_CATION_COLS', 'FEATURE_SETS',
    # ָ
    'rmse', 'mae', 'r2', 'mape', 'bias',
    'compute_metrics', 'compute_metrics_by_target', 
    'summarize_folds', 'print_summary', 'compare_experiments',
    # ӻ
    'plot_pred_vs_true', 'plot_residuals', 'plot_fold_comparison',
    'plot_experiment_summary', 'plot_full_report', 'save_figure'
]

__version__ = '0.1.0'
