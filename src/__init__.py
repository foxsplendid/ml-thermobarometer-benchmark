# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - src 模块初始化
导出所有子模块的核心类和函数
"""

# 模型模块
from .models import (
    BaseThermoModel,
    CatBoostWrapper,
    GroupAwareStacker,
    get_model,
    create_default_stacker
)

# 运行器模块
from .runner import (
    ExperimentConfig,
    SingleTargetRunner,
    ExperimentRunner,
    run_single_experiment,
    run_experiment_matrix
)

# 偏差校正模块
from .correction import (
    BiasCorrector,
    LinearBiasCorrector,
    IdentityCorrector,
    PolynomialBiasCorrector,
    get_corrector
)

# 预处理模块
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

# 指标模块
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

# 可视化模块
from .viz import (
    plot_pred_vs_true,
    plot_residuals,
    plot_fold_comparison,
    plot_experiment_summary,
    plot_full_report,
    save_figure
)

__all__ = [
    # 模型
    'BaseThermoModel', 'CatBoostWrapper', 'GroupAwareStacker', 
    'get_model', 'create_default_stacker',
    # 运行器
    'ExperimentConfig', 'SingleTargetRunner', 'ExperimentRunner',
    'run_single_experiment', 'run_experiment_matrix',
    # 校正
    'BiasCorrector', 'LinearBiasCorrector', 'IdentityCorrector', 
    'PolynomialBiasCorrector', 'get_corrector',
    # 预处理
    'load_data', 'get_feature_cols', 'prepare_data', 'augment_data', 
    'augment_noise', 'FoldScaler',
    'CPX_OXIDE_COLS', 'LIQ_OXIDE_COLS', 'CPX_CATION_COLS', 'FEATURE_SETS',
    # 指标
    'rmse', 'mae', 'r2', 'mape', 'bias',
    'compute_metrics', 'compute_metrics_by_target', 
    'summarize_folds', 'print_summary', 'compare_experiments',
    # 可视化
    'plot_pred_vs_true', 'plot_residuals', 'plot_fold_comparison',
    'plot_experiment_summary', 'plot_full_report', 'save_figure'
]

__version__ = '0.1.0'
