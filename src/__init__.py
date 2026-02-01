# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - 模块导出

机器学习温压计模块化验证框架 v2.0
"""

# ============================================================
# 日志模块
# ============================================================
from .logger import get_logger, setup_logging

# ============================================================
# 接口定义
# ============================================================
from .interfaces import (
    DataModule,
    ModelModule,
    CorrectionModule,
    UncertaintyModule,
    DataModuleState,
)

# ============================================================
# M1 数据模块
# ============================================================
from .data_modules import (
    RawDataModule,
    BalancedDataModule,
    AugmentedDataModule,
    get_data_module,
)

# ============================================================
# M2 模型模块
# ============================================================
from .model_modules import (
    ExtraTreesModel,
    CatBoostModel,
    RandomForestModel,
    StrictOOFStacking,
    RidgeModel,
    get_model_module,
)

# ============================================================
# M3 校正模块
# ============================================================
from .correction_modules import (
    NoCorrection,
    ResidualRegressionCorrector,
    SegmentedLinearCorrector,
    get_correction_module,
)

# ============================================================
# M4 不确定性模块
# ============================================================
from .uncertainty_modules import (
    MCUncertaintyEstimator,
    get_uncertainty_module,
)

# ============================================================
# Protocol
# ============================================================
from .protocol import (
    Pipeline,
    StratifiedCVProtocol,
    RandomSplitProtocol,
    ExperimentConfig,
    ExperimentMatrix,
    compute_all_metrics,
)

# ============================================================
# 指标与可视化
# ============================================================
from .metrics import (
    rmse,
    mae,
    r2,
    compute_metrics,
    compute_slope_intercept,
    compute_bias_stats,
    summarize_folds,  # 直接从 metrics 导出，避免间接导出链
)

# ============================================================
# 版本信息
# ============================================================
__version__ = '2.0.0'
__author__ = 'ML Thermobarometer Benchmark Protocol'
