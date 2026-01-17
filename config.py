# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 全局配置
Config: 路径、随机种子、CV参数、数据列定义
"""

import os

# ============================================================
# 路径配置
# ============================================================

# 项目根目录（当前文件所在目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据文件路径
DATA_PATH = os.path.join(PROJECT_ROOT, 'input.csv')

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# 模型缓存目录
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')


# ============================================================
# 随机种子与CV配置
# ============================================================

RANDOM_SEED = 42
N_SPLITS = 5  # 外层 GroupKFold 折数
INNER_CV_SPLITS = 5  # Stacking 内层 CV 折数


# ============================================================
# 数据列定义
# ============================================================

# 数据编码
DATA_ENCODING = 'latin-1'

# 目标列（温度和压力采用独立链路）
TARGET_COLS = {
    'T': 'T',  # 温度 (℃)
    'P': 'P',  # 压力 (kbar)
}

# 分组列（按文献/实验来源分组）
GROUP_COL = 'Ref'

# CPX 氧化物特征（12列）
CPX_OXIDE_COLS = [
    'SiO2.cpx', 'Al2O3.cpx', 'TiO2.cpx', 'CaO.cpx', 'Na2O.cpx', 'K2O.cpx',
    'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'Cr2O3.cpx', 'NiO.cpx', 'P2O5.cpx'
]

# LIQ 氧化物特征（12列）
LIQ_OXIDE_COLS = [
    'SiO2.liq', 'Al2O3.liq', 'TiO2.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq',
    'FeO.liq', 'MgO.liq', 'MnO.liq', 'Cr2O3.liq', 'NiO.liq', 'P2O5.liq'
]

# CPX 阳离子特征（12列，6氧基归算）
CPX_CATION_COLS = [
    'Si.cpx', 'Al.cpx', 'Ti.cpx', 'Ca.cpx', 'Na.cpx', 'K.cpx',
    'Fe.cpx', 'Mg.cpx', 'Mn.cpx', 'Cr.cpx', 'Ni.cpx', 'P.cpx'
]

# 预定义特征集合
FEATURE_SETS = {
    'cpx_oxide': CPX_OXIDE_COLS,
    'liq_oxide': LIQ_OXIDE_COLS,
    'cpx_cation': CPX_CATION_COLS,
    'cpx_only': CPX_OXIDE_COLS + CPX_CATION_COLS,  # 24列
    'cpx_liq': CPX_OXIDE_COLS + LIQ_OXIDE_COLS + CPX_CATION_COLS,  # 36列
}

# 默认特征集合
DEFAULT_FEATURE_MODE = 'cpx_liq'


# ============================================================
# 模型默认参数
# ============================================================

CATBOOST_DEFAULT_PARAMS = {
    'iterations': 1000,
    'depth': 6,
    'learning_rate': 0.03,
    'loss_function': 'RMSE',
    'random_seed': RANDOM_SEED,
    'silent': True,
}

STACKING_DEFAULT_PARAMS = {
    'inner_cv': INNER_CV_SPLITS,
    'use_scaler': True,
    'random_seed': RANDOM_SEED,
}


# ============================================================
# 实验矩阵预定义配置
# ============================================================

# 4组实验配置
EXPERIMENT_CONFIGS = {
    'exp1_catboost_base': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': False,
        'correct': False,
    },
    'exp2_catboost_aug': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': False,
    },
    'exp3_catboost_aug_corr': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': True,
    },
    'exp4_stacking_aug_corr': {
        'model_type': 'stacking',
        'model_params': STACKING_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': True,
    },
}


# ============================================================
# 数据增强配置
# ============================================================

AUGMENT_CONFIG = {
    'method': 'noise',
    'n_aug': 1,  # 增强倍数（最终样本数 = (1 + n_aug) * 原始样本数）
    'noise_level': 0.02,  # 噪声水平（相对于标准差）
}


# ============================================================
# 辅助函数
# ============================================================

def get_feature_cols(mode: str = None) -> list:
    """获取特征列名列表"""
    mode = mode or DEFAULT_FEATURE_MODE
    if mode not in FEATURE_SETS:
        raise ValueError(f"未知特征集合: {mode}，支持 {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[mode]


def ensure_dirs():
    """确保必要目录存在"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# 模块加载时执行
# ============================================================

# 确保输出目录存在
ensure_dirs()
