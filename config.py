# -*- coding: utf-8 -*-
"""
ѧϰѹ - ȫ
Config: ·ӡCVж
"""

import os

# ============================================================
# ·
# ============================================================

# ĿĿ¼ǰļĿ¼
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ļ·
DATA_PATH = os.path.join(PROJECT_ROOT, 'input.csv')

# Ŀ¼
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs')

# ģͻĿ¼
CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')


# ============================================================
# CV
# ============================================================

RANDOM_SEED = 42
N_SPLITS = 5  #  GroupKFold 
INNER_CV_SPLITS = 5  # Stacking ڲ CV 


# ============================================================
# ж
# ============================================================

# ݱ
DATA_ENCODING = 'latin-1'

# ĿУ¶Ⱥѹö·
TARGET_COLS = {
    'T': 'T',  # ¶ ()
    'P': 'P',  # ѹ (kbar)
}

# У/ʵԴ飩
GROUP_COL = 'Ref'

# CPX 12У
CPX_OXIDE_COLS = [
    'SiO2.cpx', 'Al2O3.cpx', 'TiO2.cpx', 'CaO.cpx', 'Na2O.cpx', 'K2O.cpx',
    'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'Cr2O3.cpx', 'NiO.cpx', 'P2O5.cpx'
]

# LIQ 12У
LIQ_OXIDE_COLS = [
    'SiO2.liq', 'Al2O3.liq', 'TiO2.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq',
    'FeO.liq', 'MgO.liq', 'MnO.liq', 'Cr2O3.liq', 'NiO.liq', 'P2O5.liq'
]

# CPX 12У6㣩
CPX_CATION_COLS = [
    'Si.cpx', 'Al.cpx', 'Ti.cpx', 'Ca.cpx', 'Na.cpx', 'K.cpx',
    'Fe.cpx', 'Mg.cpx', 'Mn.cpx', 'Cr.cpx', 'Ni.cpx', 'P.cpx'
]

# Ԥ
FEATURE_SETS = {
    'cpx_oxide': CPX_OXIDE_COLS,
    'liq_oxide': LIQ_OXIDE_COLS,
    'cpx_cation': CPX_CATION_COLS,
    'cpx_only': CPX_OXIDE_COLS + CPX_CATION_COLS,  # 24
    'cpx_liq': CPX_OXIDE_COLS + LIQ_OXIDE_COLS + CPX_CATION_COLS,  # 36
}

# Ĭ
DEFAULT_FEATURE_MODE = 'cpx_liq'


# ============================================================
# ģĬϲ
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
    'n_jobs': 1,
}


# ============================================================
# ʵԤ
# ============================================================

# 4ʵ
EXPERIMENT_CONFIGS = {
    'exp1_baseline': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': False,
        'correct': False,
    },
    'exp2_aug_only': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': False,
    },
    'exp3_corr_only': {  # ƫУǿ
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': False,
        'correct': True,
    },
    'exp4_aug_corr': {
        'model_type': 'catboost',
        'model_params': CATBOOST_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': True,
    },
    'exp5_stacking': {
        'model_type': 'stacking',
        'model_params': STACKING_DEFAULT_PARAMS.copy(),
        'augment': True,
        'correct': True,
    },
}


# ============================================================
# ǿ
# ============================================================

AUGMENT_CONFIG = {
    'method': 'noise',
    'n_aug': 1,  # ǿ = (1 + n_aug) * ԭʼ
    'noise_level': 0.02,  # ˮƽڱ׼
}


# ============================================================
# 
# ============================================================

def get_feature_cols(mode: str = None) -> list:
    """ȡб"""
    mode = mode or DEFAULT_FEATURE_MODE
    if mode not in FEATURE_SETS:
        raise ValueError(f"δ֪: {mode}֧ {list(FEATURE_SETS.keys())}")
    return FEATURE_SETS[mode]


def ensure_dirs():
    """ȷҪĿ¼"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)


# ============================================================
# ģʱִ
# ============================================================

# ȷĿ¼
ensure_dirs()
