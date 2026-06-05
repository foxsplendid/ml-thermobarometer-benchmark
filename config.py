# -*- coding: utf-8 -*-
"""Central configuration definitions and version metadata for the benchmark."""

import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).parent.absolute()


# ============================================================
# ============================================================

@dataclass
class DataConfig:
    """DataConfig class."""
    path: str = str(PROJECT_ROOT / 'input.csv')
    encoding: str = 'latin-1'
    target_T: str = 'T'
    target_P: str = 'P'

    feature_sets: Dict[str, List[str]] = field(default_factory=lambda: {
        'NoLiquid': [
            'SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx', 'Cr2O3.cpx',
            'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'CaO.cpx', 'Na2O.cpx'
        ],
        'Liquid': [
            'SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx', 'Cr2O3.cpx',
            'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'CaO.cpx', 'Na2O.cpx',
            'SiO2.liq', 'TiO2.liq', 'Al2O3.liq', 'FeO.liq',
            'MgO.liq', 'MnO.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq'
        ],
    })




@dataclass
class CVConfig:
    """CVConfig class."""
    n_splits: int = 10
    random_seed: int = 42


@dataclass
class OutputConfig:
    """OutputConfig class."""
    output_dir: str = str(PROJECT_ROOT / 'results')
    test_output_dir: str = str(PROJECT_ROOT / 'results_test')
    log_dir: str = str(PROJECT_ROOT / 'results' / 'logs')


@dataclass
class CatBoostConfig:
    """CatBoostConfig class."""
    iterations: int = 1000
    depth: int = 6
    learning_rate: float = 0.03
    loss_function: str = 'RMSE'
    task_type: str = 'auto'
    gpu_devices: str = '0'


@dataclass
class ModelDefaults:
    """ModelDefaults class."""
    ert: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
        'n_jobs': 4,
    })

    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)

    stacking: Dict[str, Any] = field(default_factory=lambda: {
        'inner_cv': 5,
        'use_meta_scaler': True,
    })

    stacking_base_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    rf: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    })


@dataclass
class AugmentationConfig:
    """AugmentationConfig class."""
    n_aug: int = 15



@dataclass
class UncertaintyConfig:
    """UncertaintyConfig class."""
    n_mc: int = 1000
    percentiles: tuple = (5, 16, 50, 84, 95)


@dataclass
class Config:
    """Config class."""
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelDefaults = field(default_factory=ModelDefaults)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


def get_version_info() -> Dict[str, Any]:
    """get_version_info function."""
    from src.repro import code_fingerprint, combined_sha, git_state

    info: Dict[str, Any] = {
        'python_version': sys.version,
        'platform': sys.platform,
        'timestamp': datetime.now().isoformat(),
    }

    git = git_state(PROJECT_ROOT)
    info['git_commit'] = git['commit']
    info['git_dirty'] = git['dirty']

    fp = code_fingerprint(PROJECT_ROOT)
    info['code_sha'] = combined_sha(fp)
    info['code_n_files'] = len(fp)

    import numpy
    import pandas
    import sklearn
    import catboost
    import scipy

    info['dependencies'] = {
        'numpy': numpy.__version__,
        'pandas': pandas.__version__,
        'scikit-learn': sklearn.__version__,
        'catboost': catboost.__version__,
        'scipy': scipy.__version__,
    }

    return info


# ============================================================
# ============================================================

CONFIG = Config()


# ============================================================
# ============================================================

def get_config_dict() -> Dict[str, Any]:
    """get_config_dict function."""
    return {
        'data_path': CONFIG.data.path,
        'data_encoding': CONFIG.data.encoding,
        'feature_sets': CONFIG.data.feature_sets,
        'target_T': CONFIG.data.target_T,
        'target_P': CONFIG.data.target_P,
        'n_splits': CONFIG.cv.n_splits,
        'random_seed': CONFIG.cv.random_seed,
        'output_dir': CONFIG.output.output_dir,
        'test_output_dir': CONFIG.output.test_output_dir,

        'model_defaults': {
            'ert': CONFIG.model.ert,           # ExtraTrees: n_estimators=200, max_depth=15
            'catboost': {
                'iterations': CONFIG.model.catboost.iterations,
                'depth': CONFIG.model.catboost.depth,
                'learning_rate': CONFIG.model.catboost.learning_rate,
                'loss_function': CONFIG.model.catboost.loss_function,
                'task_type': CONFIG.model.catboost.task_type,
                'gpu_devices': CONFIG.model.catboost.gpu_devices,
            },
            'stacking': CONFIG.model.stacking,  # Stacking: inner_cv=5
            'stacking_base_defaults': CONFIG.model.stacking_base_defaults,
            'rf': CONFIG.model.rf,              # RandomForest: n_estimators=200, max_depth=15
        },

        'augmentation': {
            'n_aug': CONFIG.augmentation.n_aug,
        },

        'uncertainty': {
            'n_mc': CONFIG.uncertainty.n_mc,
            'percentiles': CONFIG.uncertainty.percentiles,
        },
    }




# ============================================================
# ============================================================

if __name__ == '__main__':
    print("=== Config Module Test ===\n")

    print("--- Default Config ---")
    print(f"Data path: {CONFIG.data.path}")
    print(f"CV folds: {CONFIG.cv.n_splits}")
    print(f"Random seed: {CONFIG.cv.random_seed}")
    print(f"Output directory: {CONFIG.output.output_dir}")
    print(f"Feature sets: {list(CONFIG.data.feature_sets.keys())}")
    print(f"CatBoost task_type: {CONFIG.model.catboost.task_type}")

    print("\n--- Version Info ---")
    version_info = get_version_info()
    for k, v in version_info.items():
        if k == 'dependencies':
            print("  Dependency versions:")
            for dep, ver in v.items():
                print(f"    {dep}: {ver}")
        else:
            print(f"  {k}: {v}")

    print("\n--- Config as Dict ---")
    config_dict = get_config_dict()
    print(f"  data_path: {config_dict['data_path']}")
    print(f"  n_splits: {config_dict['n_splits']}")

    print("\nConfig module test passed.")

