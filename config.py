# -*- coding: utf-8 -*-
"""
集中配置管理模块

项目唯一配置来源，所有模块通过此处获取配置：
    from config import CONFIG, PROJECT_ROOT, get_config_dict, get_version_info

说明：
1. 默认配置（代码内定义）
2. 版本信息收集（用于结果追溯）
"""

import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()


# ============================================================
# 配置数据类
# ============================================================

@dataclass
class DataConfig:
    """数据配置"""
    path: str = str(PROJECT_ROOT / 'input.csv')
    encoding: str = 'latin-1'
    target_T: str = 'T'
    target_P: str = 'P'

    # 特征集定义
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


# 注意：EPMA 误差配置已统一由 src/perturbation.py 管理
# 详见 perturbation.py::DEFAULT_OXIDE_REL_ERR 和 get_rel_err_vector()


@dataclass
class CVConfig:
    """交叉验证配置"""
    n_splits: int = 10
    random_seed: int = 42


@dataclass
class OutputConfig:
    """输出配置"""
    output_dir: str = str(PROJECT_ROOT / 'results')
    test_output_dir: str = str(PROJECT_ROOT / 'results_test')
    log_dir: str = str(PROJECT_ROOT / 'results' / 'logs')


@dataclass
class CatBoostConfig:
    """CatBoost 专用配置"""
    iterations: int = 1000
    depth: int = 6
    learning_rate: float = 0.03
    loss_function: str = 'RMSE'
    # GPU 控制：'auto' 自动检测 | 'CPU' 强制 CPU | 'GPU' 强制 GPU
    task_type: str = 'auto'
    gpu_devices: str = '0'


@dataclass
class ModelDefaults:
    """
    模型默认参数 - 经过实验调优的设计值

    ⚠️ 警告：这些参数基于 Chapter 3 实验验证，不建议随意修改。
    修改可能导致：
    - 性能下降（特别是 max_depth、n_estimators）
    - 训练时间显著变化
    - 与论文结果不可复现

    如需调参，建议：
    1. 先在小数据集上验证
    2. 使用学习曲线工具评估影响
    3. 记录修改原因和结果对比
    """
    # ExtraTrees 参数（集成基线，高方差低偏差）
    ert: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,      # 树的数量，增加可提高稳定性但增加训练时间
        'max_depth': 15,          # 最大深度，过深易过拟合
        'min_samples_split': 5,   # 最小分裂样本数
        'n_jobs': 4,
    })

    # CatBoost 参数（强单模型，Boosting 方法）
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)

    # Stacking 参数（元学习集成）
    stacking: Dict[str, Any] = field(default_factory=lambda: {
        'inner_cv': 5,            # 内层 CV 折数，用于生成元特征
        'use_meta_scaler': True,  # 元特征标准化
    })

    # Stacking 基模型参数覆盖（可选，默认使用 ert/catboost/rf 的参数）
    # 格式：{'ert': {...}, 'catboost': {...}, 'rf': {...}}
    stacking_base_defaults: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # RandomForest 参数（经典集成，作为 Stacking 基学习器）
    rf: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    })


@dataclass
class AugmentationConfig:
    """数据增强配置（EPMA误差模型）"""
    n_aug: int = 15              # 每个原始样本生成的增强副本数
    # 注：具体误差参数由 src/perturbation.py 管理



@dataclass
class UncertaintyConfig:
    """MC不确定性配置"""
    n_mc: int = 1000
    percentiles: tuple = (5, 16, 50, 84, 95)


@dataclass
class Config:
    """主配置类 - 聚合所有子配置"""
    data: DataConfig = field(default_factory=DataConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    model: ModelDefaults = field(default_factory=ModelDefaults)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)


# ============================================================
# 配置加载与版本管理
# ============================================================

def load_config(config_path: Optional[str] = None) -> Config:
    """
    加载配置（仅使用代码内默认配置）。

    Parameters
    ----------
    config_path : str, optional
        兼容保留，当前不会读取外部文件。

    Returns
    -------
    Config
        配置实例
    """
    _ = config_path
    return Config()


def get_version_info() -> Dict[str, Any]:
    """
    收集版本信息，用于结果追溯

    Returns
    -------
    Dict[str, Any]
        包含 Python 版本、Git commit、依赖版本等信息
    """
    info = {
        'python_version': sys.version,
        'platform': sys.platform,
        'timestamp': datetime.now().isoformat(),
    }

    # Git commit hash
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        if result.returncode == 0:
            info['git_commit'] = result.stdout.strip()

        # Git dirty status
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT)
        )
        info['git_dirty'] = bool(result.stdout.strip())
    except Exception:
        info['git_commit'] = 'unknown'
        info['git_dirty'] = None

    # 关键依赖版本
    dependencies = {}
    try:
        import numpy
        dependencies['numpy'] = numpy.__version__
    except ImportError:
        pass

    try:
        import pandas
        dependencies['pandas'] = pandas.__version__
    except ImportError:
        pass

    try:
        import sklearn
        dependencies['scikit-learn'] = sklearn.__version__
    except ImportError:
        pass

    try:
        import catboost
        dependencies['catboost'] = catboost.__version__
    except ImportError:
        pass

    try:
        import scipy
        dependencies['scipy'] = scipy.__version__
    except ImportError:
        pass

    if dependencies:
        info['dependencies'] = dependencies

    return info


# ============================================================
# 辅助函数
# ============================================================

def get_feature_names(feature_set: str) -> List[str]:
    """
    获取特征集的特征名称列表

    Parameters
    ----------
    feature_set : str
        特征集名称: 'NoLiquid' 或 'Liquid'

    Returns
    -------
    List[str]
        特征名称列表
    """
    config = Config()
    feature_set_normalized = 'Liquid' if feature_set.lower() in ['liq', 'liquid'] else 'NoLiquid'
    return config.data.feature_sets.get(feature_set_normalized, [])


# ============================================================
# 全局默认配置实例
# ============================================================

CONFIG = load_config()


# ============================================================
# 字典形式配置（供 main.py 和 tools 使用）
# ============================================================

def get_config_dict() -> Dict[str, Any]:
    """
    将 CONFIG 对象转换为字典形式

    这是项目统一的配置获取接口，main.py 和 tools/*.py 都应使用此函数

    返回的配置项分为两类：
    1. 运行时配置：数据路径、CV 参数、输出目录等，可根据需要修改
    2. 模型默认参数：经过调优的设计值，不建议随意修改（见 model_defaults）
    """
    return {
        # === 运行时配置（可根据需要修改）===
        'data_path': CONFIG.data.path,
        'data_encoding': CONFIG.data.encoding,
        'feature_sets': CONFIG.data.feature_sets,
        'target_T': CONFIG.data.target_T,
        'target_P': CONFIG.data.target_P,
        'n_splits': CONFIG.cv.n_splits,
        'random_seed': CONFIG.cv.random_seed,
        'output_dir': CONFIG.output.output_dir,
        'test_output_dir': CONFIG.output.test_output_dir,  # 测试模式输出目录

        # === 模型默认参数（经过调优，不建议随意修改）===
        # 这些参数基于实验验证，修改可能导致性能下降
        'model_defaults': {
            'ert': CONFIG.model.ert,           # ExtraTrees: n_estimators=200, max_depth=15
            'catboost': {                       # CatBoost 配置
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

        # === 数据增强配置（基于 Ágreda-López 2024，不建议随意修改）===
        'augmentation': {
            'n_aug': CONFIG.augmentation.n_aug,  # 每样本增强副本数，默认 15
        },

        # === MC 不确定性配置 ===
        'uncertainty': {
            'n_mc': CONFIG.uncertainty.n_mc,
            'percentiles': CONFIG.uncertainty.percentiles,
        },
    }




# ============================================================
# 模块测试
# ============================================================

if __name__ == '__main__':
    print("=== 配置模块测试 ===\n")

    print("--- 默认配置 ---")
    print(f"数据路径: {CONFIG.data.path}")
    print(f"CV 折数: {CONFIG.cv.n_splits}")
    print(f"随机种子: {CONFIG.cv.random_seed}")
    print(f"输出目录: {CONFIG.output.output_dir}")
    print(f"特征集: {list(CONFIG.data.feature_sets.keys())}")
    print(f"CatBoost task_type: {CONFIG.model.catboost.task_type}")

    print("\n--- 版本信息 ---")
    version_info = get_version_info()
    for k, v in version_info.items():
        if k == 'dependencies':
            print(f"  依赖版本:")
            for dep, ver in v.items():
                print(f"    {dep}: {ver}")
        else:
            print(f"  {k}: {v}")

    print("\n--- 字典形式配置 ---")
    config_dict = get_config_dict()
    print(f"  data_path: {config_dict['data_path']}")
    print(f"  n_splits: {config_dict['n_splits']}")

    print("\n✅ 配置模块测试通过！")
