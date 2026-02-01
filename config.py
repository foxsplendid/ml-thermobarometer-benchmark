# -*- coding: utf-8 -*-
"""
集中配置管理模块

项目唯一配置来源，所有模块通过此处获取配置：
    from config import CONFIG, PROJECT_ROOT, get_config_dict, get_version_info

支持：
1. 默认配置（代码内定义）
2. YAML 文件覆盖（config.yaml，可选）
3. 版本信息收集（用于结果追溯）
"""

import sys
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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
    group_col: str = 'Ref'

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
    """模型默认参数"""
    ert: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    })
    catboost: CatBoostConfig = field(default_factory=CatBoostConfig)
    stacking: Dict[str, Any] = field(default_factory=lambda: {
        'inner_cv': 5,
        'use_meta_scaler': True,
    })
    rf: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 5,
    })


@dataclass
class AugmentationConfig:
    """数据增强配置（EPMA误差模型）"""
    n_aug: int = 15
    error_model: str = 'epma'
    rel_err_high: float = 0.03      # >1 wt% 使用 3% 误差
    rel_err_low: float = 0.08       # ≤1 wt% 使用 8% 误差
    error_threshold: float = 1.0    # wt% 阈值
    clip_min: float = 0.0           # 最小值裁剪


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
    加载配置，支持 YAML 文件覆盖

    Parameters
    ----------
    config_path : str, optional
        YAML 配置文件路径，None 则使用默认配置

    Returns
    -------
    Config
        配置实例
    """
    config = Config()

    # 尝试加载 YAML 覆盖
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'

    if Path(config_path).exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            if yaml_config:
                # 覆盖数据配置
                if 'data' in yaml_config:
                    for k, v in yaml_config['data'].items():
                        if hasattr(config.data, k):
                            setattr(config.data, k, v)

                # 覆盖 CV 配置
                if 'cv' in yaml_config:
                    for k, v in yaml_config['cv'].items():
                        if hasattr(config.cv, k):
                            setattr(config.cv, k, v)

                # 覆盖输出配置
                if 'output' in yaml_config:
                    for k, v in yaml_config['output'].items():
                        if hasattr(config.output, k):
                            setattr(config.output, k, v)
        except Exception as e:
            print(f"警告: 加载配置文件失败 ({e})，使用默认配置")

    return config


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

def apply_seed(params: Dict[str, Any], keys: List[str], seed: int) -> Dict[str, Any]:
    """
    为参数字典注入随机种子

    Parameters
    ----------
    params : Dict[str, Any]
        原始参数字典
    keys : List[str]
        要注入的键名列表（如 ['random_seed', 'random_state']）
    seed : int
        随机种子值

    Returns
    -------
    Dict[str, Any]
        更新后的参数字典（不修改原字典）
    """
    updated = dict(params)
    for key in keys:
        if key not in updated:
            updated[key] = seed
    return updated


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
    """
    return {
        'data_path': CONFIG.data.path,
        'data_encoding': CONFIG.data.encoding,
        'feature_sets': CONFIG.data.feature_sets,
        'target_T': CONFIG.data.target_T,
        'target_P': CONFIG.data.target_P,
        'group_col': CONFIG.data.group_col,
        'n_splits': CONFIG.cv.n_splits,
        'random_seed': CONFIG.cv.random_seed,
        'output_dir': CONFIG.output.output_dir,
    }


def get_legacy_config() -> Dict[str, Any]:
    """向后兼容别名"""
    return get_config_dict()


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
