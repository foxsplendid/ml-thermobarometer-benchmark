# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - 主入口
Main Entry Point: 一键运行实验矩阵

使用方法：
    python main.py

输出：
    results/
    ├── metrics_summary.csv
    ├── effect_table.csv
    ├── config_used.yaml
    ├── {exp_id}_{T/P}_fold_metrics.csv
    ├── {exp_id}_{T/P}_predictions.parquet
    └── figures/
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 配置
# ============================================================

CONFIG = {
    # 数据配置
    'data_path': os.path.join(PROJECT_ROOT, 'input.csv'),
    'data_encoding': 'latin-1',
    
    # 特征配置
    'feature_cols': [
        # CPX 氧化物（12列）
        'SiO2.cpx', 'Al2O3.cpx', 'TiO2.cpx', 'CaO.cpx', 'Na2O.cpx', 'K2O.cpx',
        'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'Cr2O3.cpx', 'NiO.cpx', 'P2O5.cpx',
        # LIQ 氧化物（12列）
        'SiO2.liq', 'Al2O3.liq', 'TiO2.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq',
        'FeO.liq', 'MgO.liq', 'MnO.liq', 'Cr2O3.liq', 'NiO.liq', 'P2O5.liq',
        # CPX 阳离子（12列）
        'Si.cpx', 'Al.cpx', 'Ti.cpx', 'Ca.cpx', 'Na.cpx', 'K.cpx',
        'Fe.cpx', 'Mg.cpx', 'Mn.cpx', 'Cr.cpx', 'Ni.cpx', 'P.cpx',
    ],
    'target_T': 'T',
    'target_P': 'P',
    'group_col': 'Ref',
    
    # CV 配置
    'n_splits': 10,
    'random_seed': 42,
    
    # 输出配置
    'output_dir': os.path.join(PROJECT_ROOT, 'results'),
}


# ============================================================
# 实验矩阵定义
# ============================================================

def get_experiment_configs():
    """
    定义实验矩阵
    
    核心因果矩阵：M1 × M2 × M3
    - M1: raw / balanced / augmented
    - M2: ert / catboost / stacking
    - M3: none / residual
    """
    from src.protocol import ExperimentConfig
    
    configs = [
        # ===== 基线实验 (Raw + 无校正) =====
        ExperimentConfig(
            exp_id='E01_ert_raw_none',
            data_module_name='raw',
            model_module_name='ert',
            corr_module_name='none',
        ),
        ExperimentConfig(
            exp_id='E02_catboost_raw_none',
            data_module_name='raw',
            model_module_name='catboost',
            corr_module_name='none',
            run_random_split=True,  # 对照
        ),
        ExperimentConfig(
            exp_id='E03_stacking_raw_none',
            data_module_name='raw',
            model_module_name='stacking',
            corr_module_name='none',
            run_random_split=True,  # Stacking 对照
        ),
        
        # ===== M1 消融 (Balanced + 无校正) =====
        ExperimentConfig(
            exp_id='E04_ert_balanced_none',
            data_module_name='balanced',
            model_module_name='ert',
            corr_module_name='none',
        ),
        ExperimentConfig(
            exp_id='E05_catboost_balanced_none',
            data_module_name='balanced',
            model_module_name='catboost',
            corr_module_name='none',
        ),
        ExperimentConfig(
            exp_id='E06_stacking_balanced_none',
            data_module_name='balanced',
            model_module_name='stacking',
            corr_module_name='none',
        ),
        
        # ===== M1 消融 (Augmented + 无校正) =====
        ExperimentConfig(
            exp_id='E07_ert_augmented_none',
            data_module_name='augmented',
            model_module_name='ert',
            corr_module_name='none',
        ),
        ExperimentConfig(
            exp_id='E08_catboost_augmented_none',
            data_module_name='augmented',
            model_module_name='catboost',
            corr_module_name='none',
        ),
        
        # ===== M3 消融 (Raw + 校正) =====
        ExperimentConfig(
            exp_id='E09_catboost_raw_residual',
            data_module_name='raw',
            model_module_name='catboost',
            corr_module_name='residual',
        ),
        
        # ===== 完整流程 (Balanced + 校正) =====
        ExperimentConfig(
            exp_id='E10_ert_balanced_residual',
            data_module_name='balanced',
            model_module_name='ert',
            corr_module_name='residual',
        ),
        ExperimentConfig(
            exp_id='E11_catboost_balanced_residual',  # ⭐ 主力配置
            data_module_name='balanced',
            model_module_name='catboost',
            corr_module_name='residual',
            run_uncertainty=True,  # 启用 MC
            run_random_split=True,
        ),
        ExperimentConfig(
            exp_id='E12_stacking_balanced_residual',  # 边界探索
            data_module_name='balanced',
            model_module_name='stacking',
            corr_module_name='residual',
            run_uncertainty=True,  # 启用 MC
            run_random_split=True,
        ),
    ]
    
    return configs


# ============================================================
# 数据加载
# ============================================================

def load_data(config):
    """加载并准备数据"""
    print("加载数据...")
    
    df = pd.read_csv(config['data_path'], encoding=config['data_encoding'])
    
    # 清理数据
    dirty_patterns = ['Unnamed:', 'ï..']
    cols_to_drop = [col for col in df.columns if any(p in col for p in dirty_patterns)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    # 提取特征和目标
    feature_cols = config['feature_cols']
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失特征列: {missing_cols}")
    
    X = df[feature_cols].values.astype(np.float64)
    y_T = df[config['target_T']].values.astype(np.float64)
    y_P = df[config['target_P']].values.astype(np.float64)
    groups = df[config['group_col']].values
    
    # 填充缺失值
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"  数据形状: X={X.shape}")
    print(f"  温度范围: {y_T.min():.0f} - {y_T.max():.0f} ℃")
    print(f"  压力范围: {y_P.min():.2f} - {y_P.max():.2f} kbar")
    print(f"  分组数量: {len(np.unique(groups))}")
    
    return X, y_T, y_P, groups


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口"""
    print("=" * 70)
    print("Chapter 3 Benchmark Protocol - 模块化验证框架")
    print("=" * 70)
    
    # 1. 加载数据
    X, y_T, y_P, groups = load_data(CONFIG)
    
    # 2. 获取实验配置
    configs = get_experiment_configs()
    print(f"\n实验数量: {len(configs)}")
    
    # 3. 创建实验矩阵执行器
    from src.protocol import ExperimentMatrix
    
    matrix = ExperimentMatrix(
        X=X,
        y_T=y_T,
        y_P=y_P,
        groups=groups,
        output_dir=CONFIG['output_dir'],
    )
    
    # 4. 运行实验
    print("\n" + "=" * 70)
    print("开始运行实验矩阵...")
    print("=" * 70)
    
    summary_df = matrix.run_experiments(
        configs=configs,
        n_splits=CONFIG['n_splits'],
        verbose=True,
    )
    
    # 5. 计算效应表
    effect_df = matrix.compute_effect_table(summary_df)
    
    # 6. 保存配置
    matrix.save_config(configs, extra_info={
        'n_splits': CONFIG['n_splits'],
        'random_seed': CONFIG['random_seed'],
    })
    
    # 7. 打印汇总
    print("\n" + "=" * 70)
    print("实验完成！结果汇总：")
    print("=" * 70)
    
    display_cols = ['exp_id', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))
    
    print(f"\n输出目录: {CONFIG['output_dir']}")
    print("=" * 70)
    
    return summary_df


def run_quick_test():
    """快速测试（2折，前3个实验）"""
    print("=" * 70)
    print("快速测试模式（2折，前3个实验）")
    print("=" * 70)
    
    # 修改配置
    test_config = CONFIG.copy()
    test_config['output_dir'] = os.path.join(PROJECT_ROOT, 'results_test')
    
    # 加载数据
    X, y_T, y_P, groups = load_data(test_config)
    
    # 只运行前3个实验
    configs = get_experiment_configs()[:3]
    
    from src.protocol import ExperimentMatrix
    
    matrix = ExperimentMatrix(
        X=X, y_T=y_T, y_P=y_P, groups=groups,
        output_dir=test_config['output_dir'],
    )
    
    summary_df = matrix.run_experiments(
        configs=configs,
        n_splits=2,  # 快速测试用2折
        verbose=True,
    )
    
    print("\n✅ 快速测试完成！")
    return summary_df


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Chapter 3 Benchmark Protocol')
    parser.add_argument('--test', action='store_true', help='运行快速测试')
    args = parser.parse_args()
    
    if args.test:
        run_quick_test()
    else:
        main()
