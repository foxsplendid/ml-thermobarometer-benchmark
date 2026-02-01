# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - 主入口

使用方法：
    python main.py          # 运行完整实验矩阵
    python main.py --test   # 快速测试（2折，4个实验）

输出：
    results/
    ├── metrics_summary.csv
    ├── effect_table.csv
    ├── config_used.yaml
    ├── {exp_id}_{T/P}_fold_metrics.csv
    └── {exp_id}_{T/P}_predictions.parquet

注意：
    - 绘图功能请使用 tools/plot_offline_figures.py
    - 稳定性测试请使用 tools/run_stability.py
    - MC不确定性测试请使用 tools/run_mc_uncertainty.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd

# 仅过滤常见的无害警告，保留关键警告类别
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*divide by zero.*')

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# 配置（从 config.py 集中管理）
# ============================================================
from config import get_config_dict

CONFIG = get_config_dict()


# ============================================================
# 实验矩阵定义
# ============================================================

def get_experiment_configs():
    """
    定义实验矩阵（24个实验：12个基础配置 × 2种特征集）

    核心因果矩阵：M1 × M2 × M3 × 特征集
    - M1: raw / balanced / augmented
    - M2: ert / catboost / stacking
    - M3: none / segmented
    - 特征集: NoLiquid / Liquid

    设计原则（V5更新）：
    - E01-E03: Raw 基线组（M1=Raw, M3=None）
    - E04-E06: Balanced 对比组（M1=Balanced, M3=None）
    - E07-E09: Augmented + M2 对比组（M1=Augmented, M3=None）- 完整的模型对比
    - E10-E12: Augmented + M3 对比组（M1=Augmented, M3=Segmented）- 验证校正无收益

    控制变量原则：在评估 M2/M3 时，固定使用最佳 M1 配置（Augmented）
    """
    from src.protocol import ExperimentConfig

    # 定义12个基础配置（所有实验均不启用不确定性和随机划分，这些功能在tools中运行）
    base_configs = [
        # === E01-E03: Raw 基线组 ===
        # E01: Raw + ERT + None（基线）
        {'data': 'raw', 'model': 'ert', 'corr': 'none'},
        # E02: Raw + CatBoost + None（基线）
        {'data': 'raw', 'model': 'catboost', 'corr': 'none'},
        # E03: Raw + Stacking + None（基线）
        {'data': 'raw', 'model': 'stacking', 'corr': 'none'},

        # === E04-E06: Balanced 对比组（传统数据平衡方法）===
        # E04: Balanced + ERT + None
        {'data': 'balanced', 'model': 'ert', 'corr': 'none'},
        # E05: Balanced + CatBoost + None
        {'data': 'balanced', 'model': 'catboost', 'corr': 'none'},
        # E06: Balanced + Stacking + None
        {'data': 'balanced', 'model': 'stacking', 'corr': 'none'},

        # === E07-E09: Augmented + M2 对比组（完整的模型对比）===
        # E07: Augmented + ERT + None（最佳配置 ⭐）
        {'data': 'augmented', 'model': 'ert', 'corr': 'none'},
        # E08: Augmented + CatBoost + None
        {'data': 'augmented', 'model': 'catboost', 'corr': 'none'},
        # E09: Augmented + Stacking + None（V5新增：补全M2对比）
        {'data': 'augmented', 'model': 'stacking', 'corr': 'none'},

        # === E10-E12: Augmented + M3 对比组（验证校正无收益）===
        # E10: Augmented + ERT + Segmented
        {'data': 'augmented', 'model': 'ert', 'corr': 'segmented'},
        # E11: Augmented + CatBoost + Segmented
        {'data': 'augmented', 'model': 'catboost', 'corr': 'segmented'},
        # E12: Augmented + Stacking + Segmented
        {'data': 'augmented', 'model': 'stacking', 'corr': 'segmented'},
    ]

    # 为每个配置生成NoLiquid和Liquid两个版本
    final_configs = []
    for idx, base in enumerate(base_configs, start=1):
        for fset in ['NoLiquid', 'Liquid']:
            suffix = 'noliq' if fset == 'NoLiquid' else 'liq'
            exp_id = f"E{idx:02d}_{base['model']}_{base['data']}_{base['corr']}_{suffix}"

            final_configs.append(ExperimentConfig(
                exp_id=exp_id,
                data_module_name=base['data'],
                model_module_name=base['model'],
                corr_module_name=base['corr'],
                feature_set=fset,
                run_uncertainty=False,  # 不确定性测试在tools中运行
                run_random_split=False,  # 随机划分测试在tools中运行
            ))

    return final_configs  # 24个实验


# ============================================================
# 数据加载
# ============================================================

def load_data(config, feature_set='Liquid'):
    """加载并准备数据

    Parameters
    ----------
    config : dict
        配置字典
    feature_set : str
        特征集名称，'NoLiquid' 或 'Liquid'

    Returns
    -------
    X, y_T, y_P, groups
    """
    print(f"加载数据（特征集: {feature_set}）...")

    df = pd.read_csv(config['data_path'], encoding=config['data_encoding'])

    # 清理数据
    dirty_patterns = ['Unnamed:', 'ï..']
    cols_to_drop = [col for col in df.columns if any(p in col for p in dirty_patterns)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # 获取指定特征集
    feature_cols = config['feature_sets'][feature_set]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺失特征列: {missing_cols}")

    X = df[feature_cols].values.astype(np.float64)
    y_T = df[config['target_T']].values.astype(np.float64)
    y_P = df[config['target_P']].values.astype(np.float64)
    groups = df[config['group_col']].values

    # 填充缺失值
    X = np.nan_to_num(X, nan=0.0)

    print(f"  特征集: {feature_set} ({len(feature_cols)}个特征)")
    print(f"  数据形状: X={X.shape}")
    print(f"  温度范围: {y_T.min():.0f} - {y_T.max():.0f} ℃")
    print(f"  压力范围: {y_P.min():.2f} - {y_P.max():.2f} kbar")
    print(f"  分组数量: {len(np.unique(groups))}")
    
    return X, y_T, y_P, groups


# ============================================================
# 数据划分
# ============================================================

def prepare_splits(X, y_T, y_P, groups, config):
    """准备测试集划分（P-T网格采样）

    注意：不再使用Ref分组约束，优先保证P-T分布平衡
    """
    from src.splitters import compute_pt_edges, assign_pt_bins, select_test_indices

    print("\n准备测试集划分（P-T网格采样）...")

    bins = compute_pt_edges(y_T, y_P)
    tp_bins = assign_pt_bins(y_T, y_P, bins)

    # 简化的测试集选择：每个P-T bin选择一个样本
    test_idx = select_test_indices(
        tp_bins,
        random_state=config['random_seed']
    )

    train_mask = np.ones(len(X), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    # 统计P-T分布
    test_tp_bins = tp_bins[test_idx]
    unique_bins, bin_counts = np.unique(test_tp_bins, return_counts=True)

    print(f"  测试集大小: {len(test_idx)} (从{len(np.unique(tp_bins))}个非空P-T bins采样)")
    print(f"  P-T bins覆盖: {len(unique_bins)}/{len(np.unique(tp_bins))}")
    print(f"  每bin样本数: min={bin_counts.min()}, max={bin_counts.max()}, mean={bin_counts.mean():.2f}")

    split_info = {
        'test_indices': test_idx.tolist(),
        'test_size': int(len(test_idx)),
        'p_edges': bins.p_edges.tolist(),
        't_edges': bins.t_edges.tolist(),
        'n_pt_bins': len(unique_bins),
    }

    return {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'tp_bins_train': tp_bins[train_idx],
        'split_info': split_info,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    """主入口"""
    print("=" * 70)
    print("Chapter 3 Benchmark Protocol - 模块化验证框架")
    print("=" * 70)

    # 1. 加载数据（使用默认Liquid特征集进行分割）
    # 注意：不同特征集使用相同的P-T分箱和测试集划分
    X_liquid, y_T, y_P, groups = load_data(CONFIG, feature_set='Liquid')

    split_data = prepare_splits(X_liquid, y_T, y_P, groups, CONFIG)
    split_info = split_data['split_info']
    print(f"Test size: {split_info['test_size']}, P-T bins: {split_info['n_pt_bins']}")

    # 提取训练集和测试集索引（用于所有特征集）
    train_idx = split_data['train_idx']
    test_idx = split_data['test_idx']
    tp_bins_train = split_data['tp_bins_train']

    # 2. 获取实验配置
    configs = get_experiment_configs()
    print(f"\n实验数量: {len(configs)} (12个基础配置 × 2种特征集)")

    # 3. 按特征集分组运行实验
    all_results = []

    for feature_set in ['NoLiquid', 'Liquid']:
        print(f"\n{'='*70}")
        print(f"运行特征集: {feature_set}")
        print(f"{'='*70}")

        # 加载该特征集的数据
        X, y_T, y_P, groups = load_data(CONFIG, feature_set=feature_set)

        # 使用相同的train/test划分
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_T_train = y_T[train_idx]
        y_T_test = y_T[test_idx]
        y_P_train = y_P[train_idx]
        y_P_test = y_P[test_idx]
        groups_train = groups[train_idx]

        # 筛选该特征集的实验配置
        feature_configs = [c for c in configs if c.feature_set == feature_set]
        print(f"该特征集实验数: {len(feature_configs)}")

        # 创建实验矩阵执行器
        from src.protocol import ExperimentMatrix

        matrix = ExperimentMatrix(
            X=X_train,
            y_T=y_T_train,
            y_P=y_P_train,
            groups=groups_train,
            output_dir=CONFIG['output_dir'],
        )

        # 运行实验
        summary_df = matrix.run_experiments(
            configs=feature_configs,
            n_splits=CONFIG['n_splits'],
            stratify_labels=tp_bins_train,
            X_test=X_test,
            y_T_test=y_T_test,
            y_P_test=y_P_test,
            random_seed=CONFIG['random_seed'],
            verbose=True,
        )

        all_results.append(summary_df)

    # 4. 合并所有结果
    summary_df = pd.concat(all_results, ignore_index=True)

    # 5. 计算效应表
    matrix.compute_effect_table(summary_df)

    # 6. 保存配置
    matrix.save_config(configs, extra_info={
        'n_splits': CONFIG['n_splits'],
        'random_seed': CONFIG['random_seed'],
        'test_split': split_info,
        'feature_sets': list(CONFIG['feature_sets'].keys()),
        'n_features_by_feature_set': {k: len(v) for k, v in CONFIG['feature_sets'].items()},
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
    """快速测试（2折，前2个基础配置 × 2种特征集 = 4个实验）"""
    print("=" * 70)
    print("快速测试模式（2折，4个实验）")
    print("=" * 70)

    # 修改配置
    test_config = CONFIG.copy()
    test_config['output_dir'] = os.path.join(PROJECT_ROOT, 'results_test')

    # 1. 加载数据（使用Liquid特征集进行分割）
    X_liquid, y_T, y_P, groups = load_data(test_config, feature_set='Liquid')

    split_data = prepare_splits(X_liquid, y_T, y_P, groups, test_config)
    split_info = split_data['split_info']
    print(f"Test size: {split_info['test_size']}, P-T bins: {split_info['n_pt_bins']}")

    # 提取训练集和测试集索引
    train_idx = split_data['train_idx']
    test_idx = split_data['test_idx']
    tp_bins_train = split_data['tp_bins_train']

    # 2. 获取前2个基础配置（快速测试）
    all_configs = get_experiment_configs()
    # 筛选前2个基础配置的所有特征集版本（共4个实验）
    test_configs = [c for c in all_configs if any(c.exp_id.startswith(f"E{i:02d}") for i in [1, 2])]
    print(f"\n快速测试实验数: {len(test_configs)}")

    # 3. 按特征集分组运行
    all_results = []

    for feature_set in ['NoLiquid', 'Liquid']:
        print(f"\n{'='*70}")
        print(f"运行特征集: {feature_set}")
        print(f"{'='*70}")

        # 加载该特征集的数据
        X, y_T, y_P, groups = load_data(test_config, feature_set=feature_set)

        # 使用相同的train/test划分
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_T_train = y_T[train_idx]
        y_T_test = y_T[test_idx]
        y_P_train = y_P[train_idx]
        y_P_test = y_P[test_idx]
        groups_train = groups[train_idx]

        # 筛选该特征集的实验配置
        feature_configs = [c for c in test_configs if c.feature_set == feature_set]
        print(f"该特征集实验数: {len(feature_configs)}")

        # 创建实验矩阵执行器
        from src.protocol import ExperimentMatrix

        matrix = ExperimentMatrix(
            X=X_train,
            y_T=y_T_train,
            y_P=y_P_train,
            groups=groups_train,
            output_dir=test_config['output_dir'],
        )

        # 运行实验（2折快速测试）
        summary_df = matrix.run_experiments(
            configs=feature_configs,
            n_splits=2,  # 快速测试使用2折
            stratify_labels=tp_bins_train,
            X_test=X_test,
            y_T_test=y_T_test,
            y_P_test=y_P_test,
            random_seed=test_config['random_seed'],
            verbose=True,
        )

        all_results.append(summary_df)

    # 4. 合并结果
    summary_df = pd.concat(all_results, ignore_index=True)

    # 5. 打印结果
    print("\n" + "=" * 70)
    print("快速测试完成！")
    print("=" * 70)

    display_cols = ['exp_id', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

    print(f"\n输出目录: {test_config['output_dir']}")
    print("=" * 70)

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
