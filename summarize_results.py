# -*- coding: utf-8 -*-
"""
快速汇总实验结果脚本
从已完成的24个实验中读取结果并生成分析报告
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')

def load_metrics_summary():
    """加载汇总指标文件"""
    path = os.path.join(RESULTS_DIR, 'metrics_summary.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def aggregate_fold_metrics():
    """从每折指标文件汇总结果"""
    all_results = []
    
    # 遍历所有fold_metrics文件
    for fname in os.listdir(RESULTS_DIR):
        if fname.endswith('_fold_metrics.csv'):
            fpath = os.path.join(RESULTS_DIR, fname)
            df = pd.read_csv(fpath)
            
            # 解析文件名: E01_ert_raw_none_noliq_T_fold_metrics.csv
            parts = fname.replace('_fold_metrics.csv', '').rsplit('_', 1)
            exp_id = parts[0]  # E01_ert_raw_none_noliq
            target = parts[1]  # T or P
            
            # 计算mean和std
            for col in ['rmse', 'mae', 'r2', 'mbe']:
                if col in df.columns:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    all_results.append({
                        'exp_id': exp_id,
                        'target': target,
                        'metric': col,
                        'mean': mean_val,
                        'std': std_val,
                    })
    
    return pd.DataFrame(all_results)

def pivot_results(df):
    """将结果透视为宽格式"""
    # 创建透视表
    pivot_df = df.pivot_table(
        index='exp_id',
        columns=['target', 'metric'],
        values='mean'
    )
    
    # 展平多级列名
    pivot_df.columns = [f'{t}_{m}' for t, m in pivot_df.columns]
    pivot_df = pivot_df.reset_index()
    
    return pivot_df

def parse_exp_id(exp_id):
    """解析实验ID获取模块信息"""
    # E01_ert_raw_none_noliq
    parts = exp_id.split('_')
    if len(parts) >= 5:
        return {
            'exp_num': parts[0],
            'model': parts[1],
            'data': parts[2],
            'corr': parts[3],
            'feature_set': 'NoLiquid' if parts[4] == 'noliq' else 'Liquid'
        }
    return {}

def compute_effect_table(pivot_df):
    """计算模块效应表"""
    # 添加模块信息列
    for idx, row in pivot_df.iterrows():
        info = parse_exp_id(row['exp_id'])
        for k, v in info.items():
            pivot_df.at[idx, k] = v
    
    effects = []
    
    # 按不同维度分组计算效应
    for target in ['T', 'P']:
        rmse_col = f'{target}_rmse'
        r2_col = f'{target}_r2'
        
        if rmse_col not in pivot_df.columns:
            continue
        
        # 1. 特征集效应 (Liquid vs NoLiquid)
        for fs in ['NoLiquid', 'Liquid']:
            subset = pivot_df[pivot_df['feature_set'] == fs]
            if len(subset) > 0:
                effects.append({
                    'factor': 'feature_set',
                    'level': fs,
                    'target': target,
                    'rmse_mean': subset[rmse_col].mean(),
                    'r2_mean': subset[r2_col].mean(),
                    'n_exp': len(subset)
                })
        
        # 2. 模型效应
        for model in ['ert', 'catboost', 'stacking']:
            subset = pivot_df[pivot_df['model'] == model]
            if len(subset) > 0:
                effects.append({
                    'factor': 'model',
                    'level': model,
                    'target': target,
                    'rmse_mean': subset[rmse_col].mean(),
                    'r2_mean': subset[r2_col].mean(),
                    'n_exp': len(subset)
                })
        
        # 3. 数据模块效应
        for data in ['raw', 'balanced', 'augmented']:
            subset = pivot_df[pivot_df['data'] == data]
            if len(subset) > 0:
                effects.append({
                    'factor': 'data',
                    'level': data,
                    'target': target,
                    'rmse_mean': subset[rmse_col].mean(),
                    'r2_mean': subset[r2_col].mean(),
                    'n_exp': len(subset)
                })
        
        # 4. 校正模块效应
        for corr in ['none', 'residual']:
            subset = pivot_df[pivot_df['corr'] == corr]
            if len(subset) > 0:
                effects.append({
                    'factor': 'correction',
                    'level': corr,
                    'target': target,
                    'rmse_mean': subset[rmse_col].mean(),
                    'r2_mean': subset[r2_col].mean(),
                    'n_exp': len(subset)
                })
    
    return pd.DataFrame(effects)

def main():
    print("=" * 70)
    print("ML温压计评估实验结果汇总")
    print("=" * 70)
    
    # 1. 加载/汇总指标
    summary = load_metrics_summary()
    if summary is not None:
        print(f"\n从 metrics_summary.csv 加载了 {len(summary)} 条记录")
    
    # 2. 从fold文件汇总
    fold_df = aggregate_fold_metrics()
    print(f"从fold文件汇总了 {len(fold_df)} 条指标记录")
    
    # 3. 透视为宽格式
    pivot_df = pivot_results(fold_df)
    print(f"\n实验数量: {len(pivot_df)}")
    
    # 4. 打印核心指标
    print("\n" + "=" * 70)
    print("24个实验核心指标汇总 (按实验ID排序)")
    print("=" * 70)
    
    # 选择显示列
    display_cols = ['exp_id', 'T_rmse', 'T_r2', 'P_rmse', 'P_r2']
    available_cols = [c for c in display_cols if c in pivot_df.columns]
    
    # 按exp_id排序
    pivot_df_sorted = pivot_df.sort_values('exp_id')
    
    # 格式化输出
    pd.set_option('display.max_rows', 30)
    pd.set_option('display.width', 120)
    pd.set_option('display.precision', 3)
    
    print(pivot_df_sorted[available_cols].to_string(index=False))

    # Add parsed fields
    pivot_df_sorted = pivot_df_sorted.reset_index(drop=True)
    info_df = pivot_df_sorted['exp_id'].apply(parse_exp_id).apply(pd.Series)
    pivot_df_sorted = pd.concat([pivot_df_sorted, info_df], axis=1)
    parse_cols = ['exp_num', 'model', 'data', 'corr', 'feature_set']
    ordered_cols = ['exp_id'] + [c for c in parse_cols if c in pivot_df_sorted.columns]
    ordered_cols += [c for c in pivot_df_sorted.columns if c not in ordered_cols]
    pivot_df_sorted = pivot_df_sorted[ordered_cols]

    
    # 6. 最佳配置
    print("\n" + "=" * 70)
    print("最佳配置推荐")
    print("=" * 70)
    
    for target in ['T', 'P']:
        rmse_col = f'{target}_rmse'
        r2_col = f'{target}_r2'
        
        if rmse_col in pivot_df.columns:
            best_idx = pivot_df[rmse_col].idxmin()
            best = pivot_df.loc[best_idx]
            unit = '°C' if target == 'T' else 'kbar'
            print(f"\n{target}模型最佳配置: {best['exp_id']}")
            print(f"  RMSE = {best[rmse_col]:.2f} {unit}")
            print(f"  R2   = {best[r2_col]:.4f}")
    
    # 8. 保存汇总表
    summary_path = os.path.join(RESULTS_DIR, 'experiment_summary.csv')
    pivot_df_sorted.to_csv(summary_path, index=False)
    print(f"汇总表已保存: {summary_path}")
    
    return pivot_df_sorted

if __name__ == '__main__':
    main()
