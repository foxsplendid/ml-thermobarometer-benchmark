# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 可视化模块
Viz Module: 散点图、残差图、实验对比图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, List, Tuple, Any

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 单目标可视化
# ============================================================

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                      target_name: str = 'T', unit: str = '℃',
                      ax: Optional[plt.Axes] = None,
                      figsize: Tuple[int, int] = (6, 6),
                      title: Optional[str] = None,
                      show_metrics: bool = True,
                      alpha: float = 0.5) -> plt.Axes:
    """
    预测值 vs 真实值散点图
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    target_name : str
        目标名称（'T' 或 'P'）
    unit : str
        单位
    ax : plt.Axes, optional
        绑定的坐标轴
    figsize : tuple
        图形大小
    title : str, optional
        自定义标题
    show_metrics : bool
        是否显示 RMSE 和 R²
    alpha : float
        点透明度
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=alpha, s=15, edgecolors='none')
    
    # 1:1 参考线
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='1:1 线')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # 标签和标题
    ax.set_xlabel(f'{target_name} 真实值 ({unit})')
    ax.set_ylabel(f'{target_name} 预测值 ({unit})')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{target_name} 预测 vs 真实')
    
    # 添加指标
    if show_metrics:
        from .metrics import rmse, r2
        rmse_val = rmse(y_true, y_pred)
        r2_val = r2(y_true, y_pred)
        text = f'RMSE = {rmse_val:.2f} {unit}\nR² = {r2_val:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   target_name: str = 'T', unit: str = '℃',
                   ax: Optional[plt.Axes] = None,
                   figsize: Tuple[int, int] = (8, 5),
                   show_hist: bool = True) -> plt.Axes:
    """
    残差分布图
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    target_name : str
        目标名称
    unit : str
        单位
    show_hist : bool
        是否显示残差直方图
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_pred - y_true
    
    if show_hist:
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            ax1, ax2 = axes
        else:
            ax1 = ax
            ax2 = None
    else:
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(6, 5))
        else:
            ax1 = ax
        ax2 = None
    
    # 残差 vs 预测值
    ax1.scatter(y_pred, residuals, alpha=0.5, s=15, edgecolors='none')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
    ax1.set_xlabel(f'{target_name} 预测值 ({unit})')
    ax1.set_ylabel(f'残差 ({unit})')
    ax1.set_title(f'{target_name} 残差分布')
    
    # 添加残差统计
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    text = f'均值 = {mean_res:.2f}\n标准差 = {std_res:.2f}'
    ax1.text(0.95, 0.95, text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 残差直方图
    if ax2 is not None:
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
        ax2.set_xlabel(f'残差 ({unit})')
        ax2.set_ylabel('频数')
        ax2.set_title(f'{target_name} 残差直方图')
    
    plt.tight_layout()
    return ax1


# ============================================================
# 折叠结果可视化
# ============================================================

def plot_fold_comparison(metrics_df: pd.DataFrame,
                         target: str = 'T',
                         metric: str = 'rmse',
                         figsize: Tuple[int, int] = (8, 5)) -> plt.Figure:
    """
    各折指标对比柱状图
    
    Parameters
    ----------
    metrics_df : pd.DataFrame
        各折指标表（需包含 fold_id 列）
    target : str
        目标名称（'T' 或 'P'）
    metric : str
        指标名称（'rmse', 'mae', 'r2'）
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    col_name = f'{metric}_{target}'
    if col_name not in metrics_df.columns:
        raise ValueError(f"列 {col_name} 不存在")
    
    x = metrics_df['fold_id'].values
    y = metrics_df[col_name].values
    
    bars = ax.bar(x, y, edgecolor='black', alpha=0.8)
    
    # 添加均值线
    mean_val = np.mean(y)
    ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1.5, label=f'均值 = {mean_val:.4f}')
    
    ax.set_xlabel('Fold')
    ax.set_ylabel(f'{metric.upper()}')
    ax.set_title(f'{target} - {metric.upper()} 各折对比')
    ax.set_xticks(x)
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_experiment_summary(results_df: pd.DataFrame,
                            figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    实验汇总热力图
    
    Parameters
    ----------
    results_df : pd.DataFrame
        各实验汇总结果表（需包含 exp_name 列）
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 选择要显示的指标
    metrics_T = ['rmse_T_mean', 'mae_T_mean', 'r2_T_mean']
    metrics_P = ['rmse_P_mean', 'mae_P_mean', 'r2_P_mean']
    
    for ax, metrics, target in [(axes[0], metrics_T, 'T'), (axes[1], metrics_P, 'P')]:
        available_metrics = [m for m in metrics if m in results_df.columns]
        if not available_metrics:
            continue
        
        data = results_df[['exp_name'] + available_metrics].set_index('exp_name')
        
        # 简化列名
        data.columns = [c.replace(f'_{target}_mean', '') for c in data.columns]
        
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax,
                    cbar_kws={'label': '指标值'})
        ax.set_title(f'{target} 指标汇总')
        ax.set_ylabel('实验')
    
    plt.tight_layout()
    return fig


# ============================================================
# 完整报告
# ============================================================

def plot_full_report(y_T_true: np.ndarray, y_T_pred: np.ndarray,
                     y_P_true: np.ndarray, y_P_pred: np.ndarray,
                     exp_name: str = '',
                     figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    生成完整评估报告图
    
    包含 T 和 P 的预测-真实散点图和残差图
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # T 预测 vs 真实
    plot_pred_vs_true(y_T_true, y_T_pred, target_name='T', unit='℃', ax=axes[0, 0])
    
    # T 残差
    plot_residuals(y_T_true, y_T_pred, target_name='T', unit='℃', ax=axes[0, 1], show_hist=False)
    
    # P 预测 vs 真实
    plot_pred_vs_true(y_P_true, y_P_pred, target_name='P', unit='kbar', ax=axes[1, 0])
    
    # P 残差
    plot_residuals(y_P_true, y_P_pred, target_name='P', unit='kbar', ax=axes[1, 1], show_hist=False)
    
    if exp_name:
        fig.suptitle(f'实验报告: {exp_name}', fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
    """保存图形"""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"图形已保存: {filepath}")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    print("=== 可视化示例 ===")
    
    # 模拟数据
    np.random.seed(42)
    n = 200
    y_T_true = np.random.uniform(800, 1200, n)
    y_T_pred = y_T_true + np.random.normal(0, 30, n)
    y_P_true = np.random.uniform(0.1, 20, n)
    y_P_pred = y_P_true + np.random.normal(0, 1.5, n)
    
    # 生成完整报告
    fig = plot_full_report(y_T_true, y_T_pred, y_P_true, y_P_pred, exp_name='示例实验')
    plt.show()
