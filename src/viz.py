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

# ============================================================
# 论文核心可视化函数（新增）
# ============================================================

def plot_stepwise_rmse_comparison(results_dict: Dict[str, pd.DataFrame],
                                   target: str = 'T',
                                   save_path: Optional[str] = None,
                                   figsize: Tuple[int, int] = (10, 6)):
    """
    阶梯误差对比图（论文核心图1）

    展示 Exp1→Exp5 的 RMSE 递减趋势，量化各模块贡献

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        实验结果字典 {exp_name: metrics_df}
    target : str
        目标变量（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    # 提取各实验的 RMSE 均值和标准差
    exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']
    exp_labels = ['Exp1\n基线', 'Exp2\n仅增强', 'Exp3\n仅校正', 'Exp4\n增强+校正', 'Exp5\nStacking']

    rmse_means = []
    rmse_stds = []

    for exp_name in exp_names:
        if exp_name in results_dict:
            df = results_dict[exp_name]
            col_name = f'rmse_{target}' if f'rmse_{target}' in df.columns else f'{target}_rmse'
            rmse_means.append(df[col_name].mean())
            rmse_stds.append(df[col_name].std())
        else:
            rmse_means.append(np.nan)
            rmse_stds.append(np.nan)

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    x_pos = np.arange(len(exp_names))
    bars = ax.bar(x_pos, rmse_means, yerr=rmse_stds,
                  capsize=5, alpha=0.7,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # 标注相对 Exp1 的提升百分比
    baseline_rmse = rmse_means[0]
    for i, (mean_val, bar) in enumerate(zip(rmse_means, bars)):
        if i > 0 and not np.isnan(mean_val):
            improvement = (baseline_rmse - mean_val) / baseline_rmse * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + rmse_stds[i] + 0.5,
                   f'↓{improvement:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('实验组', fontsize=12)
    ax.set_ylabel(f'RMSE ({["℃", "kbar"][target == "P"]})', fontsize=12)
    ax.set_title(f'{"温度" if target == "T" else "压力"}预测性能对比（阶梯误差图）', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(exp_labels)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    return fig


def plot_correction_effect(preds_df: pd.DataFrame,
                           exp_name: str,
                           target: str = 'T',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 6)):
    """
    校正前后散点图对比（论文核心图2）

    双子图展示校正效果，斜率从偏离1回归到~1

    Parameters
    ----------
    preds_df : pd.DataFrame
        预测数据框（必须包含 {target}_true, {target}_pred_raw, {target}_pred_corr 列）
    exp_name : str
        实验名称
    target : str
        目标变量（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    from .metrics import compute_slope_intercept, rmse

    # 提取数据
    y_true = preds_df[f'{target}_true'].values
    y_pred_raw = preds_df[f'{target}_pred_raw'].values
    y_pred_corr = preds_df[f'{target}_pred_corr'].values

    unit = '℃' if target == 'T' else 'kbar'

    # 计算指标
    slope_raw, intercept_raw = compute_slope_intercept(y_true, y_pred_raw)
    slope_corr, intercept_corr = compute_slope_intercept(y_true, y_pred_corr)
    rmse_raw = rmse(y_true, y_pred_raw)
    rmse_corr = rmse(y_true, y_pred_corr)

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图：校正前
    ax1.scatter(y_true, y_pred_raw, alpha=0.4, s=15, label='预测值')
    lims = [min(y_true.min(), y_pred_raw.min()), max(y_true.max(), y_pred_raw.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    ax1.plot(lims, lims, 'r--', linewidth=2, label='1:1 理想线')

    # 绘制实际回归线
    x_fit = np.array(lims)
    y_fit = slope_raw * x_fit + intercept_raw
    ax1.plot(x_fit, y_fit, 'b-', linewidth=2, alpha=0.7, label='实际回归线')

    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(f'{target} 真实值 ({unit})', fontsize=11)
    ax1.set_ylabel(f'{target} 预测值 ({unit})', fontsize=11)
    ax1.set_title(f'校正前（{exp_name}）', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')

    # 添加指标文本
    text_raw = f'RMSE = {rmse_raw:.2f} {unit}\nSlope = {slope_raw:.3f}\nIntercept = {intercept_raw:.2f}'
    ax1.text(0.95, 0.05, text_raw, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：校正后
    ax2.scatter(y_true, y_pred_corr, alpha=0.4, s=15, color='green', label='校正后预测值')
    ax2.plot(lims, lims, 'r--', linewidth=2, label='1:1 理想线')

    y_fit_corr = slope_corr * x_fit + intercept_corr
    ax2.plot(x_fit, y_fit_corr, 'g-', linewidth=2, alpha=0.7, label='实际回归线')

    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(f'{target} 真实值 ({unit})', fontsize=11)
    ax2.set_ylabel(f'{target} 预测值 ({unit})', fontsize=11)
    ax2.set_title(f'校正后（{exp_name}）', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')

    text_corr = f'RMSE = {rmse_corr:.2f} {unit}\nSlope = {slope_corr:.3f}\nIntercept = {intercept_corr:.2f}'
    ax2.text(0.95, 0.05, text_corr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    return fig


def plot_feature_importance(model,
                           feature_names: List[str],
                           target: str = 'T',
                           top_n: int = 20,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    特征重要性图（论文核心图3）

    使用 CatBoost 的 get_feature_importance() 提取重要性

    Parameters
    ----------
    model : CatBoostRegressor
        训练好的 CatBoost 模型
    feature_names : List[str]
        特征名称列表
    target : str
        目标变量（用于标题）
    top_n : int
        显示 Top N 特征
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    # 提取特征重要性
    try:
        importances = model.get_feature_importance()
    except AttributeError:
        # 如果是包装器，尝试访问内部模型
        if hasattr(model, '_model'):
            importances = model._model.get_feature_importance()
        else:
            raise ValueError("模型不支持 get_feature_importance() 方法")

    # 创建 DataFrame 并排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # 绘图
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'].values, alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'].values)
    ax.invert_yaxis()  # 最重要的在顶部
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title(f'{"温度" if target == "T" else "压力"}预测的 Top {top_n} 特征重要性',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    return fig


def plot_residual_distribution_comparison(results_dict: Dict[str, pd.DataFrame],
                                          exp_names: List[str] = ['exp4_aug_corr', 'exp5_stacking'],
                                          target: str = 'T',
                                          save_path: Optional[str] = None,
                                          figsize: Tuple[int, int] = (10, 6)):
    """
    残差分布对比图（论文核心图4）

    叠加直方图 + 核密度估计，对比 Exp4 vs Exp5

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        预测结果字典 {exp_name: preds_df}
    exp_names : List[str]
        要对比的实验名称列表
    target : str
        目标变量（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    unit = '℃' if target == 'T' else 'kbar'

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_map = {
        'exp4_aug_corr': 'Exp4 (CatBoost + 增强 + 校正)',
        'exp5_stacking': 'Exp5 (Stacking + 增强 + 校正)'
    }

    for i, exp_name in enumerate(exp_names):
        if exp_name in results_dict:
            preds_df = results_dict[exp_name]
            residual_col = f'{target}_residual'

            if residual_col in preds_df.columns:
                residuals = preds_df[residual_col].values
            else:
                # 计算残差
                residuals = preds_df[f'{target}_true'].values - preds_df[f'{target}_pred_corr'].values

            # 绘制直方图 + KDE
            ax.hist(residuals, bins=30, alpha=0.4, color=colors[i],
                   label=labels_map.get(exp_name, exp_name), density=True)

            # KDE 曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linewidth=2, alpha=0.8)

            # 添加统计量
            mean_resid = np.mean(residuals)
            std_resid = np.std(residuals)
            ax.axvline(mean_resid, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(f'残差 ({unit})', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title(f'{"温度" if target == "T" else "压力"}预测残差分布对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axvline(0, color='red', linestyle=':', linewidth=2, label='零残差线')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

    return fig


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
