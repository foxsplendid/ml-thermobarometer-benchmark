# -*- coding: utf-8 -*-
"""
可视化模块 - 散点图、残差图、实验对比图

注意：本模块需要 matplotlib 和 seaborn，如未安装将无法使用绑图功能
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Dict, List, Tuple

# 可视化库导入保护
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None


def _check_plotting_available():
    """检查绑图库是否可用"""
    if not HAS_PLOTTING:
        raise ImportError(
            "可视化功能需要 matplotlib 和 seaborn，"
            "请运行: pip install matplotlib seaborn"
        )


# ============================================================
# 单目标可视化
# ============================================================

def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray,
                      target_name: str = 'T', unit: str = '',
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
        透明度
    """
    _check_plotting_available()

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
    ax.plot(lims, lims, 'r--', linewidth=1.5, label='1:1 line')
    ax.set_xlim((float(lims[0]), float(lims[1])))
    ax.set_ylim((float(lims[0]), float(lims[1])))

    # 标签和标题
    ax.set_xlabel(f'{target_name} True ({unit})')
    ax.set_ylabel(f'{target_name} Predicted ({unit})')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'{target_name} Predicted vs True')

    # 添加指标
    if show_metrics:
        from .metrics import rmse, r2
        rmse_val = rmse(y_true, y_pred)
        r2_val = r2(y_true, y_pred)
        text = f'RMSE = {rmse_val:.2f} {unit}\nR$^2$ = {r2_val:.4f}'
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='lower right')
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   target_name: str = 'T', unit: str = '',
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
    ax1.set_xlabel(f'{target_name} Predicted ({unit})')
    ax1.set_ylabel(f'Residual ({unit})')
    ax1.set_title(f'{target_name} Residual Distribution')

    # 添加残差统计
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    text = f'Mean = {mean_res:.2f}\nStd = {std_res:.2f}'
    ax1.text(0.95, 0.95, text, transform=ax1.transAxes, fontsize=9,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 残差直方图
    if ax2 is not None:
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=1.5)
        ax2.set_xlabel(f'Residual ({unit})')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{target_name} Residual Histogram')

    plt.tight_layout()
    return ax1


# ============================================================
# 折叠可视化
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
        指标数据框，必须包含 fold_id 列
    target : str
        目标名称（'T' 或 'P'）
    metric : str
        指标名称（'rmse', 'mae', 'r2'）
    """
    fig, ax = plt.subplots(figsize=figsize)

    col_name = f'{target}_{metric}'
    if col_name not in metrics_df.columns:
        raise ValueError(f"列 {col_name} 不存在")

    x = metrics_df['fold_id'].values
    y = metrics_df[col_name].values

    bars = ax.bar(x, y, edgecolor='black', alpha=0.8)

    # 添加均值线
    mean_val = float(np.mean(y))
    ax.axhline(y=mean_val, color='r', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.4f}')

    ax.set_xlabel('Fold')
    ax.set_ylabel(f'{metric.upper()}')
    ax.set_title(f'{target} - {metric.upper()} Fold Comparison')
    ax.set_xticks(x)
    ax.legend()

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
    完整报告图

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
        fig.suptitle(f'Experiment Report: {exp_name}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150) -> None:
    """保存图表"""
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存: {filepath}")


# ============================================================
# 稳定性可视化
# ============================================================

def plot_stability_overview(stability_T: pd.DataFrame,
                            stability_P: pd.DataFrame,
                            metrics: Tuple[str, ...] = ("rmse", "mae", "mbe"),
                            bins: int = 30,
                            figsize: Tuple[int, int] = (12, 10),
                            title: Optional[str] = None) -> plt.Figure:
    """
    稳定性分布图（重复实验）

    以分布形式展示重复实验的波动范围，便于观察稳定性。
    """
    _check_plotting_available()

    unit_map = {"T": "℃", "P": "kbar"}
    colors = {"T": "#1f77b4", "P": "#ff7f0e"}
    unitless = {"r2", "slope", "bins_merged", "n_splits_requested", "n_splits_used",
                "n_bins_raw", "n_bins_merged", "repeat_id"}

    def _metric_label(metric: str, unit: str) -> str:
        label = metric.replace("_", " ").upper()
        if metric in unitless:
            return label
        return f"{label} ({unit})"

    metrics = tuple(metrics)
    n_rows = len(metrics)
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, squeeze=False)

    for col, (target_name, df) in enumerate([("T", stability_T), ("P", stability_P)]):
        unit = unit_map.get(target_name, "")
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            if metric not in df.columns:
                raise ValueError(f"stability metric column missing: {metric}")

            values = df[metric].dropna().values
            if values.size == 0:
                ax.set_title(f"{target_name} {metric.upper()} (empty)")
                ax.set_axis_off()
                continue

            sns.histplot(values, bins=bins, kde=True, stat="density",
                         color=colors[target_name], alpha=0.6, ax=ax)

            mean_val = float(np.mean(values))
            std_val = float(np.std(values))
            ax.axvline(mean_val, color="black", linestyle="--", linewidth=1.2)

            label = _metric_label(metric, unit)
            ax.set_title(f"{target_name} {label}")
            ax.set_xlabel(label)
            ax.set_ylabel("Density")
            ax.text(
                0.98, 0.95,
                f"mean={mean_val:.2f}\nstd={std_val:.2f}",
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


# ============================================================
# 学习曲线可视化
# ============================================================

def plot_learning_curve(summary_df: pd.DataFrame,
                        target: str = "T",
                        figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    学习曲线图（基于汇总统计）
    """
    _check_plotting_available()

    df = summary_df[summary_df["target"] == target].copy()
    if df.empty:
        raise ValueError(f"learning curve summary missing target: {target}")

    fig, ax = plt.subplots(figsize=figsize)

    models = df["model"].unique()
    colors = {"ert": "#1f77b4", "stacking": "#ff7f0e", "catboost": "#2ca02c"}
    markers = {"ert": "o", "stacking": "s", "catboost": "^"}

    for model in models:
        model_df = df[df["model"] == model].sort_values("fraction")
        x = model_df["n_train_sub_mean"].values
        y = model_df["rmse_mean_of_repeats"].values
        yerr = model_df["rmse_std_of_repeats"].values

        color = colors.get(model, "gray")
        marker = markers.get(model, "o")

        ax.errorbar(
            x, y, yerr=yerr, label=model.upper(),
            color=color, marker=marker, capsize=3, linewidth=2, markersize=8
        )

    ax.set_xlabel("Training Samples", fontsize=12)
    ylabel = "RMSE (°C)" if target == "T" else "RMSE (kbar)"
    ax.set_ylabel(ylabel, fontsize=12)
    title = f"Learning Curve - {'Temperature' if target == 'T' else 'Pressure'}"
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============================================================
# 四个核心可视化函数（论文级）
# ============================================================



def plot_correction_effect(preds_df: pd.DataFrame,
                           exp_name: str,
                           target: str = 'T',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (14, 6)):
    """
    校正前后散点图对比（论文图3-2）

    双面板展示校正效果：斜率从偏离1回归到~1

    Parameters
    ----------
    preds_df : pd.DataFrame
        预测数据框（必须包含 {target}_true, {target}_pred_raw, {target}_pred_corr 列）
    exp_name : str
        实验名称
    target : str
        目标名称（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    random_seed : int, optional
        设置抖动随机种子，为 None 时不固定
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

    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图：校正前
    ax1.scatter(y_true, y_pred_raw, alpha=0.4, s=15, label='Predicted')
    lims = [min(y_true.min(), y_pred_raw.min()), max(y_true.max(), y_pred_raw.max())]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    ax1.plot(lims, lims, 'r--', linewidth=2, label='1:1 line')

    # 实际回归线
    x_fit = np.array(lims)
    y_fit = slope_raw * x_fit + intercept_raw
    ax1.plot(x_fit, y_fit, 'b-', linewidth=2, alpha=0.7, label='Regression line')

    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(f'{target} True ({unit})', fontsize=11)
    ax1.set_ylabel(f'{target} Predicted ({unit})', fontsize=11)
    ax1.set_title(f'Before Correction ({exp_name})', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.set_aspect('equal')

    # 指标标签
    text_raw = f'RMSE = {rmse_raw:.2f} {unit}\nSlope = {slope_raw:.3f}\nIntercept = {intercept_raw:.2f}'
    ax1.text(0.95, 0.05, text_raw, transform=ax1.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：校正后
    ax2.scatter(y_true, y_pred_corr, alpha=0.4, s=15, color='green', label='Corrected')
    ax2.plot(lims, lims, 'r--', linewidth=2, label='1:1 line')

    y_fit_corr = slope_corr * x_fit + intercept_corr
    ax2.plot(x_fit, y_fit_corr, 'g-', linewidth=2, alpha=0.7, label='Regression line')

    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(f'{target} True ({unit})', fontsize=11)
    ax2.set_ylabel(f'{target} Predicted ({unit})', fontsize=11)
    ax2.set_title(f'After Correction ({exp_name})', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_aspect('equal')

    text_corr = f'RMSE = {rmse_corr:.2f} {unit}\nSlope = {slope_corr:.3f}\nIntercept = {intercept_corr:.2f}'
    ax2.text(0.95, 0.05, text_corr, transform=ax2.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_feature_importance(importances_or_model,
                           feature_names: List[str],
                           target: str = 'T',
                           top_n: int = 20,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)):
    """
    特征重要性图（论文图3-3）

    Parameters
    ----------
    importances_or_model : np.ndarray or model
        特征重要性数组，或具有 get_feature_importance() 方法的模型
    feature_names : List[str]
        特征名称列表
    target : str
        目标名称（用于标题）
    top_n : int
        显示 Top N 特征
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    # 获取特征重要性
    if isinstance(importances_or_model, np.ndarray):
        importances = importances_or_model
    else:
        # 尝试从模型获取
        model = importances_or_model
        try:
            importances = model.get_feature_importance()
        except AttributeError:
            # 如果是封装器，访问内部模型
            if hasattr(model, '_model'):
                importances = model._model.get_feature_importance()
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                raise ValueError("模型不支持 get_feature_importance() 方法")

    # 创建 DataFrame 排序
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'].values, alpha=0.7, color='steelblue')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'].values)
    ax.invert_yaxis()  # 最重要的在顶部
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'{"Temperature" if target == "T" else "Pressure"} Prediction Top {top_n} Feature Importance',
                fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_residual_distribution_comparison(results_dict: Dict[str, pd.DataFrame],
                                          exp_names: Optional[List[str]] = None,
                                          target: str = 'T',
                                          save_path: Optional[str] = None,
                                          figsize: Tuple[int, int] = (10, 6)):
    """
    残差分布对比图（论文图3-4）

    直方图 + 密度曲线，对比 Exp4 vs Exp5

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        预测结果字典 {exp_name: preds_df}
    exp_names : List[str]
        要对比的实验列表
    target : str
        目标名称（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    if exp_names is None:
        exp_names = ['exp4_aug_corr', 'exp5_stacking']

    unit = '℃' if target == 'T' else 'kbar'

    fig, ax = plt.subplots(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    labels_map = {
        'exp4_aug_corr': 'Exp4 (CatBoost + Augmented + Correction)',
        'exp5_stacking': 'Exp5 (Stacking + Augmented + Correction)'
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

            # 直方图 + KDE
            ax.hist(residuals, bins=30, alpha=0.4, color=colors[i],
                   label=labels_map.get(exp_name, exp_name), density=True)

            # KDE 曲线
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(residuals)
            x_range = np.linspace(residuals.min(), residuals.max(), 200)
            ax.plot(x_range, kde(x_range), color=colors[i], linewidth=2, alpha=0.8)

            # 均值线
            mean_resid = float(np.mean(residuals))
            ax.axvline(mean_resid, color=colors[i], linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel(f'Residual ({unit})', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{"Temperature" if target == "T" else "Pressure"} Prediction Residual Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axvline(0, color='red', linestyle=':', linewidth=2, label='Zero residual')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


# ============================================================
# 论文图件函数
# ============================================================

def plot_pt_grid_cv_splits(y_t: np.ndarray,
                           y_p: np.ndarray,
                           tp_bins: np.ndarray,
                           fold_assignments: np.ndarray,
                           p_edges: np.ndarray,
                           t_edges: np.ndarray,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    图 3-2：P-T 空间网格分层 CV 示意图

    展示 P-T 散点图，点按 fold 上色，背景叠加网格 bins

    Parameters
    ----------
    y_t : np.ndarray
        温度真实值
    y_p : np.ndarray
        压力真实值
    tp_bins : np.ndarray
        P-T bin 标签
    fold_assignments : np.ndarray
        每个样本的 fold ID (0-9)
    p_edges : np.ndarray
        压力 bin 边界
    t_edges : np.ndarray
        温度 bin 边界
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制网格背景
    for p_edge in p_edges:
        ax.axhline(y=p_edge, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    for t_edge in t_edges:
        ax.axvline(x=t_edge, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    # 按 fold 上色绘制散点
    n_folds = len(np.unique(fold_assignments))
    cmap = plt.cm.get_cmap('tab10', n_folds)

    for fold_id in range(n_folds):
        mask = fold_assignments == fold_id
        ax.scatter(y_t[mask], y_p[mask], c=[cmap(fold_id)], s=15, alpha=0.6,
                   label=f'Fold {fold_id}', edgecolors='none')

    ax.set_xlabel('Temperature T (°C)', fontsize=12)
    ax.set_ylabel('Pressure P (kbar)', fontsize=12)
    ax.set_title('P-T Grid Stratified Cross-Validation', fontsize=14, fontweight='bold')

    # 紧凑图例（左上角）
    ax.legend(loc='upper left', fontsize=8, ncol=2, markerscale=1.5)
    ax.grid(False)

    # 设置坐标范围
    ax.set_xlim(t_edges.min() - 10, t_edges.max() + 10)
    ax.set_ylim(p_edges.min() - 0.5, p_edges.max() + 0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_feature_set_comparison_boxplot(metrics_df: pd.DataFrame,
                                        target: str = 'T',
                                        metric: str = 'rmse',
                                        save_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (8, 6),
                                        random_seed: Optional[int] = 42) -> plt.Figure:
    """
    图 3-3：特征集效能对比箱线图

    对比 NoLiquid vs Liquid 的 RMSE 分布

    Parameters
    ----------
    metrics_df : pd.DataFrame
        汇总指标数据框，需包含 exp_id 列（用于识别特征集）
    target : str
        目标名称（'T' 或 'P'）
    metric : str
        指标名称（'rmse', 'mae', 'r2'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    random_seed : int, optional
        设置抖动随机种子，为 None 时不固定
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 从 exp_id 或 exp_name 提取特征集标识
    id_col = 'exp_id' if 'exp_id' in metrics_df.columns else 'exp_name'
    metric_col = f'{target}_{metric}_mean' if f'{target}_{metric}_mean' in metrics_df.columns else f'{target}_{metric}'

    if metric_col not in metrics_df.columns:
        print(f"警告: 列 {metric_col} 不存在")
        return fig

    df = metrics_df.copy()
    df['feature_set'] = df[id_col].apply(lambda x: 'NoLiquid' if '_noliq' in str(x) else 'Liquid')

    # 分组数据
    noliq_data = df[df['feature_set'] == 'NoLiquid'][metric_col].dropna().values
    liq_data = df[df['feature_set'] == 'Liquid'][metric_col].dropna().values

    # 箱线图
    box_data = [noliq_data, liq_data]
    positions = [1, 2]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True)

    # 颜色
    colors_box = ['#ff7f0e', '#1f77b4']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加散点
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

    for i, (data, pos) in enumerate(zip(box_data, positions)):
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(pos + jitter, data, alpha=0.4, s=20, color=colors_box[i], edgecolors='none')

    # 标注均值
    for i, (data, pos) in enumerate(zip(box_data, positions)):
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.text(pos, ax.get_ylim()[1] * 0.95, f'{mean_val:.1f}±{std_val:.1f}',
                ha='center', va='top', fontsize=10, fontweight='bold')

    ax.set_xticks(positions)
    ax.set_xticklabels(['NoLiquid\n(9 features)', 'Liquid\n(18 features)'])
    unit = '°C' if target == 'T' else 'kbar'
    ax.set_ylabel(f'{metric.upper()} ({unit})', fontsize=12)
    ax.set_title(f'{"Temperature" if target == "T" else "Pressure"} Prediction - Feature Set Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_parity_comparison(preds_noliq: Dict[str, np.ndarray],
                           preds_liq: Dict[str, np.ndarray],
                           target: str = 'T',
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
    """
    图 3-4：最佳模型 1:1 预测对比图（NoLiquid vs Liquid 并排）

    Parameters
    ----------
    preds_noliq : dict
        NoLiquid 预测结果，需包含 'y_true', 'y_pred' 键
    preds_liq : dict
        Liquid 预测结果，需包含 'y_true', 'y_pred' 键
    target : str
        目标名称（'T' 或 'P'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    from .metrics import rmse, r2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    unit = '°C' if target == 'T' else 'kbar'

    # 计算共用坐标范围
    all_values = np.concatenate([
        preds_noliq['y_true'], preds_noliq['y_pred'],
        preds_liq['y_true'], preds_liq['y_pred']
    ])
    lims = [all_values.min(), all_values.max()]
    margin = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - margin, lims[1] + margin]

    # 左图：NoLiquid
    ax1.scatter(preds_noliq['y_true'], preds_noliq['y_pred'], alpha=0.4, s=15, color='#ff7f0e', edgecolors='none')
    ax1.plot(lims, lims, 'r--', linewidth=2, label='1:1 line')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel(f'{target} True ({unit})', fontsize=11)
    ax1.set_ylabel(f'{target} Predicted ({unit})', fontsize=11)
    ax1.set_title(f'NoLiquid (9 features)', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    rmse_noliq = rmse(preds_noliq['y_true'], preds_noliq['y_pred'])
    r2_noliq = r2(preds_noliq['y_true'], preds_noliq['y_pred'])
    ax1.text(0.05, 0.95, f'RMSE = {rmse_noliq:.1f} {unit}\nR² = {r2_noliq:.3f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.legend(loc='lower right')

    # 右图：Liquid
    ax2.scatter(preds_liq['y_true'], preds_liq['y_pred'], alpha=0.4, s=15, color='#1f77b4', edgecolors='none')
    ax2.plot(lims, lims, 'r--', linewidth=2, label='1:1 line')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel(f'{target} True ({unit})', fontsize=11)
    ax2.set_ylabel(f'{target} Predicted ({unit})', fontsize=11)
    ax2.set_title(f'Liquid (18 features)', fontsize=12, fontweight='bold')
    ax2.set_aspect('equal')
    rmse_liq = rmse(preds_liq['y_true'], preds_liq['y_pred'])
    r2_liq = r2(preds_liq['y_true'], preds_liq['y_pred'])
    ax2.text(0.05, 0.95, f'RMSE = {rmse_liq:.1f} {unit}\nR² = {r2_liq:.3f}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2.legend(loc='lower right')

    # 总标题
    fig.suptitle(f'{"Temperature" if target == "T" else "Pressure"} Prediction - Feature Set Comparison (1:1 Plot)', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_m1_ablation_stepwise(metrics_df: pd.DataFrame,
                              target: str = 'T',
                              model: str = 'ert',
                              feature_set: str = 'liq',
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
    """
    图 3-5：M1 消融阶梯式效能提升图（Raw → Balanced → Augmented）

    固定 M2 模型和特征集，展示 M1 的独立贡献

    Parameters
    ----------
    metrics_df : pd.DataFrame
        汇总指标数据框
    target : str
        目标名称（'T' 或 'P'）
    model : str
        模型名称（'ert', 'catboost', 'stacking'）
    feature_set : str
        特征集后缀（'liq' 或 'noliq'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 查找对应实验
    id_col = 'exp_id' if 'exp_id' in metrics_df.columns else 'exp_name'
    metric_col = f'{target}_rmse_mean'
    std_col = f'{target}_rmse_std'

    # M1 处理策略顺序
    m1_order = ['raw', 'balanced', 'augmented']
    m1_labels = ['Raw\n(Baseline)', 'Balanced\n(Reweighted)', 'Augmented\n(Physics-based)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    rmse_means = []
    rmse_stds = []

    for m1 in m1_order:
        # 构建实验 ID 模式：E0X_{model}_{m1}_none_{feature_set}
        pattern = f'_{model}_{m1}_none_{feature_set}'
        matched = metrics_df[metrics_df[id_col].str.contains(pattern, na=False)]
        if len(matched) > 0:
            rmse_means.append(matched[metric_col].values[0])
            rmse_stds.append(matched[std_col].values[0] if std_col in matched.columns else 0)
        else:
            rmse_means.append(np.nan)
            rmse_stds.append(np.nan)

    # 绘制柱状图
    x_pos = np.arange(len(m1_order))
    bars = ax.bar(x_pos, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7, color=colors)

    # 标注相对基线的改善（在误差棒上方显示）
    baseline_rmse = rmse_means[0]
    for i, (mean_val, bar) in enumerate(zip(rmse_means, bars)):
        if i > 0 and not np.isnan(mean_val) and not np.isnan(baseline_rmse):
            improvement = (baseline_rmse - mean_val) / baseline_rmse * 100
            # 在误差棒顶部上方标注
            err_val = rmse_stds[i] if not np.isnan(rmse_stds[i]) else 0
            y_pos = bar.get_height() + err_val + 0.3
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f'{improvement:+.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='green' if improvement > 0 else 'red')

    # 设置y轴上限，留出足够空间给标注
    max_height = max(rmse_means) + max(rmse_stds) if rmse_stds else max(rmse_means)
    ax.set_ylim(0, max_height * 1.2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(m1_labels)
    unit = '°C' if target == 'T' else 'kbar'
    ax.set_ylabel(f'RMSE ({unit})', fontsize=12)
    model_name = {'ert': 'ERT', 'catboost': 'CatBoost', 'stacking': 'Stacking'}.get(model, model.upper())
    ax.set_title(f'{"Temperature" if target == "T" else "Pressure"} Prediction - M1 Data Module Ablation ({model_name})', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


def plot_performance_heatmap_matrix(metrics_df: pd.DataFrame,
                                    target: str = 'T',
                                    feature_set: str = 'liq',
                                    save_path: Optional[str] = None,
                                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    图 3-6：算法×处理×校正的性能热力图

    X 轴：算法（ERT/CatBoost/Stacking）
    Y 轴：处理策略（Raw/Balanced/Augmented/Augmented+Segmented）

    Parameters
    ----------
    metrics_df : pd.DataFrame
        汇总指标数据框
    target : str
        目标名称（'T' 或 'P'）
    feature_set : str
        特征集后缀（'liq' 或 'noliq'）
    save_path : str, optional
        保存路径
    figsize : tuple
        图形大小
    """
    fig, ax = plt.subplots(figsize=figsize)

    id_col = 'exp_id' if 'exp_id' in metrics_df.columns else 'exp_name'
    metric_col = f'{target}_rmse_mean'

    # 定义矩阵维度
    models = ['ert', 'catboost', 'stacking']
    model_labels = ['ERT', 'CatBoost', 'Stacking']

    # 处理策略（Y轴）
    treatments = [
        ('raw', 'none', 'Raw'),
        ('balanced', 'none', 'Balanced'),
        ('augmented', 'none', 'Augmented'),
        ('augmented', 'segmented', 'Aug+Segmented'),
    ]

    # 构建矩阵
    matrix = np.full((len(treatments), len(models)), np.nan)

    for i, (data_mod, corr_mod, _) in enumerate(treatments):
        for j, model in enumerate(models):
            pattern = f'_{model}_{data_mod}_{corr_mod}_{feature_set}'
            matched = metrics_df[metrics_df[id_col].str.contains(pattern, na=False)]
            if len(matched) > 0:
                matrix[i, j] = matched[metric_col].values[0]

    # 绘制热力图
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

    # 添加数值标注
    for i in range(len(treatments)):
        for j in range(len(models)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > np.nanmedian(matrix) else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=11, color=text_color, fontweight='bold')

    # 标记最优值（如果存在有效数据）
    if not np.all(np.isnan(matrix)):
        min_idx = np.unravel_index(np.nanargmin(matrix), matrix.shape)
        ax.add_patch(plt.Rectangle((min_idx[1] - 0.5, min_idx[0] - 0.5), 1, 1,
                                    fill=False, edgecolor='gold', linewidth=3))

    # 设置轴标签
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(model_labels, fontsize=11)
    ax.set_yticks(np.arange(len(treatments)))
    ax.set_yticklabels([t[2] for t in treatments], fontsize=11)

    ax.set_xlabel('Model (M2)', fontsize=12)
    ax.set_ylabel('Data Processing + Correction (M1 + M3)', fontsize=12)

    unit = '°C' if target == 'T' else 'kbar'
    ax.set_title(f'{"Temperature" if target == "T" else "Pressure"} Performance Matrix - RMSE ({unit})\n(Gold box = Best config)', fontsize=14, fontweight='bold')

    # 添加颜色条
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f'RMSE ({unit})', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存: {save_path}")

    return fig


# ============================================================
# 使用示例
# ============================================================

def plot_combined_shap_summary(shap_values: Any,
                               X: Any,
                               model_name: str = "Model",
                               max_display: int = 20,
                               figsize: Tuple[int, int] = (12, 10),
                               font_size: int = 12,
                               show_suptitle: bool = True,
                               show_bottom_axis_labels: bool = True) -> plt.Figure:
    """
    Draw combined SHAP dot+bar summary following BayesV5 notebook style.
    """
    _check_plotting_available()

    try:
        import shap as shap_lib
    except Exception as e:
        raise ImportError(f"SHAP is required for plot_combined_shap_summary: {e}")

    if isinstance(X, pd.DataFrame):
        x_df = X.copy()
    else:
        x_arr = np.asarray(X)
        if x_arr.ndim != 2:
            raise ValueError("X must be a 2D array-like for SHAP plotting")
        x_df = pd.DataFrame(x_arr, columns=[f"Feature_{i}" for i in range(x_arr.shape[1])])

    # Keep the notebook defaults when caller does not override.
    if max_display is None:
        max_display = 22 if "Meta" not in model_name else 6
    if figsize is None:
        figsize = (12, 10) if "Meta" not in model_name else (12, 6)

    alpha = 0.4
    dpi = 300
    show_colorbar = True

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    _ = ax1  # Placeholder to keep variable naming aligned with notebook style.

    title_name = model_name
    # Remove trailing short hash suffix if present (e.g., "_a1b2c3d4").
    if "_" in model_name:
        suffix = model_name.rsplit("_", 1)[-1]
        if len(suffix) == 8 and suffix.isalnum():
            title_name = model_name.rsplit("_", 1)[0]

    if show_suptitle:
        plt.suptitle(f"SHAP Analysis for {title_name}", fontsize=font_size + 2, y=0.98)

    shap_lib.summary_plot(
        shap_values,
        x_df,
        plot_type="dot",
        feature_names=x_df.columns,
        max_display=max_display,
        show=False,
        color_bar=show_colorbar,
    )
    plt.gca().set_position([0.5, 0.5, 0.65, 0.65])
    ax1 = plt.gca()

    ax2 = ax1.twiny()
    _ = ax2
    shap_lib.summary_plot(
        shap_values,
        x_df,
        plot_type="bar",
        feature_names=x_df.columns,
        max_display=max_display,
        show=False,
    )
    plt.gca().set_position([0.5, 0.5, 0.65, 0.65])
    ax2 = plt.gca()

    for bar in ax2.patches:
        bar.set_alpha(alpha)

    if show_bottom_axis_labels:
        ax1.set_xlabel("SHAP Value Contribution (Bee Swarm)", fontsize=font_size)
        ax2.set_xlabel("Mean SHAP Value (Feature Importance)", fontsize=font_size)
    else:
        ax1.set_xlabel("")
        ax2.set_xlabel("")
    ax1.set_ylabel("Features", fontsize=font_size)
    ax2.xaxis.set_label_position('top')
    ax2.xaxis.tick_top()

    plt.tight_layout()
    if show_suptitle:
        plt.subplots_adjust(top=0.92)
    else:
        plt.subplots_adjust(top=0.96)
    return fig


def plot_correction_delta_scatter_tp(t_true: np.ndarray,
                                     t_pred_raw: np.ndarray,
                                     t_pred_corr: np.ndarray,
                                     p_true: np.ndarray,
                                     p_pred_raw: np.ndarray,
                                     p_pred_corr: np.ndarray,
                                     title: str = "Segmented Correction Effect",
                                     t_unit: str = r"$^\circ$C",
                                     p_unit: str = "kbar",
                                     bg_color: str = "#ffffff",
                                     figsize: Tuple[int, int] = (12, 12.6)) -> plt.Figure:
    """
    Plot correction-delta figure with joint panels for T/P.
    Layout and styling follow the provided reference implementation.
    """
    _check_plotting_available()

    t_true = np.asarray(t_true).ravel()
    t_pred_raw = np.asarray(t_pred_raw).ravel()
    t_pred_corr = np.asarray(t_pred_corr).ravel()
    p_true = np.asarray(p_true).ravel()
    p_pred_raw = np.asarray(p_pred_raw).ravel()
    p_pred_corr = np.asarray(p_pred_corr).ravel()

    if not (len(t_true) == len(t_pred_raw) == len(t_pred_corr)):
        raise ValueError("T arrays must have the same length")
    if not (len(p_true) == len(p_pred_raw) == len(p_pred_corr)):
        raise ValueError("P arrays must have the same length")

    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    t_delta = t_pred_corr - t_pred_raw
    p_delta = p_pred_corr - p_pred_raw

    t_color = "#d95f02"
    p_color = "#2c7fb8"
    trend_color = "#d62728"
    ci_color = "#cfcfcf"
    boundary_color = "#111111"

    def _kernel_smooth_with_ci(x: np.ndarray, y: np.ndarray, n_grid: int = 320):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(x)
        if n < 10:
            order = np.argsort(x)
            xs = x[order]
            ys = y[order]
            return xs, ys, ys, ys

        x_min, x_max = float(np.min(x)), float(np.max(x))
        xs = np.linspace(x_min, x_max, n_grid)

        std = float(np.std(x, ddof=1)) if n > 1 else 0.0
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        sigma = min(std, iqr / 1.34) if iqr > 0 else std
        if sigma <= 0:
            sigma = max((x_max - x_min) / 10.0, 1e-6)
        h = 1.06 * sigma * (n ** (-1 / 5))
        h = max(h, (x_max - x_min) / 35.0)

        y_mean = np.empty_like(xs)
        y_lo = np.empty_like(xs)
        y_hi = np.empty_like(xs)

        for i, xg in enumerate(xs):
            z = (x - xg) / h
            w = np.exp(-0.5 * z * z)
            sw = np.sum(w)
            if sw <= 1e-12:
                y_mean[i] = np.nan
                y_lo[i] = np.nan
                y_hi[i] = np.nan
                continue

            mu = np.sum(w * y) / sw
            var = np.sum(w * (y - mu) ** 2) / sw
            n_eff = (sw ** 2) / np.sum(w ** 2)
            se = np.sqrt(max(var, 0.0) / max(n_eff, 1.0))

            y_mean[i] = mu
            y_lo[i] = mu - 1.96 * se
            y_hi[i] = mu + 1.96 * se

        valid = np.isfinite(y_mean)
        return xs[valid], y_mean[valid], y_lo[valid], y_hi[valid]

    def _draw_joint_block(fig_obj,
                          spec,
                          x_true,
                          y_delta,
                          y_pred_raw,
                          x_label,
                          y_label,
                          base_color,
                          legend_loc="upper left",
                          stat_pos=(0.98, 0.04),
                          stat_ha="right",
                          stat_va="bottom"):
        inner = GridSpecFromSubplotSpec(
            2, 2,
            subplot_spec=spec,
            height_ratios=[1, 4],
            width_ratios=[4, 1],
            hspace=0.02,
            wspace=0.02,
        )
        ax_top = fig_obj.add_subplot(inner[0, 0])
        ax_main = fig_obj.add_subplot(inner[1, 0], sharex=ax_top)
        ax_right = fig_obj.add_subplot(inner[1, 1], sharey=ax_main)
        ax_empty = fig_obj.add_subplot(inner[0, 1])
        ax_empty.axis("off")

        for ax in (ax_top, ax_main, ax_right):
            ax.set_facecolor(bg_color)

        ax_main.scatter(x_true, y_delta, s=18, alpha=0.35, color=base_color,
                        edgecolors="white", linewidths=0.2)

        xs, m, lo, hi = _kernel_smooth_with_ci(x_true, y_delta)
        ax_main.plot(xs, m, color=trend_color, linestyle="--", linewidth=1.8)
        ax_main.fill_between(xs, lo, hi, color=ci_color, alpha=0.35, linewidth=0)

        ax_main.axhline(0.0, color="#404040", linestyle="--", linewidth=1.0, alpha=0.9)

        q33, q67 = np.quantile(y_pred_raw, [1 / 3, 2 / 3])
        ax_main.axvline(q33, color=boundary_color, linestyle=":", linewidth=1.3)
        ax_main.axvline(q67, color=boundary_color, linestyle=":", linewidth=1.3)

        xmin, xmax = ax_main.get_xlim()
        ymin, ymax = ax_main.get_ylim()
        x_shift = 0.010 * (xmax - xmin)
        y_pos = ymax - 0.08 * (ymax - ymin)
        ax_main.text(q33 + x_shift, y_pos, "q33", ha="left", va="bottom", fontsize=9, color=boundary_color)
        ax_main.text(q67 + x_shift, y_pos, "q67", ha="left", va="bottom", fontsize=9, color=boundary_color)

        ax_main.set_xlabel(x_label)
        ax_main.set_ylabel(y_label)

        bins_x = max(18, min(32, int(np.sqrt(len(x_true)))))
        bins_y = max(18, min(32, int(np.sqrt(len(y_delta)))))

        ax_top.hist(x_true, bins=bins_x, color=base_color, edgecolor="#666666", alpha=0.45)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", left=False, labelleft=False)
        ax_top.spines["left"].set_visible(False)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)

        ax_right.hist(y_delta, bins=bins_y, orientation="horizontal", color=base_color,
                      edgecolor="#666666", alpha=0.45)
        ax_right.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.spines["bottom"].set_visible(False)

        mean_delta = float(np.mean(y_delta))
        std_delta = float(np.std(y_delta, ddof=1)) if len(y_delta) > 1 else 0.0
        ax_main.text(
            stat_pos[0], stat_pos[1],
            f"n={len(y_delta)}\nmean={mean_delta:.3f}\nstd={std_delta:.3f}",
            transform=ax_main.transAxes,
            ha=stat_ha, va=stat_va, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#b5b5b5", alpha=0.95),
        )

        legend_handles = [
            Line2D([0], [0], marker='o', color='none', markerfacecolor=base_color, markeredgecolor='white',
                   markeredgewidth=0.3, markersize=6, alpha=0.8, label='Samples'),
            Line2D([0], [0], color=trend_color, linestyle='--', linewidth=1.8, label='Smoothed trend'),
            Patch(facecolor=ci_color, edgecolor='none', alpha=0.35, label='95% interval'),
            Line2D([0], [0], color=boundary_color, linestyle=':', linewidth=1.3, label='Segment boundaries'),
        ]
        ax_main.legend(handles=legend_handles, loc=legend_loc, frameon=False, fontsize=9)

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "axes.facecolor": bg_color,
        "figure.facecolor": bg_color,
        "savefig.facecolor": bg_color,
        "savefig.edgecolor": bg_color,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig = plt.figure(figsize=figsize, constrained_layout=False, facecolor=bg_color)
        outer = GridSpec(2, 1, figure=fig, hspace=0.26)

        _draw_joint_block(
            fig,
            outer[0],
            t_true,
            t_delta,
            t_pred_raw,
            f"T true ({t_unit})",
            f"Correction Value({t_unit})",
            t_color,
            legend_loc="upper left",
            stat_pos=(0.98, 0.04),
            stat_ha="right",
            stat_va="bottom",
        )

        _draw_joint_block(
            fig,
            outer[1],
            p_true,
            p_delta,
            p_pred_raw,
            f"P true ({p_unit})",
            f"Correction Value({p_unit})",
            p_color,
            legend_loc="lower right",
            stat_pos=(0.76, 0.06),
            stat_ha="right",
            stat_va="bottom",
        )

        fig.suptitle(title, fontsize=15, y=0.985)
        fig.subplots_adjust(top=0.955, left=0.08, right=0.95, bottom=0.05)
        return fig


if __name__ == "__main__":
    print("=== 可视化示例 ===")

    # 模拟数据
    np.random.seed(42)
    n = 200
    y_T_true = np.random.uniform(800, 1200, n)
    y_T_pred = y_T_true + np.random.normal(0, 30, n)
    y_P_true = np.random.uniform(0.1, 20, n)
    y_P_pred = y_P_true + np.random.normal(0, 1.5, n)

    # 生成报告
    fig = plot_full_report(y_T_true, y_T_pred, y_P_true, y_P_pred, exp_name='示例实验')
    plt.show()
