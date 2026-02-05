# -*- coding: utf-8 -*-
"""
机器学习温压计评估协议 - 指标计算模块
Metrics Module: RMSE, MAE, R², 斜率、截距、偏差统计等
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 基础指标函数
# ============================================================

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方根误差 (RMSE)"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算平均绝对误差 (MAE)"""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算决定系数 (R²)"""
    return r2_score(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """计算平均绝对百分比误差 (MAPE)"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算偏差（真实值 - 预测值的均值），正值表示模型低估"""
    return np.mean(y_true - y_pred)


def compute_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """
    计算 y_true ~ y_pred 的线性回归斜率和截距

    用于评估校正效果：理想情况下 slope≈1, intercept≈0

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值

    Returns
    -------
    slope : float
        回归斜率
    intercept : float
        回归截距
    """
    from sklearn.linear_model import LinearRegression
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)

    return lr.coef_[0], lr.intercept_


def compute_bias_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算偏差统计量

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值

    Returns
    -------
    stats : dict
        包含 bias_mean（偏差均值）和 resid_std（残差标准差）
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    residuals = y_true - y_pred

    return {
        'bias_mean': np.mean(residuals),
        'resid_std': np.std(residuals, ddof=1)  # 使用样本标准差
    }


# ============================================================
# 综合指标计算
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    prefix: str = '') -> Dict[str, float]:
    """
    计算所有评估指标（完整版，包含校正诊断指标）

    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    prefix : str, optional
        指标名称前缀（如 'T_' 或 'P_'）

    Returns
    -------
    metrics : dict
        包含 rmse, mae, r2, slope, intercept, bias_mean, resid_std 的字典
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # 基础指标
    slope, intercept = compute_slope_intercept(y_true, y_pred)
    bias_stats = compute_bias_stats(y_true, y_pred)

    return {
        f'{prefix}rmse': rmse(y_true, y_pred),
        f'{prefix}mae': mae(y_true, y_pred),
        f'{prefix}r2': r2(y_true, y_pred),
        f'{prefix}slope': slope,
        f'{prefix}intercept': intercept,
        f'{prefix}bias_mean': bias_stats['bias_mean'],
        f'{prefix}resid_std': bias_stats['resid_std']
    }


def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_pred_raw: Union[np.ndarray, None] = None,
                        n_bins: int = 5) -> Dict[str, float]:
    """
    计算完整指标集（含校正前后对比与分箱误差）。
    """
    from scipy.stats import linregress

    metrics: Dict[str, float] = {}

    # 基础指标
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    # MBE: 正值表示预测偏低（y_true > y_pred），负值表示预测偏高
    metrics['mbe'] = np.mean(y_true - y_pred)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # 回归诊断（y_pred 预测 y_true）
    if len(y_pred) > 2 and np.std(y_pred) > 1e-10:
        reg = linregress(y_pred, y_true)
        metrics['slope'] = reg.slope
        metrics['intercept'] = reg.intercept
    else:
        metrics['slope'] = np.nan
        metrics['intercept'] = np.nan

    # 残差标准差
    metrics['resid_std'] = np.std(y_true - y_pred)

    # 校正前指标
    if y_pred_raw is not None:
        metrics['rmse_raw'] = np.sqrt(mean_squared_error(y_true, y_pred_raw))
        metrics['mae_raw'] = mean_absolute_error(y_true, y_pred_raw)
        metrics['mbe_raw'] = np.mean(y_true - y_pred_raw)

        if len(y_pred_raw) > 2 and np.std(y_pred_raw) > 1e-10:
            reg_raw = linregress(y_pred_raw, y_true)
            metrics['slope_raw'] = reg_raw.slope
            metrics['intercept_raw'] = reg_raw.intercept

    # 分箱误差
    if n_bins > 0 and len(y_true) >= n_bins:
        try:
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(y_true, percentiles)

            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
                else:
                    mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])

                if np.sum(mask) >= 3:
                    metrics[f'mae_bin{i}'] = mean_absolute_error(y_true[mask], y_pred[mask])
                    metrics[f'mbe_bin{i}'] = np.mean(y_true[mask] - y_pred[mask])
        except Exception:
            pass

    return metrics


def compute_metrics_by_target(y_T_true: np.ndarray, y_T_pred: np.ndarray,
                               y_P_true: np.ndarray, y_P_pred: np.ndarray) -> Dict[str, float]:
    """
    分别计算 T 和 P 的评估指标

    Returns
    -------
    metrics : dict
        包含 T 和 P 的所有指标
    """
    metrics_T = compute_metrics(y_T_true, y_T_pred, prefix='T_')
    metrics_P = compute_metrics(y_P_true, y_P_pred, prefix='P_')
    return {**metrics_T, **metrics_P}


# ============================================================
# 折叠指标汇总
# ============================================================

def summarize_folds(fold_metrics: List[Dict[str, float]],
                    compute_ci: bool = True,
                    ci_level: float = 0.95) -> Dict[str, float]:
    """
    汇总各折的指标（计算均值和标准差）

    Parameters
    ----------
    fold_metrics : List[Dict[str, float]]
        各折的指标列表
    compute_ci : bool, default=True
        是否计算置信区间
    ci_level : float, default=0.95
        置信水平

    Returns
    -------
    summary : dict
        包含 {metric}_mean, {metric}_std 的字典
    """
    df = pd.DataFrame(fold_metrics)

    # 排除非数值列
    exclude_cols = ['fold_id', 'exp_name']
    numeric_cols = [col for col in df.columns if col not in exclude_cols]

    summary = {}
    for col in numeric_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            summary[f'{col}_mean'] = np.nan
            summary[f'{col}_std'] = np.nan
            if compute_ci:
                summary[f'{col}_ci_lower'] = np.nan
                summary[f'{col}_ci_upper'] = np.nan
            continue

        summary[f'{col}_mean'] = np.mean(values)
        summary[f'{col}_std'] = np.std(values, ddof=1) if len(values) > 1 else np.nan

        if compute_ci:
            if len(values) > 2:
                from scipy import stats
                se = stats.sem(values, ddof=1)
                ci = stats.t.interval(ci_level, len(values) - 1, loc=np.mean(values), scale=se)
                summary[f'{col}_ci_lower'] = ci[0]
                summary[f'{col}_ci_upper'] = ci[1]
            else:
                summary[f'{col}_ci_lower'] = np.nan
                summary[f'{col}_ci_upper'] = np.nan

    return summary


def format_metrics_table(fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
    """
    格式化指标表格（用于显示）

    Returns
    -------
    df : pd.DataFrame
        格式化后的指标表格
    """
    df = pd.DataFrame(fold_metrics)

    # 重新排列列顺序
    cols_order = ['fold_id']
    for target in ['T', 'P']:
        for metric in ['rmse', 'mae', 'r2']:
            col = f'{target}_{metric}'
            if col in df.columns:
                cols_order.append(col)

    available_cols = [c for c in cols_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in cols_order]

    return df[available_cols + other_cols]


def print_summary(summary: Dict[str, float], exp_name: str = "") -> None:
    """打印指标汇总"""
    print("\n" + "=" * 60)
    if exp_name:
        print(f"实验: {exp_name}")
    print("-" * 60)
    if "T_rmse_mean" in summary:
        print("目标 T:")
        print(f"  RMSE: {summary['T_rmse_mean']:.2f} ± {summary['T_rmse_std']:.2f}")
        print(f"  MAE:  {summary['T_mae_mean']:.2f} ± {summary['T_mae_std']:.2f}")
        print(f"  R²:   {summary['T_r2_mean']:.4f} ± {summary['T_r2_std']:.4f}")
    if "P_rmse_mean" in summary:
        print("目标 P:")
        print(f"  RMSE: {summary['P_rmse_mean']:.3f} ± {summary['P_rmse_std']:.3f}")
        print(f"  MAE:  {summary['P_mae_mean']:.3f} ± {summary['P_mae_std']:.3f}")
        print(f"  R²:   {summary['P_r2_mean']:.4f} ± {summary['P_r2_std']:.4f}")
    print("=" * 60)


def compare_experiments(results_list: List[Dict[str, float]]) -> pd.DataFrame:
    """
    比较多个实验的结果

    Parameters
    ----------
    results_list : List[Dict]
        各实验的汇总结果列表

    Returns
    -------
    comparison_df : pd.DataFrame
        实验对比表格
    """
    df = pd.DataFrame(results_list)

    # 保留关键列
    key_cols = ['exp_name']
    metric_cols = [c for c in df.columns if any(
        m in c for m in ['rmse', 'mae', 'r2']
    ) and 'mean' in c]

    available_cols = [c for c in key_cols + metric_cols if c in df.columns]

    return df[available_cols]

