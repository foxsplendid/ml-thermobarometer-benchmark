# -*- coding: utf-8 -*-
"""Metric helpers for regression evaluation and fold summarization."""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute root-mean-square error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error."""
    return mean_absolute_error(y_true, y_pred)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination."""
    return r2_score(y_true, y_pred)


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Compute mean absolute percentage error in percent."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute signed bias (y_true - y_pred)."""
    return np.mean(y_true - y_pred)


def compute_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Fit y_true = slope * y_pred + intercept and return parameters."""
    from sklearn.linear_model import LinearRegression

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    lr = LinearRegression()
    lr.fit(y_pred.reshape(-1, 1), y_true)
    return lr.coef_[0], lr.intercept_


def compute_bias_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute residual mean and standard deviation."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    return {
        "bias_mean": np.mean(residuals),
        "resid_std": np.std(residuals, ddof=1),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """Compute core regression metrics with an optional key prefix."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    slope, intercept = compute_slope_intercept(y_true, y_pred)
    bias_stats = compute_bias_stats(y_true, y_pred)

    return {
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}mae": mae(y_true, y_pred),
        f"{prefix}r2": r2(y_true, y_pred),
        f"{prefix}slope": slope,
        f"{prefix}intercept": intercept,
        f"{prefix}bias_mean": bias_stats["bias_mean"],
        f"{prefix}resid_std": bias_stats["resid_std"],
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_raw: Union[np.ndarray, None] = None,
    n_bins: int = 5,
) -> Dict[str, float]:
    """Compute benchmark metrics used across experiments."""
    from scipy.stats import linregress

    metrics: Dict[str, float] = {}

    metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics["mae"] = mean_absolute_error(y_true, y_pred)
    metrics["mbe"] = np.mean(y_true - y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if len(y_pred) > 2 and np.std(y_pred) > 1e-10:
        reg = linregress(y_pred, y_true)
        metrics["slope"] = reg.slope
        metrics["intercept"] = reg.intercept
    else:
        metrics["slope"] = np.nan
        metrics["intercept"] = np.nan

    metrics["resid_std"] = np.std(y_true - y_pred)

    if y_pred_raw is not None:
        metrics["rmse_raw"] = np.sqrt(mean_squared_error(y_true, y_pred_raw))
        metrics["mae_raw"] = mean_absolute_error(y_true, y_pred_raw)
        metrics["mbe_raw"] = np.mean(y_true - y_pred_raw)

        if len(y_pred_raw) > 2 and np.std(y_pred_raw) > 1e-10:
            reg_raw = linregress(y_pred_raw, y_true)
            metrics["slope_raw"] = reg_raw.slope
            metrics["intercept_raw"] = reg_raw.intercept

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
                    metrics[f"mae_bin{i}"] = mean_absolute_error(y_true[mask], y_pred[mask])
                    metrics[f"mbe_bin{i}"] = np.mean(y_true[mask] - y_pred[mask])
        except Exception:
            pass

    return metrics


def summarize_folds(
    fold_metrics: List[Dict[str, float]],
    compute_ci: bool = True,
    ci_level: float = 0.95,
) -> Dict[str, float]:
    """Aggregate per-fold metrics into mean/std and optional confidence intervals."""
    df = pd.DataFrame(fold_metrics)

    exclude_cols = ["fold_id", "exp_name"]
    numeric_cols = [col for col in df.columns if col not in exclude_cols]

    summary: Dict[str, float] = {}
    for col in numeric_cols:
        values = df[col].dropna().values
        if len(values) == 0:
            summary[f"{col}_mean"] = np.nan
            summary[f"{col}_std"] = np.nan
            if compute_ci:
                summary[f"{col}_ci_lower"] = np.nan
                summary[f"{col}_ci_upper"] = np.nan
            continue

        summary[f"{col}_mean"] = np.mean(values)
        summary[f"{col}_std"] = np.std(values, ddof=1) if len(values) > 1 else np.nan

        if compute_ci:
            if len(values) > 2:
                from scipy import stats

                se = stats.sem(values, ddof=1)
                ci = stats.t.interval(ci_level, len(values) - 1, loc=np.mean(values), scale=se)
                summary[f"{col}_ci_lower"] = ci[0]
                summary[f"{col}_ci_upper"] = ci[1]
            else:
                summary[f"{col}_ci_lower"] = np.nan
                summary[f"{col}_ci_upper"] = np.nan

    return summary
