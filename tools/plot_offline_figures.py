#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline plotting smoke test for available experiment artifacts.

This script only reads existing files under results/ and generates figures.
No model training is performed.
"""
import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.viz import (
    plot_pred_vs_true,
    plot_residuals,
    plot_fold_comparison,
    plot_experiment_summary,
    plot_full_report,
    plot_stepwise_rmse_comparison,
    plot_correction_effect,
    plot_residual_distribution_comparison,
    plot_feature_importance,
    save_figure,
)

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_any(fig_or_ax, filepath: str) -> None:
    if fig_or_ax is None:
        return
    fig = fig_or_ax if hasattr(fig_or_ax, "savefig") else getattr(fig_or_ax, "figure", None)
    if fig is None:
        print(f"skip: cannot save figure to {filepath}")
        return
    save_figure(fig, filepath)


def _load_predictions(results_dir: str, exp_id: str, target: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{exp_id}_{target}_predictions.parquet")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return pd.read_parquet(path)


def _load_fold_metrics(results_dir: str, exp_id: str, target: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{exp_id}_{target}_fold_metrics.csv")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return pd.read_csv(path)


def _prepare_correction_df(df: pd.DataFrame, target: str) -> pd.DataFrame:
    return df.rename(
        columns={
            "y_true": f"{target}_true",
            "y_pred_raw": f"{target}_pred_raw",
            "y_pred_corr": f"{target}_pred_corr",
        }
    )


def _load_metrics_summary(results_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, "metrics_summary.csv")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    df = pd.read_csv(path)
    if "exp_id" in df.columns and "exp_name" not in df.columns:
        df = df.rename(columns={"exp_id": "exp_name"})
    return df


def _load_model(results_dir: str, exp_id: str, target: str) -> Optional[Dict]:
    """加载保存的模型文件"""
    if not HAS_JOBLIB:
        print("skip: joblib not available for model loading")
        return None
    path = os.path.join(results_dir, "models", f"{exp_id}_{target}_model.joblib")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return joblib.load(path)


def _plot_importance(exp_id: str, results_dir: str, fig_dir: str, feature_names: Optional[List[str]] = None) -> None:
    """绘制特征重要性图（从保存的模型加载）"""
    for target in ['T', 'P']:
        model_data = _load_model(results_dir, exp_id, target)
        if model_data is None:
            continue
        
        model = model_data.get('model')
        model_module = model_data.get('model_module')

        if model is None:
            print(f"skip: model is None for {exp_id}_{target}")
            continue
        
        # 获取特征重要性（根据不同模型类型处理）
        importances = None

        # 方式1: 使用 model_module 的 get_feature_importance 方法
        if model_module is not None and hasattr(model_module, 'get_feature_importance'):
            try:
                importances = model_module.get_feature_importance(model)
            except Exception as e:
                print(f"note: model_module.get_feature_importance failed for {exp_id}_{target}: {e}")

        # 方式2: 直接从模型获取（sklearn 模型）
        if importances is None:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                # CatBoost 模型
                try:
                    importances = model.get_feature_importance()
                except Exception:
                    pass

        # 方式3: 对于 Stacking（字典类型模型），尝试从基模型获取
        if importances is None and isinstance(model, dict):
            base_models = model.get('base', [])
            if base_models:
                # 收集各基模型的特征重要性并平均
                imp_list = []
                for bm in base_models:
                    if hasattr(bm, 'feature_importances_'):
                        imp_list.append(bm.feature_importances_)
                    elif hasattr(bm, 'get_feature_importance'):
                        try:
                            imp_list.append(bm.get_feature_importance())
                        except Exception:
                            pass
                if imp_list:
                    importances = np.mean(imp_list, axis=0)

        if importances is None:
            print(f"skip: cannot extract feature importance for {exp_id}_{target}")
            continue
        
        # 尝试获取特征名称
        names = feature_names
        if names is None:
            # 方式1: 从保存的 config 中获取 feature_set，映射到特征名称
            config = model_data.get('config', {})
            feature_set = config.get('feature_set')
            if feature_set:
                # 定义特征集映射（与 main.py CONFIG 保持一致）
                FEATURE_SETS = {
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
                }
                names = FEATURE_SETS.get(feature_set)
        if names is None:
            # 方式2: 尝试从 data_state 推断
            state = model_data.get('data_state')
            if state is not None and hasattr(state, 'feature_names'):
                names = state.feature_names
        if names is None:
            names = [f"Feature_{i}" for i in range(len(importances))]
        
        try:
            fig = plot_feature_importance(importances, names, target=target)
            _save_any(fig, os.path.join(fig_dir, f"{exp_id}_{target}_importance.png"))
            print(f"图表已保存: {os.path.join(fig_dir, f'{exp_id}_{target}_importance.png')}")
        except Exception as e:
            print(f"skip: error plotting importance for {exp_id}_{target}: {e}")


def _plot_basic(exp_id: str, results_dir: str, fig_dir: str) -> None:
    df_T = _load_predictions(results_dir, exp_id, "T")
    df_P = _load_predictions(results_dir, exp_id, "P")
    if df_T is None or df_P is None:
        return

    # Pred vs true
    fig = plot_pred_vs_true(df_T["y_true"], df_T["y_pred_corr"], target_name="T", unit="C")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_pred_vs_true.png"))

    fig = plot_pred_vs_true(df_P["y_true"], df_P["y_pred_corr"], target_name="P", unit="kbar")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_pred_vs_true.png"))

    # Residuals
    fig = plot_residuals(df_T["y_true"], df_T["y_pred_corr"], target_name="T", unit="C")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_residuals.png"))

    fig = plot_residuals(df_P["y_true"], df_P["y_pred_corr"], target_name="P", unit="kbar")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_residuals.png"))

    # Full report (T+P)
    fig = plot_full_report(
        df_T["y_true"], df_T["y_pred_corr"],
        df_P["y_true"], df_P["y_pred_corr"],
        exp_name=exp_id,
    )
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_full_report.png"))

    # Correction effect (requires raw/corr columns)
    if all(col in df_T.columns for col in ["y_true", "y_pred_raw", "y_pred_corr"]):
        df_corr_T = _prepare_correction_df(df_T, "T")
        fig = plot_correction_effect(df_corr_T, exp_name=exp_id, target="T")
        _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_correction_effect.png"))
    else:
        print("skip: correction effect (T) missing columns")

    if all(col in df_P.columns for col in ["y_true", "y_pred_raw", "y_pred_corr"]):
        df_corr_P = _prepare_correction_df(df_P, "P")
        fig = plot_correction_effect(df_corr_P, exp_name=exp_id, target="P")
        _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_correction_effect.png"))
    else:
        print("skip: correction effect (P) missing columns")


def _plot_fold(exp_id: str, results_dir: str, fig_dir: str) -> None:
    df_T = _load_fold_metrics(results_dir, exp_id, "T")
    df_P = _load_fold_metrics(results_dir, exp_id, "P")
    if df_T is not None:
        df_T = df_T.rename(columns={"rmse": "T_rmse", "mae": "T_mae", "r2": "T_r2"})
        fig = plot_fold_comparison(df_T, target="T", metric="rmse")
        _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_fold_rmse.png"))
    if df_P is not None:
        df_P = df_P.rename(columns={"rmse": "P_rmse", "mae": "P_mae", "r2": "P_r2"})
        fig = plot_fold_comparison(df_P, target="P", metric="rmse")
        _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_fold_rmse.png"))


def _plot_summary(results_dir: str, fig_dir: str) -> None:
    df = _load_metrics_summary(results_dir)
    if df is None:
        return
    fig = plot_experiment_summary(df)
    _save_any(fig, os.path.join(fig_dir, "experiment_summary_heatmap.png"))


def _plot_stepwise(results_dir: str, fig_dir: str) -> None:
    # Map expected keys to existing experiments (Liquid set)
    step_map = {
        "exp1_baseline": "E01_ert_raw_none_liq",
        "exp2_aug_only": "E07_ert_augmented_none_liq",
        "exp3_corr_only": "E09_catboost_raw_segmented_liq",
        "exp4_aug_corr": "E11_catboost_balanced_segmented_liq",
        "exp5_stacking": "E12_stacking_balanced_segmented_liq",
    }

    def build_dict(target: str) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for key, exp_id in step_map.items():
            df = _load_fold_metrics(results_dir, exp_id, target)
            if df is None:
                continue
            df = df.rename(columns={"rmse": f"{target}_rmse"})
            out[key] = df
        return out

    results_T = build_dict("T")
    if results_T:
        fig = plot_stepwise_rmse_comparison(results_T, target="T")
        _save_any(fig, os.path.join(fig_dir, "stepwise_rmse_T.png"))

    results_P = build_dict("P")
    if results_P:
        fig = plot_stepwise_rmse_comparison(results_P, target="P")
        _save_any(fig, os.path.join(fig_dir, "stepwise_rmse_P.png"))


def _plot_residual_compare(results_dir: str, fig_dir: str) -> None:
    # Use E08 vs E12 for comparison, mapped to expected keys
    comp_map = {
        "exp4_aug_corr": "E08_catboost_augmented_none_liq",
        "exp5_stacking": "E12_stacking_balanced_segmented_liq",
    }

    def build_dict(target: str) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for key, exp_id in comp_map.items():
            df = _load_predictions(results_dir, exp_id, target)
            if df is None:
                continue
            df = df.rename(
                columns={
                    "y_true": f"{target}_true",
                    "y_pred_corr": f"{target}_pred_corr",
                }
            )
            out[key] = df
        return out

    results_T = build_dict("T")
    if len(results_T) >= 2:
        fig = plot_residual_distribution_comparison(results_T, target="T")
        _save_any(fig, os.path.join(fig_dir, "residual_compare_T.png"))
    else:
        print("skip: residual compare (T) missing data")

    results_P = build_dict("P")
    if len(results_P) >= 2:
        fig = plot_residual_distribution_comparison(results_P, target="P")
        _save_any(fig, os.path.join(fig_dir, "residual_compare_P.png"))
    else:
        print("skip: residual compare (P) missing data")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline plotting smoke test.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--exp-id", default="E08_catboost_augmented_none_liq")
    parser.add_argument("--fig-subdir", default=os.path.join("figures", "plot_smoke_test"))
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, args.fig_subdir)
    _ensure_dir(fig_dir)

    _plot_basic(args.exp_id, args.results_dir, fig_dir)
    _plot_fold(args.exp_id, args.results_dir, fig_dir)
    _plot_summary(args.results_dir, fig_dir)
    _plot_stepwise(args.results_dir, fig_dir)
    _plot_residual_compare(args.results_dir, fig_dir)
    _plot_importance(args.exp_id, args.results_dir, fig_dir)

    print(f"plots saved under {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
