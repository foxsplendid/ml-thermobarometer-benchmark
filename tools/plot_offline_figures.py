# -*- coding: utf-8 -*-
"""
Offline figure generation for paper figures.

Run without arguments to generate all paper figures (Fig. 1, 3-8).
Use --debug for additional diagnostic plots.
"""
import argparse
import logging
import os
import sys
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn.model_selection import StratifiedKFold

# Ensure repo root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config_dict
from src.protocol import _merge_sparse_bins
from src.logger import setup_logging, get_logger
from src.viz import (
    plot_pred_vs_true,
    plot_residuals,
    plot_correction_delta_scatter_tp,
    plot_full_report,
    plot_correction_effect,
    plot_residual_distribution_comparison,
    plot_feature_importance,
    plot_learning_curve,
    plot_stability_overview,
    plot_combined_shap_summary,
    save_figure,
    plot_pt_grid_cv_splits,
    plot_feature_set_comparison_boxplot,
    plot_parity_comparison,
)

logger = logging.getLogger(__name__)

DEFAULT_SHAP_MAX_SAMPLES = 300
DEFAULT_SHAP_BG_K = 50
DEFAULT_SHAP_FORCE = True


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


def _save_paper_figure(fig: plt.Figure, fig_dir: str, stem: str, dpi: int = 300) -> None:
    """Save a paper figure as both PNG and PDF, then close it."""
    for ext in ('png', 'pdf'):
        path = os.path.join(fig_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"saved: {path}")
    plt.close(fig)


def _load_predictions(results_dir: str, exp_id: str, target: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, f"{exp_id}_{target}_predictions.parquet")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return pd.read_parquet(path)


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


def _load_stability_metrics(results_dir: str, exp_id: str, target: str) -> Optional[pd.DataFrame]:
    path = os.path.join(results_dir, "stability", f"{exp_id}_{target}_test_metrics.csv")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return pd.read_csv(path)


def _load_learning_curve_summary(learning_curve_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(learning_curve_dir, "learning_curve_summary.csv")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return pd.read_csv(path)


def _load_model(results_dir: str, exp_id: str, target: str) -> Optional[Dict]:
    path = os.path.join(results_dir, "models", f"{exp_id}_{target}_model.joblib")
    if not os.path.exists(path):
        print(f"skip: missing {path}")
        return None
    return joblib.load(path)


def _resolve_feature_names(model_data: Dict[str, Any],
                           feature_names: Optional[List[str]] = None) -> Optional[List[str]]:
    names = feature_names

    if names is None:
        config = model_data.get('config', {})
        feature_set = config.get('feature_set')
        if feature_set:
            names = get_config_dict().get('feature_sets', {}).get(feature_set)

    if names is None:
        state = model_data.get('data_state')
        if state is not None and hasattr(state, 'feature_names'):
            names = state.feature_names

    return names


def _predict_with_module(model_module: Any, model: Any, X: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X)
    if model_module is not None and hasattr(model_module, 'predict'):
        return model_module.predict(model, X_arr)
    if hasattr(model, 'predict'):
        return model.predict(X_arr)
    raise ValueError("model does not provide a usable predict method")


def _is_tree_model(model: Any) -> bool:
    cls_name = model.__class__.__name__.lower()
    return (
        hasattr(model, 'feature_importances_')
        or ('catboost' in cls_name)
        or ('forest' in cls_name)
        or ('tree' in cls_name)
        or ('xgb' in cls_name)
        or ('lightgbm' in cls_name)
    )


def _normalize_shap_values(shap_values: Any) -> Any:
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return shap_values[0]
    return shap_values


def _compute_shap_values(model: Any,
                         predict_fn,
                         X_df: pd.DataFrame,
                         bg_k: int) -> Any:
    if _is_tree_model(model):
        explainer = shap.Explainer(model, X_df)
        shap_values = explainer(X_df)
    else:
        k = max(1, min(bg_k, max(1, X_df.shape[0] // 10)))
        background = shap.kmeans(X_df, k)
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_df)

    return _normalize_shap_values(shap_values)


def _load_shap_input_df(data_path: str,
                        data_encoding: str,
                        model_data: Dict[str, Any],
                        feature_names: List[str],
                        max_samples: int,
                        random_seed: int = 42) -> Optional[pd.DataFrame]:
    if not os.path.exists(data_path):
        print(f"skip: missing data file for SHAP: {data_path}")
        return None

    try:
        df = pd.read_csv(data_path, encoding=data_encoding)
    except Exception as e:
        print(f"skip: failed to read data file for SHAP: {e}")
        return None

    missing_cols = [c for c in feature_names if c not in df.columns]
    if missing_cols:
        print(f"skip: SHAP missing feature columns in data: {missing_cols[:5]}")
        return None

    X_raw = df[feature_names].values
    state = model_data.get('data_state')
    if state is not None and hasattr(state, 'scaler') and state.scaler is not None:
        try:
            X_raw = state.scaler.transform(X_raw)
        except Exception as e:
            print(f"skip: SHAP scaler transform failed: {e}")
            return None

    X_df = pd.DataFrame(X_raw, columns=feature_names)

    if max_samples > 0 and len(X_df) > max_samples:
        rng = np.random.RandomState(random_seed)
        idx = rng.choice(len(X_df), size=max_samples, replace=False)
        X_df = X_df.iloc[idx].reset_index(drop=True)

    return X_df


def _base_model_name(base_module: Any, fitted_model: Any, idx: int) -> str:
    if base_module is not None:
        name = base_module.__class__.__name__
        if name.endswith('Model'):
            name = name[:-5]
        return name
    return f"Base{idx + 1}_{fitted_model.__class__.__name__}"


def _plot_single_shap_for_model(model: Any,
                                model_module: Any,
                                X_df: pd.DataFrame,
                                save_path: str,
                                model_name: str,
                                max_display: int = 22,
                                bg_k: int = 50,
                                force: bool = False,
                                figsize: tuple = None,
                                font_size: int = 12) -> None:
    if (not force) and os.path.exists(save_path):
        print(f"SHAP exists, skip: {save_path}")
        return

    if figsize is None:
        fig_h = max(10.0 * X_df.shape[1] / 18 + 1.5, 5.0)
        figsize = (12, fig_h)

    def predict_fn(X):
        return _predict_with_module(model_module, model, X)

    shap_values = _compute_shap_values(model, predict_fn, X_df, bg_k=bg_k)
    fig = plot_combined_shap_summary(
        shap_values=shap_values,
        X=X_df,
        model_name=model_name,
        max_display=min(max_display, X_df.shape[1]),
        figsize=figsize,
        font_size=font_size,
        show_suptitle=False,
    )
    _save_any(fig, save_path)
    print(f"saved: {save_path}")


def _merge_two_images_horizontally(left_path: str, right_path: str,
                                   output_path: Optional[str] = None) -> Optional[plt.Figure]:
    import matplotlib.image as mpimg

    img_l = mpimg.imread(left_path)
    img_r = mpimg.imread(right_path)

    ratio_l = img_l.shape[1] / max(1, img_l.shape[0])
    ratio_r = img_r.shape[1] / max(1, img_r.shape[0])
    fig_h = 6.0
    fig_w = fig_h * (ratio_l + ratio_r)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), dpi=300)
    axes[0].imshow(img_l)
    axes[1].imshow(img_r)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout(pad=0.1)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def _merge_two_images_vertically(top_path: str, bottom_path: str,
                                 output_path: Optional[str] = None) -> Optional[plt.Figure]:
    import matplotlib.image as mpimg

    img_t = mpimg.imread(top_path)
    img_b = mpimg.imread(bottom_path)

    ratio_t = img_t.shape[1] / max(1, img_t.shape[0])
    ratio_b = img_b.shape[1] / max(1, img_b.shape[0])
    fig_w = max(8.0, 6.0 * max(ratio_t, ratio_b))
    fig_h = fig_w / max(1e-6, ratio_t) + fig_w / max(1e-6, ratio_b)

    fig, axes = plt.subplots(2, 1, figsize=(fig_w, fig_h), dpi=300)
    axes[0].imshow(img_t)
    axes[1].imshow(img_b)
    axes[0].axis('off')
    axes[1].axis('off')
    plt.tight_layout(pad=0.1)

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return None
    return fig


def _merge_shap_tp_images(exp_id: str, fig_dir: str, remove_single: bool = True) -> None:
    prefix_t = f"{exp_id}_T_SHAP_combined_"
    prefix_p = f"{exp_id}_P_SHAP_combined_"

    files = [f for f in os.listdir(fig_dir) if f.startswith(prefix_t) and f.endswith(".png")]
    for t_name in files:
        suffix = t_name[len(prefix_t):]
        p_name = f"{prefix_p}{suffix}"
        t_path = os.path.join(fig_dir, t_name)
        p_path = os.path.join(fig_dir, p_name)
        if not os.path.exists(p_path):
            continue

        tp_name = f"{exp_id}_TP_SHAP_combined_{suffix}"
        tp_path = os.path.join(fig_dir, tp_name)
        _merge_two_images_horizontally(t_path, p_path, output_path=tp_path)
        print(f"saved: {tp_path}")

        if remove_single:
            for path in (t_path, p_path):
                try:
                    os.remove(path)
                except OSError:
                    pass


def _merge_residual_analysis_quad(fig_dir: str) -> None:
    raw_t = os.path.join(fig_dir, "residual_analysis_raw_T.png")
    raw_p = os.path.join(fig_dir, "residual_analysis_raw_P.png")
    corr_t = os.path.join(fig_dir, "residual_analysis_corr_T.png")
    corr_p = os.path.join(fig_dir, "residual_analysis_corr_P.png")
    out_path = os.path.join(fig_dir, "residual_analysis_raw_corr_TP.png")

    if not all(os.path.exists(p) for p in (raw_t, raw_p, corr_t, corr_p)):
        print("skip: residual_analysis_raw_corr_TP merge missing source images")
        return

    import matplotlib.image as mpimg

    img_rt = mpimg.imread(raw_t)
    img_rp = mpimg.imread(raw_p)
    img_ct = mpimg.imread(corr_t)
    img_cp = mpimg.imread(corr_p)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes[0, 0].imshow(img_rt)
    axes[0, 1].imshow(img_rp)
    axes[1, 0].imshow(img_ct)
    axes[1, 1].imshow(img_cp)
    for ax in axes.ravel():
        ax.axis('off')

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {out_path}")


def _plot_stacking_shap(exp_id: str,
                        target: str,
                        model: Dict[str, Any],
                        model_module: Any,
                        X_df: pd.DataFrame,
                        fig_dir: str,
                        bg_k: int,
                        force: bool) -> None:
    base_models = model.get('base', [])
    meta_model = model.get('meta')
    meta_scaler = model.get('meta_scaler')

    if not base_models or meta_model is None:
        print(f"skip: invalid stacking model structure for {exp_id}_{target}")
        return

    base_modules = getattr(model_module, 'base_models', []) if model_module is not None else []
    base_preds = []
    base_pred_names = []

    for idx, fitted_base in enumerate(base_models):
        base_module = base_modules[idx] if idx < len(base_modules) else None
        base_name = _base_model_name(base_module, fitted_base, idx)
        save_path = os.path.join(fig_dir, f"{exp_id}_{target}_SHAP_combined_{base_name}.png")
        try:
            _plot_single_shap_for_model(
                model=fitted_base,
                model_module=base_module,
                X_df=X_df,
                save_path=save_path,
                model_name=f"{base_name} ({target})",
                max_display=22,
                bg_k=bg_k,
                force=force,
                figsize=(12, 10),
                font_size=12,
            )
            base_pred = _predict_with_module(base_module, fitted_base, X_df.values)
            base_preds.append(base_pred)
            base_pred_names.append(f"{base_name}_Pred")
        except Exception as e:
            print(f"skip: SHAP base model {base_name} failed: {e}")

    if not base_preds:
        print(f"skip: no base predictions available for stacking meta SHAP: {exp_id}_{target}")
        return

    meta_features = pd.DataFrame(np.column_stack(base_preds), columns=base_pred_names)

    meta_save_path = os.path.join(fig_dir, f"{exp_id}_{target}_SHAP_combined_Meta.png")
    if (not force) and os.path.exists(meta_save_path):
        print(f"SHAP exists, skip: {meta_save_path}")
        return

    def predict_meta(X):
        X_arr = np.asarray(X)
        if meta_scaler is not None:
            X_arr = meta_scaler.transform(X_arr)
        if model_module is not None and hasattr(model_module, 'meta_model') and hasattr(model_module.meta_model, 'predict'):
            return model_module.meta_model.predict(meta_model, X_arr)
        if hasattr(meta_model, 'predict'):
            return meta_model.predict(X_arr)
        raise ValueError("stacking meta model does not support prediction")

    try:
        shap_values_meta = _compute_shap_values(meta_model, predict_meta, meta_features, bg_k=min(20, bg_k))
        fig = plot_combined_shap_summary(
            shap_values=shap_values_meta,
            X=meta_features,
            model_name=f"Meta ({target})",
            max_display=min(6, meta_features.shape[1]),
            figsize=(12, 6),
            font_size=10,
            show_suptitle=False,
        )
        _save_any(fig, meta_save_path)
        print(f"saved: {meta_save_path}")
    except Exception as e:
        print(f"skip: SHAP meta model failed for {exp_id}_{target}: {e}")


def _plot_shap(exp_id: str,
               results_dir: str,
               fig_dir: str,
               data_path: str,
               data_encoding: str = 'latin-1',
               max_samples: int = 300,
               bg_k: int = 50,
               force: bool = False,
               remove_single: bool = True) -> None:
    for target in ['T', 'P']:
        model_data = _load_model(results_dir, exp_id, target)
        if model_data is None:
            continue

        model = model_data.get('model')
        model_module = model_data.get('model_module')
        if model is None:
            print(f"skip: model is None for SHAP {exp_id}_{target}")
            continue

        feature_names = _resolve_feature_names(model_data)
        if not feature_names:
            print(f"skip: cannot resolve feature names for SHAP {exp_id}_{target}")
            continue

        X_df = _load_shap_input_df(
            data_path=data_path,
            data_encoding=data_encoding,
            model_data=model_data,
            feature_names=feature_names,
            max_samples=max_samples,
        )
        if X_df is None or X_df.empty:
            print(f"skip: no SHAP input data for {exp_id}_{target}")
            continue

        if isinstance(model, dict) and 'base' in model and 'meta' in model:
            _plot_stacking_shap(
                exp_id=exp_id,
                target=target,
                model=model,
                model_module=model_module,
                X_df=X_df,
                fig_dir=fig_dir,
                bg_k=bg_k,
                force=force,
            )
        else:
            model_name = model.__class__.__name__
            save_path = os.path.join(fig_dir, f"{exp_id}_{target}_SHAP_combined_{model_name}.png")
            try:
                _plot_single_shap_for_model(
                    model=model,
                    model_module=model_module,
                    X_df=X_df,
                    save_path=save_path,
                    model_name=f"{model_name} ({target})",
                    max_display=22,
                    bg_k=bg_k,
                    force=force,
                    font_size=12,
                )
            except Exception as e:
                print(f"skip: SHAP failed for {exp_id}_{target}: {e}")

    _merge_shap_tp_images(exp_id=exp_id, fig_dir=fig_dir, remove_single=remove_single)


def _merge_shap_2x2(exp_id_liq: str,
                    exp_id_noliq: str,
                    fig_dir: str,
                    out_stem: str = "Fig.5_SHAP_Analysis",
                    remove_panels: bool = True) -> None:
    """Compose a 2x2 SHAP figure: rows = Liquid / NoLiquid; cols = T / P."""
    import matplotlib.image as mpimg

    def _find_panel(exp_id: str, target: str) -> Optional[str]:
        prefix = f"{exp_id}_{target}_SHAP_combined_"
        candidates = [
            os.path.join(fig_dir, f)
            for f in os.listdir(fig_dir)
            if f.startswith(prefix) and f.endswith(".png")
        ]
        if not candidates:
            print(f"skip: _merge_shap_2x2 missing panel for {exp_id}_{target}")
            return None
        return candidates[0]

    paths = {
        (exp_id_liq,   'T'): _find_panel(exp_id_liq,   'T'),
        (exp_id_liq,   'P'): _find_panel(exp_id_liq,   'P'),
        (exp_id_noliq, 'T'): _find_panel(exp_id_noliq, 'T'),
        (exp_id_noliq, 'P'): _find_panel(exp_id_noliq, 'P'),
    }
    if any(v is None for v in paths.values()):
        return

    imgs = [mpimg.imread(p) for p in paths.values()]

    def _to_float(arr: np.ndarray) -> np.ndarray:
        return arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr.astype(np.float32)

    imgs = [_to_float(im) for im in imgs]

    max_h = max(im.shape[0] for im in imgs)
    max_w = max(im.shape[1] for im in imgs)

    def _pad_white(im: np.ndarray) -> np.ndarray:
        h, w = im.shape[:2]
        c = im.shape[2] if im.ndim == 3 else 1
        canvas = np.ones((max_h, max_w, c), dtype=np.float32)
        ph, pw = (max_h - h) // 2, (max_w - w) // 2
        canvas[ph:ph + h, pw:pw + w] = im
        return canvas

    imgs = [_pad_white(im) for im in imgs]

    panel_w_in = max_w / 300.0
    panel_h_in = max_h / 300.0
    fig, axes = plt.subplots(2, 2, figsize=(panel_w_in * 2, panel_h_in * 2), dpi=300)
    for ax, im in zip(axes.ravel(), imgs):
        ax.imshow(im, interpolation='lanczos')
        ax.axis('off')
    plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002,
                        wspace=0.01, hspace=0.01)

    _save_paper_figure(fig, fig_dir, out_stem, dpi=300)

    if remove_panels:
        for p in paths.values():
            try:
                os.remove(p)
            except OSError:
                pass


def _plot_shap_figure5(exp_id_liq: str,
                       results_dir: str,
                       fig_dir: str,
                       data_path: str,
                       data_encoding: str = 'latin-1',
                       max_samples: int = 300,
                       bg_k: int = 50,
                       force: bool = False,
                       out_stem: str = "Fig.5_SHAP_Analysis") -> None:
    """Plot SHAP for both Liquid and NoLiquid configs and assemble a 2x2 Figure 5."""
    exp_id_noliq = exp_id_liq.replace('_liq', '_noliq')

    for exp_id in (exp_id_liq, exp_id_noliq):
        _plot_shap(
            exp_id=exp_id,
            results_dir=results_dir,
            fig_dir=fig_dir,
            data_path=data_path,
            data_encoding=data_encoding,
            max_samples=max_samples,
            bg_k=bg_k,
            force=force,
            remove_single=False,
        )

    _merge_shap_2x2(
        exp_id_liq=exp_id_liq,
        exp_id_noliq=exp_id_noliq,
        fig_dir=fig_dir,
        out_stem=out_stem,
        remove_panels=True,
    )


def _plot_correction_delta_scatter(results_dir: str, fig_dir: str, exp_id: str) -> Optional[plt.Figure]:
    """Plot TP correction delta scatter. Returns fig for caller to save."""
    df_t = _load_predictions(results_dir, exp_id, "T")
    df_p = _load_predictions(results_dir, exp_id, "P")
    if df_t is None or df_p is None:
        print("skip: correction delta scatter missing T/P prediction files")
        return None

    required_cols = {"y_true", "y_pred_raw", "y_pred_corr"}
    if not required_cols.issubset(df_t.columns):
        print(f"skip: correction delta scatter missing T columns for {exp_id}")
        return None
    if not required_cols.issubset(df_p.columns):
        print(f"skip: correction delta scatter missing P columns for {exp_id}")
        return None

    try:
        return plot_correction_delta_scatter_tp(
            t_true=df_t["y_true"].values,
            t_pred_raw=df_t["y_pred_raw"].values,
            t_pred_corr=df_t["y_pred_corr"].values,
            p_true=df_p["y_true"].values,
            p_pred_raw=df_p["y_pred_raw"].values,
            p_pred_corr=df_p["y_pred_corr"].values,
            title=None,
            t_unit=r"$^\circ$C",
            p_unit="kbar",
            bg_color="#ffffff",
        )
    except Exception as e:
        print(f"skip: correction delta scatter error: {e}")
        return None


def _plot_sampling_bias_triptych(data_path: str,
                                 fig_dir: str,
                                 data_encoding: str = "latin-1",
                                 grid_bins: int = 10,
                                 show_subplot_titles: bool = True,
                                 show_suptitle: bool = False) -> Optional[plt.Figure]:
    """Plot raw-data P-T distribution overview. Returns fig for caller to save."""
    if not os.path.exists(data_path):
        print(f"skip: missing data file for sampling bias figure: {data_path}")
        return None

    try:
        df = pd.read_csv(data_path, encoding=data_encoding)
    except Exception as e:
        print(f"skip: failed to read data file for sampling bias figure: {e}")
        return None

    if "T" not in df.columns or "P" not in df.columns:
        print("skip: sampling bias figure requires columns 'T' and 'P'")
        return None

    y_t = pd.to_numeric(df["T"], errors="coerce").to_numpy()
    y_p = pd.to_numeric(df["P"], errors="coerce").to_numpy()
    valid = np.isfinite(y_t) & np.isfinite(y_p)
    y_t = y_t[valid]
    y_p = y_p[valid]

    n_samples = y_t.size
    if n_samples == 0:
        print("skip: no valid T/P rows for sampling bias figure")
        return None

    ratio_p_le_2p5 = float(np.mean(y_p <= 2.5))
    ratio_p_ge_20 = float(np.mean(y_p >= 20.0))

    p_edges = np.linspace(float(y_p.min()), float(y_p.max()), grid_bins + 1)
    t_edges = np.linspace(float(y_t.min()), float(y_t.max()), grid_bins + 1)
    hist2d, _, _ = np.histogram2d(y_p, y_t, bins=[p_edges, t_edges])
    occupied_ratio = float(np.mean(hist2d > 0))
    sparse_ratio = float(np.mean(hist2d <= 3))
    top10_ratio = float(np.sort(hist2d.ravel())[::-1][:10].sum() / max(1.0, hist2d.sum()))
    _ = (occupied_ratio, sparse_ratio, top10_ratio)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=300)

    hb = axes[0].hexbin(
        y_t,
        y_p,
        gridsize=38,
        mincnt=1,
        cmap="viridis",
        norm=LogNorm()
    )
    axes[0].axhline(2.5, color="#ff7f0e", linestyle="--", linewidth=1.3, alpha=0.9, label="P = 2.5 kbar")
    axes[0].axhline(20.0, color="#d62728", linestyle="--", linewidth=1.3, alpha=0.9, label="P = 20 kbar")
    axes[0].set_xlabel("Temperature T (°C)")
    axes[0].set_ylabel("Pressure P (kbar)")
    if show_subplot_titles:
        axes[0].set_title("a) P-T Density (Hexbin, log scale)")
    axes[0].legend(loc="upper left", fontsize=8, frameon=True)
    cbar0 = fig.colorbar(hb, ax=axes[0], shrink=0.9)
    cbar0.set_label("Samples per hex (log)")

    n_bins_1d = max(20, min(60, int(np.sqrt(n_samples))))
    axes[1].hist(y_p, bins=n_bins_1d, color="#1f77b4", alpha=0.75, edgecolor="white")
    axes[1].set_xlabel("Pressure P (kbar)")
    axes[1].set_ylabel("Count")
    if show_subplot_titles:
        axes[1].set_title("b) Pressure Marginal Distribution")
    for thr, color in [(2.5, "#ff7f0e"), (10.0, "#2ca02c"), (20.0, "#d62728")]:
        axes[1].axvline(thr, color=color, linestyle="--", linewidth=1.2, alpha=0.9)

    cdf_ax = axes[1].twinx()
    p_sorted = np.sort(y_p)
    cdf = np.arange(1, n_samples + 1, dtype=float) / float(n_samples)
    cdf_ax.plot(p_sorted, cdf, color="black", linewidth=2.0, label="CDF")
    cdf_ax.set_ylim(0.0, 1.02)
    cdf_ax.set_ylabel("Cumulative fraction")
    axes[1].text(
        0.97, 0.10,
        f"P <= 2.5: {ratio_p_le_2p5 * 100:.1f}%\nP >= 20: {ratio_p_ge_20 * 100:.1f}%",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    if show_suptitle:
        fig.suptitle(f"Raw Experimental Data Sampling Bias Overview (n={n_samples})", fontsize=14, y=1.02)
    plt.tight_layout()

    return fig


def _plot_importance(exp_id: str, results_dir: str, fig_dir: str, feature_names: Optional[List[str]] = None) -> None:
    """Plot feature importance from saved model artifacts (debug use)."""
    for target in ['T', 'P']:
        model_data = _load_model(results_dir, exp_id, target)
        if model_data is None:
            continue

        model = model_data.get('model')
        model_module = model_data.get('model_module')

        if model is None:
            print(f"skip: model is None for {exp_id}_{target}")
            continue

        importances = None

        if model_module is not None and hasattr(model_module, 'get_feature_importance'):
            try:
                importances = model_module.get_feature_importance(model)
            except Exception as e:
                print(f"note: model_module.get_feature_importance failed for {exp_id}_{target}: {e}")

        if importances is None:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'get_feature_importance'):
                try:
                    importances = model.get_feature_importance()
                except Exception:
                    pass

        if importances is None and isinstance(model, dict):
            base_models = model.get('base', [])
            if base_models:
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

        names = feature_names
        if names is None:
            config = model_data.get('config', {})
            feature_set = config.get('feature_set')
            if feature_set:
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
            state = model_data.get('data_state')
            if state is not None and hasattr(state, 'feature_names'):
                names = state.feature_names
        if names is None:
            names = [f"Feature_{i}" for i in range(len(importances))]

        try:
            fig = plot_feature_importance(importances, names, target=target)
            _save_any(fig, os.path.join(fig_dir, f"{exp_id}_{target}_importance.png"))
            print(f"saved: {os.path.join(fig_dir, f'{exp_id}_{target}_importance.png')}")
        except Exception as e:
            print(f"skip: error plotting importance for {exp_id}_{target}: {e}")


def _plot_basic(exp_id: str, results_dir: str, fig_dir: str) -> None:
    df_T = _load_predictions(results_dir, exp_id, "T")
    df_P = _load_predictions(results_dir, exp_id, "P")
    if df_T is None or df_P is None:
        return

    fig = plot_pred_vs_true(df_T["y_true"], df_T["y_pred_corr"], target_name="T", unit="°C")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_pred_vs_true.png"))

    fig = plot_pred_vs_true(df_P["y_true"], df_P["y_pred_corr"], target_name="P", unit="kbar")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_pred_vs_true.png"))

    fig = plot_residuals(df_T["y_true"], df_T["y_pred_corr"], target_name="T", unit="°C")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_T_residuals.png"))

    fig = plot_residuals(df_P["y_true"], df_P["y_pred_corr"], target_name="P", unit="kbar")
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_P_residuals.png"))

    fig = plot_full_report(
        df_T["y_true"], df_T["y_pred_corr"],
        df_P["y_true"], df_P["y_pred_corr"],
        exp_name=exp_id,
    )
    _save_any(fig, os.path.join(fig_dir, f"{exp_id}_full_report.png"))

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


def _plot_residual_compare(results_dir: str, fig_dir: str) -> None:
    comp_map = {
        "exp4_aug_corr": "E07_ert_augmented_none_liq",
        "exp5_stacking": "E09_stacking_augmented_none_liq",
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


def _plot_feature_set_boxplot(results_dir: str, fig_dir: str) -> None:
    df = _load_metrics_summary(results_dir)
    if df is None:
        return

    for target in ['T', 'P']:
        try:
            fig = plot_feature_set_comparison_boxplot(df, target=target, metric='rmse')
            _save_any(fig, os.path.join(fig_dir, f"feature_set_boxplot_{target}.png"))
        except Exception as e:
            print(f"skip: feature_set_boxplot_{target} error: {e}")


def _plot_parity_compare(results_dir: str, fig_dir: str) -> Optional[plt.Figure]:
    """NoLiquid vs Liquid parity comparison. Returns vertically merged T/P fig."""
    temp_t = os.path.join(fig_dir, "_tmp_parity_T.png")
    temp_p = os.path.join(fig_dir, "_tmp_parity_P.png")

    for target, temp_path in [("T", temp_t), ("P", temp_p)]:
        df_noliq = _load_predictions(results_dir, "E07_ert_augmented_none_noliq", target)
        df_liq = _load_predictions(results_dir, "E07_ert_augmented_none_liq", target)

        if df_noliq is None or df_liq is None:
            print(f"skip: parity_compare_{target} missing data")
            return None

        preds_noliq = {'y_true': df_noliq['y_true'].values, 'y_pred': df_noliq['y_pred_corr'].values}
        preds_liq = {'y_true': df_liq['y_true'].values, 'y_pred': df_liq['y_pred_corr'].values}

        try:
            fig = plot_parity_comparison(
                preds_noliq,
                preds_liq,
                target=target,
                show_subplot_titles=True,
                show_suptitle=False,
            )
            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"skip: parity_compare_{target} error: {e}")
            return None

    if not (os.path.exists(temp_t) and os.path.exists(temp_p)):
        return None

    merged_fig = _merge_two_images_vertically(temp_t, temp_p)
    for p in (temp_t, temp_p):
        try:
            os.remove(p)
        except OSError:
            pass
    return merged_fig


def _plot_stability(stability_exp_id: Optional[str], results_dir: str) -> Optional[plt.Figure]:
    """Stability distribution figure. Returns fig for caller to save."""
    if not stability_exp_id:
        return None
    df_T = _load_stability_metrics(results_dir, stability_exp_id, "T")
    df_P = _load_stability_metrics(results_dir, stability_exp_id, "P")
    if df_T is None or df_P is None:
        return None
    try:
        return plot_stability_overview(df_T, df_P, metrics=("rmse", "mae", "mbe"))
    except Exception as e:
        print(f"skip: stability overview error: {e}")
        return None


def _plot_learning_curve(learning_curve_dir: str, fig_dir: str) -> Optional[plt.Figure]:
    """Learning-curve summary figure. Returns horizontally merged T+P fig."""
    summary_df = _load_learning_curve_summary(learning_curve_dir)
    if summary_df is None:
        return None

    temp_t = os.path.join(fig_dir, "_tmp_lc_T.png")
    temp_p = os.path.join(fig_dir, "_tmp_lc_P.png")

    for target, temp_path in [("T", temp_t), ("P", temp_p)]:
        try:
            fig = plot_learning_curve(summary_df, target=target)
            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            print(f"skip: learning curve {target} error: {e}")
            return None

    if not (os.path.exists(temp_t) and os.path.exists(temp_p)):
        return None

    merged_fig = _merge_two_images_horizontally(temp_t, temp_p)
    for p in (temp_t, temp_p):
        try:
            os.remove(p)
        except OSError:
            pass
    return merged_fig


def _plot_pt_grid_cv(data_path: str, fig_dir: str, random_seed: int = 42) -> Optional[plt.Figure]:
    """P-T grid stratified CV diagram. Returns fig for caller to save."""
    try:
        from src.splitters import compute_pt_edges, assign_pt_bins

        df = pd.read_csv(data_path, encoding='latin-1')
        y_T = df['T'].values
        y_P = df['P'].values

        pt_bins = compute_pt_edges(y_T, y_P)
        tp_labels = assign_pt_bins(y_T, y_P, pt_bins)

        n_splits = 10
        merged_labels = _merge_sparse_bins(tp_labels, min_samples_per_bin=n_splits)

        _, bin_counts = np.unique(merged_labels, return_counts=True)
        effective_n_splits = max(2, min(n_splits, bin_counts.min()))

        skf = StratifiedKFold(n_splits=effective_n_splits, shuffle=True, random_state=random_seed)
        fold_assignments = np.zeros(len(y_T), dtype=int)
        for fold_id, (_, val_idx) in enumerate(skf.split(y_T, merged_labels)):
            fold_assignments[val_idx] = fold_id

        fig = plot_pt_grid_cv_splits(
            y_T, y_P, tp_labels, fold_assignments,
            pt_bins.p_edges, pt_bins.t_edges,
            show_title=False,
        )
        if effective_n_splits < 10:
            print(f"note: P-T CV figure uses {effective_n_splits} folds after sparse-bin merge")
        return fig
    except Exception as e:
        print(f"skip: pt_grid_cv_splits error: {e}")
        return None


def _run_paper_mode(args, fig_dir: str, base_config: dict) -> None:
    """Generate all paper figures (Fig. 1, 3-8) with standardised PNG+PDF output."""

    # Fig. 1 — P-T distribution of raw dataset
    fig = _plot_sampling_bias_triptych(
        data_path=args.data_path,
        fig_dir=fig_dir,
        data_encoding=base_config.get("data_encoding", "latin-1"),
    )
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.1_PT_Distribution")

    # Fig. 3 — P-T grid-stratified CV
    fig = _plot_pt_grid_cv(args.data_path, fig_dir)
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.3_PT_Grid_CV")

    # Fig. 4 — NoLiquid vs Liquid parity comparison
    fig = _plot_parity_compare(args.results_dir, fig_dir)
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.4_Parity_Comparison")

    # Fig. 5 — SHAP 2x2 (disk-based pipeline, saves PNG+PDF internally)
    _plot_shap_figure5(
        exp_id_liq=args.exp_id,
        results_dir=args.results_dir,
        fig_dir=fig_dir,
        data_path=args.data_path,
        data_encoding=base_config.get("data_encoding", "latin-1"),
        max_samples=DEFAULT_SHAP_MAX_SAMPLES,
        bg_k=DEFAULT_SHAP_BG_K,
        force=DEFAULT_SHAP_FORCE,
        out_stem="Fig.5_SHAP_Analysis",
    )

    # Fig. 6 — Stability error distributions
    fig = _plot_stability(args.stability_exp_id, args.results_dir)
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.6_Stability")

    # Fig. 7 — Learning curves
    fig = _plot_learning_curve(args.learning_curve_dir, fig_dir)
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.7_Learning_Curves")

    # Fig. 8 — Correction delta scatter
    fig = _plot_correction_delta_scatter(args.results_dir, fig_dir, args.correction_delta_exp_id)
    if fig is not None:
        _save_paper_figure(fig, fig_dir, "Fig.8_Correction_Delta")

    print(f"\nPaper figures saved under {fig_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline figure generation for paper.")
    base_config = get_config_dict()
    parser.add_argument("--results-dir", default=base_config["output_dir"])
    parser.add_argument("--exp-id", default="E07_ert_augmented_none_liq")
    parser.add_argument("--fig-subdir", default="figures")
    parser.add_argument("--data-path", default=base_config["data_path"])
    parser.add_argument("--stability-exp-id", default="E07_stability_nj4")
    parser.add_argument("--learning-curve-dir",
                        default=os.path.join(base_config["output_dir"], "learning_curve"))
    parser.add_argument("--correction-delta-exp-id", default="E10_ert_augmented_segmented_liq")
    parser.add_argument("--debug", action="store_true",
                        help="Also generate diagnostic plots in addition to paper figures.")
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, args.fig_subdir)
    _ensure_dir(fig_dir)

    _run_paper_mode(args, fig_dir, base_config)

    if args.debug:
        _plot_basic(args.exp_id, args.results_dir, fig_dir)
        _plot_importance(args.exp_id, args.results_dir, fig_dir)
        _plot_residual_compare(args.results_dir, fig_dir)
        _plot_feature_set_boxplot(args.results_dir, fig_dir)
        print(f"Debug figures saved under {fig_dir}")

    return 0


if __name__ == "__main__":
    def _init_logging():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"plot_offline_figures_{timestamp}_{os.getpid()}.log"
        setup_logging(log_filename=log_filename)
        global logger
        logger = get_logger(__name__)

    _init_logging()
    try:
        exit_code = main()
    except Exception:
        logger.exception("offline plotting failed")
        raise
    sys.exit(exit_code)
