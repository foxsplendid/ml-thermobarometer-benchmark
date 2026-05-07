# -*- coding: utf-8 -*-
"""Build revised Figure 5 (2x2: Liquid/NoLiquid x T/P) for reviewer #2 comment 11.

Reproduces the existing single-panel SHAP rendering (E07b, Liquid) and adds the
NoLiquid counterpart (E07a) using the same explainer, sampling, and layout
parameters reported in stage 1. Final 2x2 panel is composed by image-level
mosaic (matching the existing repo convention in plot_offline_figures.py).
"""

import argparse
import os
import sys
import time
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config_dict
from src.viz import plot_combined_shap_summary

# Reuse the verified SHAP helpers from the offline plotter to guarantee
# byte-for-byte identical computation (same explainer, same scaler, same
# subsample seed, same predict path).
from tools.plot_offline_figures import (
    _load_model,
    _resolve_feature_names,
    _load_shap_input_df,
    _compute_shap_values,
    _predict_with_module,
)


EXP_LIQ = "E07_ert_augmented_none_liq"
EXP_NOLIQ = "E07_ert_augmented_none_noliq"
TARGETS = ("T", "P")
PANEL_TAG = {
    (EXP_LIQ, "T"): "a_Liquid_T",
    (EXP_LIQ, "P"): "b_Liquid_P",
    (EXP_NOLIQ, "T"): "c_NoLiquid_T",
    (EXP_NOLIQ, "P"): "d_NoLiquid_P",
}


def _shap_values_to_array(shap_values) -> np.ndarray:
    """Extract a (n_samples, n_features) ndarray from any SHAP return type."""
    if hasattr(shap_values, "values"):
        return np.asarray(shap_values.values)
    return np.asarray(shap_values)


def render_panel(
    exp_id: str,
    target: str,
    results_dir: str,
    cache_dir: str,
    panel_dir: str,
    data_path: str,
    data_encoding: str,
    max_samples: int = 300,
    bg_k: int = 50,
) -> Tuple[str, np.ndarray, pd.DataFrame]:
    """Render a single SHAP panel; cache values; return panel PNG path."""
    print(f"\n>> Panel {PANEL_TAG[(exp_id, target)]}: loading model {exp_id}_{target} ...")
    model_data = _load_model(results_dir, exp_id, target)
    if model_data is None:
        raise FileNotFoundError(f"Missing saved model for {exp_id}_{target}")

    model = model_data.get("model")
    model_module = model_data.get("model_module")
    if model is None:
        raise ValueError(f"model is None for {exp_id}_{target}")

    feature_names = _resolve_feature_names(model_data)
    if not feature_names:
        raise ValueError(f"cannot resolve feature names for {exp_id}_{target}")

    X_df = _load_shap_input_df(
        data_path=data_path,
        data_encoding=data_encoding,
        model_data=model_data,
        feature_names=list(feature_names),
        max_samples=max_samples,
        random_seed=42,
    )
    if X_df is None or X_df.empty:
        raise ValueError(f"empty SHAP input for {exp_id}_{target}")

    def predict_fn(X):
        return _predict_with_module(model_module, model, X)

    t0 = time.time()
    shap_values = _compute_shap_values(model, predict_fn, X_df, bg_k=bg_k)
    dt = time.time() - t0
    print(f"   SHAP computed in {dt:.1f}s; X shape = {X_df.shape}; "
          f"feature_names = {list(X_df.columns)}")

    sv_array = _shap_values_to_array(shap_values)

    np.save(
        os.path.join(cache_dir, f"{exp_id}_{target}_shap_values.npy"),
        sv_array,
    )
    X_df.to_parquet(
        os.path.join(cache_dir, f"{exp_id}_{target}_shap_X.parquet"),
        index=False,
    )

    # Scale figure height so every feature row gets the same vertical space.
    FIXED_WIDTH = 12.0
    n_features = X_df.shape[1]
    fig_h = max(10.0 * n_features / 18 + 1.5, 5.0)

    fig = plot_combined_shap_summary(
        shap_values=shap_values,
        X=X_df,
        model_name=f"ExtraTreesRegressor ({target})",
        max_display=min(22, n_features),
        figsize=(FIXED_WIDTH, fig_h),
        font_size=12,
        show_suptitle=False,
    )

    panel_path = os.path.join(panel_dir, f"Figure5_panel_{PANEL_TAG[(exp_id, target)]}.png")
    # bbox_inches='tight' keeps axes labels and colorbar; different feature
    # counts produce different panel heights — compose_2x2 pads to a shared canvas.
    fig.savefig(panel_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"   saved panel: {panel_path}")

    return panel_path, sv_array, X_df


def _pad_to_canvas(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Centre-pad img (H×W×C or H×W×4) to target_h × target_w with white."""
    h, w = img.shape[:2]
    # Determine channel count; use white (1.0 for float, 255 for uint8).
    if img.dtype == np.uint8:
        white_val = 255
    else:
        white_val = 1.0

    channels = img.shape[2] if img.ndim == 3 else 1
    canvas = np.full((target_h, target_w, channels), white_val, dtype=img.dtype)
    if img.ndim == 2:
        canvas = np.full((target_h, target_w), white_val, dtype=img.dtype)

    pad_top = (target_h - h) // 2
    pad_left = (target_w - w) // 2
    if img.ndim == 3:
        canvas[pad_top:pad_top + h, pad_left:pad_left + w, :] = img
    else:
        canvas[pad_top:pad_top + h, pad_left:pad_left + w] = img
    return canvas


def compose_2x2(panel_paths: dict, png_path: str, pdf_path: str) -> None:
    """Compose 4 PNG panels into a 2x2 mosaic; save as PNG and PDF.

    Panels have different aspect ratios because NoLiquid (9 features) is
    shorter than Liquid (18 features) after bbox_inches='tight' cropping.
    Each panel is centre-padded with white to the maximum bounding box so
    all cells in the 2x2 grid are visually equal-sized.
    """
    img_a = mpimg.imread(panel_paths[(EXP_LIQ, "T")])
    img_b = mpimg.imread(panel_paths[(EXP_LIQ, "P")])
    img_c = mpimg.imread(panel_paths[(EXP_NOLIQ, "T")])
    img_d = mpimg.imread(panel_paths[(EXP_NOLIQ, "P")])

    imgs = [img_a, img_b, img_c, img_d]

    # Unify dtype to float32 in [0,1] for consistent padding.
    def to_float(arr):
        if arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)

    imgs = [to_float(im) for im in imgs]

    # Find the maximum height and width across all panels.
    max_h = max(im.shape[0] for im in imgs)
    max_w = max(im.shape[1] for im in imgs)
    print(f"  panel sizes before padding: {[im.shape[:2] for im in imgs]}")
    print(f"  common canvas: {max_h} x {max_w}")

    imgs = [_pad_to_canvas(im, max_h, max_w) for im in imgs]

    # Figure size: each panel maps to (max_w/300) × (max_h/300) inches at 300 dpi.
    panel_w_in = max_w / 300.0
    panel_h_in = max_h / 300.0
    fig_w = panel_w_in * 2
    fig_h = panel_h_in * 2

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), dpi=300)
    for ax, img in zip(axes.ravel(), imgs):
        ax.imshow(img, interpolation="lanczos")
        ax.axis("off")

    plt.subplots_adjust(left=0.002, right=0.998, top=0.998, bottom=0.002,
                        wspace=0.01, hspace=0.01)

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"\nsaved: {png_path}")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print(f"saved: {pdf_path}  (note: panels embedded as raster within PDF)")
    plt.close(fig)


def report_top_features(cache_dir: str, k: int = 10) -> None:
    """Print mean|SHAP| ranking for each panel — used for caption / paragraph."""
    print("\n" + "=" * 70)
    print("Top features by mean|SHAP| (used for §3.2 paragraph)")
    print("=" * 70)
    for (exp_id, target), tag in PANEL_TAG.items():
        sv_path = os.path.join(cache_dir, f"{exp_id}_{target}_shap_values.npy")
        x_path = os.path.join(cache_dir, f"{exp_id}_{target}_shap_X.parquet")
        if not (os.path.exists(sv_path) and os.path.exists(x_path)):
            continue
        sv = np.load(sv_path)
        X_df = pd.read_parquet(x_path)
        mean_abs = np.abs(sv).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        print(f"\n[{tag}]  exp={exp_id}  target={target}  n_features={len(mean_abs)}")
        for rank, i in enumerate(order[:k], start=1):
            print(f"  {rank:>2}. {X_df.columns[i]:<14s}  mean|SHAP| = {mean_abs[i]:.4f}")


def main() -> int:
    base_config = get_config_dict()
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=base_config["output_dir"])
    parser.add_argument("--data-path", default=base_config["data_path"])
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--bg-k", type=int, default=50)
    args = parser.parse_args()

    fig_dir = os.path.join(args.results_dir, "figures")
    cache_dir = os.path.join(args.results_dir, "shap_cache")
    panel_dir = os.path.join(fig_dir, "_figure5_panels")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(panel_dir, exist_ok=True)

    panel_paths = {}
    print(f"Estimated total runtime: ~30-90s (4 panels x TreeExplainer on 300 samples)")
    t_start = time.time()

    for exp_id in (EXP_LIQ, EXP_NOLIQ):
        for target in TARGETS:
            panel_path, _, _ = render_panel(
                exp_id=exp_id,
                target=target,
                results_dir=args.results_dir,
                cache_dir=cache_dir,
                panel_dir=panel_dir,
                data_path=args.data_path,
                data_encoding=base_config.get("data_encoding", "latin-1"),
                max_samples=args.max_samples,
                bg_k=args.bg_k,
            )
            panel_paths[(exp_id, target)] = panel_path

    print(f"\n[timing] all 4 panels rendered in {time.time() - t_start:.1f}s")

    png_out = os.path.join(fig_dir, "Figure5_revised.png")
    pdf_out = os.path.join(fig_dir, "Figure5_revised.pdf")
    compose_2x2(panel_paths, png_out, pdf_out)

    report_top_features(cache_dir, k=10)

    return 0


if __name__ == "__main__":
    sys.exit(main())
