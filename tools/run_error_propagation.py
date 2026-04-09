# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config_dict
from main import load_data, prepare_splits
from src.data_modules import get_data_module
from src.model_modules import get_model_module
from src.correction_modules import get_correction_module
from src.experiment_params import build_data_params, build_model_params
from src.protocol import (
    ExperimentConfig,
    Pipeline,
    _derive_target_seed,
)
from src.metrics import compute_all_metrics
from src.uncertainty_modules import MCUncertaintyEstimator
from src.logger import setup_logging, get_logger

logger = logging.getLogger(__name__)

BASE_CONFIG = get_config_dict()

OUTPUT_SUBDIR = "error_propagation"

def _load_main_experiment_test_rmse(output_dir: str, exp_id: str, target: str) -> Optional[float]:
    """_load_main_experiment_test_rmse function."""
    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    if not os.path.exists(summary_path):
        logger.warning(f"Main-experiment metrics_summary.csv does not exist: {summary_path}")
        return None

    try:
        df = pd.read_csv(summary_path)
        if "exp_id" not in df.columns:
            logger.warning(f"metrics_summary.csv is missing exp_id column: {summary_path}")
            return None

        row = df[df["exp_id"] == exp_id]
        if row.empty:
            logger.warning(f"exp_id={exp_id} not found in metrics_summary.csv")
            return None

        col = f"{target}_test_rmse"
        if col not in row.columns:
            logger.warning(f"metrics_summary.csv is missing column {col}")
            return None

        test_rmse = float(row.iloc[0][col])
        logger.info(f"[{target}] main-experiment test_RMSE: {test_rmse:.2f} (from {exp_id})")
        return test_rmse
    except Exception as e:
        logger.warning(f"Failed to read main-experiment test_RMSE: {e}")
        return None


def _build_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    """_build_config function."""
    model_params = build_model_params(BASE_CONFIG, model_module, random_seed)
    data_params = build_data_params(BASE_CONFIG, data_module, feature_set, random_seed)

    return ExperimentConfig(
        exp_id=exp_id,
        data_module_name=data_module,
        model_module_name=model_module,
        corr_module_name=corr_module,
        feature_set=feature_set,
        data_params=data_params,
        model_params=model_params,
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

def _parse_feature_names(raw_value: Optional[str]) -> Optional[List[str]]:
    if raw_value is None:
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    if raw_value.startswith("@"):
        path = raw_value[1:]
        with open(path, "r", encoding="utf-8") as f:
            raw_value = f.read().strip()
    if raw_value.startswith("["):
        value = json.loads(raw_value)
        if isinstance(value, list):
            return [str(v) for v in value]
    return [item.strip() for item in raw_value.split(",") if item.strip()]

def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {
            "min": float("nan"),
            "p5": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
        }
    return {
        "min": float(np.min(values)),
        "p5": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _run_error_propagation(
    config: ExperimentConfig,
    X_train: np.ndarray,
    y_T_train: np.ndarray,
    y_P_train: np.ndarray,
    X_test: np.ndarray,
    y_T_test: np.ndarray,
    y_P_test: np.ndarray,
    output_dir: str,
    random_seed: int,
    n_mc: int,
    mc_sample_size: int,
    mc_seed: int,
    feature_names_raw: Optional[str],
) -> None:
    """_run_error_propagation function."""
    rng = np.random.default_rng(mc_seed)

    if mc_sample_size <= 0:
        sample_size = len(X_test)
        sample_idx = np.arange(len(X_test))
        logger.info(f"Using full test set: {sample_size} samples")
    else:
        sample_size = min(mc_sample_size, len(X_test))
        sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
        logger.info(f"Randomly sampled test set: {sample_size} / {len(X_test)} samples")

    X_mc = X_test[sample_idx]
    y_T_mc = y_T_test[sample_idx]
    y_P_mc = y_P_test[sample_idx]

    ep_dir = os.path.join(output_dir, OUTPUT_SUBDIR)
    _ensure_dir(ep_dir)

    from config import DataConfig
    data_config = DataConfig()
    feature_names = _parse_feature_names(feature_names_raw)
    if feature_names is None:
        if X_test.shape[1] == 18:
            feature_names = data_config.feature_sets['Liquid']
        elif X_test.shape[1] == 9:
            feature_names = data_config.feature_sets['NoLiquid']
        else:
            raise ValueError(
                f"Cannot infer feature_names from n_features={X_test.shape[1]}; "
                f"please provide custom names via --feature-names"
            )
    if len(feature_names) != X_test.shape[1]:
        raise ValueError(f"feature_names length must match feature dimension, len={len(feature_names)} n_features={X_test.shape[1]}")

    from src.perturbation import get_rel_err_vector
    rel_err_vec = get_rel_err_vector(feature_names, strict=True)

    meta = {
        "exp_id": config.exp_id,
        "description": "Analysis error propagation experiment - evaluates EPMA analysis error amplification through fixed model",
        "design_notes": {
            "model_fixed": "Model fitted on full training data, no CV or retraining",
            "perturbation_target": "All input features (EPMA oxide wt% columns)",
            "rel_err_method": "Per-oxide mapping (Ágreda-López et al. 2024), not threshold-based",
            "no_clip": "Negative perturbations allowed to preserve distribution shape",
            "no_closure": "No sum normalization, consistent with training data preprocessing",
        },
        "metric_notes": {
            "analysis_std": "Reflects prediction dispersion caused only by input perturbation",
            "total_error": "total_* is baseline (unperturbed) prediction error and includes model error",
            "ratio_note": "analysis_contribution_ratio is an approximate contribution ratio, not a strict error decomposition",
        },
        "corr_module": config.corr_module_name,
        "n_mc": n_mc,
        "mc_sample_size": int(sample_size),
        "mc_seed": int(mc_seed),
        "test_indices_sampled": sample_idx.tolist(),
        "test_sample_stats": {
            "n_test_total": int(len(X_test)),
            "n_test_sampled": int(sample_size),
            "sample_fraction": float(sample_size / len(X_test)) if len(X_test) > 0 else np.nan,
            "y_T": _summary_stats(y_T_mc),
            "y_P": _summary_stats(y_P_mc),
        },
        "epma_error_spec": {
            "type": "per_oxide_mapping",
            "description": "Relative error mapped by oxide column name (3% for major, 8% for trace)",
            "feature_names": feature_names,
            "rel_err_vec": rel_err_vec.tolist(),
        },
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    _save_json(os.path.join(ep_dir, f"{config.exp_id}_ep_meta.json"), meta)

    def run_target(tag: str, y_true_mc: np.ndarray, y_train: np.ndarray) -> None:
        target_seed = _derive_target_seed(random_seed, tag)
        mc_seed_target = _derive_target_seed(mc_seed, tag)

        data_params = dict(config.data_params)
        model_params = dict(config.model_params)
        data_params["random_seed"] = target_seed
        model_params["random_seed"] = target_seed

        data_mod = get_data_module(config.data_module_name, **data_params)
        model_mod = get_model_module(config.model_module_name, **model_params)
        corr_mod = get_correction_module(config.corr_module_name)
        pipeline = Pipeline(data_mod, model_mod, corr_mod)
        pipeline.fit(X_train, y_train)

        if config.corr_module_name != "none":
            y_pred_train_raw = pipeline.predict_from_raw_input(X_train, apply_correction=False)
            corr_model = corr_mod.fit(y_train, y_pred_train_raw)
            pipeline.set_correction(corr_mod, corr_model)
        else:
            pipeline.set_correction(corr_mod, None)

        y_pred_base_raw = pipeline.predict_from_raw_input(X_mc, apply_correction=False)
        y_pred_base = pipeline.predict_from_raw_input(X_mc, apply_correction=True)

        mc = MCUncertaintyEstimator(
            n_mc=n_mc,
            feature_names=feature_names,
            random_seed=mc_seed_target,
        )
        dist = mc.predict_distribution(pipeline, X_mc)

        samples = dist["samples"]
        median = dist.get("median", np.median(samples, axis=0))
        mad = np.median(np.abs(samples - median), axis=0)
        analysis_2mad = 2.0 * mad
        interval_68 = dist["p84"] - dist["p16"]
        interval_90 = dist["p95"] - dist["p5"]

        metrics = compute_all_metrics(y_true_mc, y_pred_base, y_pred_base_raw)
        analysis_std_mean = float(np.mean(dist["std"]))
        rmse_val = float(metrics.get("rmse", np.nan))

        summary = {k: float(v) for k, v in metrics.items()}
        summary.update({
            "exp_id": config.exp_id,
            "target": tag,
            "n_samples": int(sample_size),
            "n_mc": int(n_mc),
            "analysis_std_mean": analysis_std_mean,
            "analysis_std_median": float(np.median(dist["std"])),
            "analysis_2mad_mean": float(np.mean(analysis_2mad)),
            "analysis_interval_68_mean": float(np.mean(interval_68)),
            "analysis_interval_90_mean": float(np.mean(interval_90)),
            "analysis_contribution_ratio": analysis_std_mean / rmse_val if rmse_val > 0 else np.nan,
        })

        main_exp_rmse = _load_main_experiment_test_rmse(output_dir, config.exp_id, tag)
        if main_exp_rmse is not None and main_exp_rmse > 0:
            summary["main_test_rmse"] = float(main_exp_rmse)
            summary["propagated_vs_test_ratio"] = analysis_std_mean / main_exp_rmse

        samples_df = pd.DataFrame({
            "test_index": sample_idx,
            "y_true": y_true_mc,
            "y_pred_base": y_pred_base,
            "pred_mean": dist["mean"],
            "pred_std": dist["std"],
            "pred_median": median,
            "p16": dist["p16"],
            "p84": dist["p84"],
            "p5": dist["p5"],
            "p95": dist["p95"],
            "analysis_interval_68": interval_68,
            "analysis_interval_90": interval_90,
            "analysis_2mad": analysis_2mad,
            "abs_error_base": np.abs(y_true_mc - y_pred_base),
        })

        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(ep_dir, f"{config.exp_id}_ep_{tag}_summary.csv")
        samples_path = os.path.join(ep_dir, f"{config.exp_id}_ep_{tag}_samples.csv")
        summary_df.to_csv(summary_path, index=False)
        samples_df.to_csv(samples_path, index=False)

        cv_info = f", main_test_rmse={summary['main_test_rmse']:.2f}" if summary.get("main_test_rmse") else ""
        ratio_info = f", propagated/test={summary['propagated_vs_test_ratio']:.1%}" if summary.get("propagated_vs_test_ratio") else ""
        logger.info(f"[{tag}] done: analysis_std_mean={summary['analysis_std_mean']:.2f}, "
                    f"analysis_2mad_mean={summary['analysis_2mad_mean']:.2f}{cv_info}{ratio_info}")

    run_target("T", y_T_mc, y_T_train)
    run_target("P", y_P_mc, y_P_train)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analysis error propagation tool - evaluate EPMA error amplification through fixed model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run error propagation with default settings (ERT, augmented, Liquid)
  python tools/run_error_propagation.py --model-module ert

  # Run with a different model and feature set
  python tools/run_error_propagation.py --model-module catboost --feature-set NoLiquid

  # Use custom exp_id
  python tools/run_error_propagation.py --exp-id my_custom_exp --model-module stacking

Output:
  results/error_propagation/
  - {exp_id}_ep_meta.json       # Experiment metadata
  - {exp_id}_ep_T_summary.csv   # Temperature summary statistics
  - {exp_id}_ep_T_samples.csv   # Per-sample T predictions and uncertainty
  - {exp_id}_ep_P_summary.csv   # Pressure summary statistics
  - {exp_id}_ep_P_samples.csv   # Per-sample P predictions and uncertainty
"""
    )
    parser.add_argument("--exp-id", default=None,
                        help="experiment id (default: auto-generate based on config)")
    parser.add_argument("--data-module", default="augmented",
                        choices=["raw", "balanced", "augmented"],
                        help="data module (default: augmented)")
    parser.add_argument("--model-module", default="ert",
                        choices=["ert", "extratrees", "catboost", "rf", "randomforest", "stacking"],
                        help="model module (default: ert)")
    parser.add_argument("--corr-module", default="none",
                        choices=["none", "segmented"],
                        help="correction module (default: none)")
    parser.add_argument("--feature-set", default="Liquid",
                        choices=["NoLiquid", "Liquid"],
                        help="feature set (default: Liquid)")
    parser.add_argument("--feature-names", default=None,
                        help="custom feature names (comma-separated / JSON list / @file)")

    parser.add_argument("--data-path", default=BASE_CONFIG["data_path"])
    parser.add_argument("--output-dir", default=BASE_CONFIG["output_dir"])
    parser.add_argument("--random-seed", type=int, default=BASE_CONFIG["random_seed"])

    parser.add_argument("--n-mc", type=int, default=1000,
                        help="MC samples per prediction (default: 1000)")
    parser.add_argument("--mc-sample-size", type=int, default=-1,
                        help="Test sample size for MC (-1 = use all test samples)")
    parser.add_argument("--mc-seed", type=int, default=42,
                        help="Random seed for MC sampling")

    args = parser.parse_args()

    # E01-E03: raw + ert/catboost/stacking + none
    # E04-E06: balanced + ert/catboost/stacking + none
    # E07-E09: augmented + ert/catboost/stacking + none
    # E10-E12: augmented + ert/catboost/stacking + segmented
    if args.exp_id is None:
        data_module = args.data_module.lower()
        model_module = args.model_module.lower()

        model_offset = {'ert': 0, 'extratrees': 0, 'catboost': 1, 'rf': 0, 'randomforest': 0, 'stacking': 2}
        offset = model_offset.get(model_module, 0)

        data_base_none = {'raw': 1, 'balanced': 4, 'augmented': 7}
        if args.corr_module.lower() == "segmented" and data_module == "augmented":
            base_num = 10
        else:
            base_num = data_base_none.get(data_module, 7)

        exp_num = base_num + offset
        suffix = 'noliq' if args.feature_set == 'NoLiquid' else 'liq'
        corr_tag = args.corr_module.lower()
        exp_id = f"E{exp_num:02d}_{model_module}_{data_module}_{corr_tag}_{suffix}"
    else:
        exp_id = args.exp_id

    config = _build_config(
        exp_id=exp_id,
        data_module=args.data_module,
        model_module=args.model_module,
        corr_module=args.corr_module,
        feature_set=args.feature_set,
        random_seed=args.random_seed,
    )

    print(f"="*60)
    print(f"Analysis Error Propagation Experiment")
    print(f"="*60)
    print(f"Experiment ID: {config.exp_id}")
    print(f"  data_module: {config.data_module_name}")
    print(f"  model_module: {config.model_module_name}")
    print(f"  corr_module: {config.corr_module_name}")
    print(f"  feature_set: {config.feature_set}")
    print(f"  n_mc: {args.n_mc}")
    print(f"  mc_seed: {args.mc_seed}")
    print(f"="*60)

    load_config = BASE_CONFIG.copy()
    load_config["data_path"] = args.data_path
    load_config["output_dir"] = args.output_dir

    X, y_T, y_P = load_data(load_config, feature_set=args.feature_set)
    split = prepare_splits(X, y_T, y_P, {"random_seed": args.random_seed})

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]

    X_train = X[train_idx]
    y_T_train = y_T[train_idx]
    y_P_train = y_P[train_idx]

    X_test = X[test_idx]
    y_T_test = y_T[test_idx]
    y_P_test = y_P[test_idx]

    _ensure_dir(args.output_dir)

    start = time.time()
    _run_error_propagation(
        config,
        X_train, y_T_train, y_P_train,
        X_test, y_T_test, y_P_test,
        args.output_dir,
        args.random_seed,
        args.n_mc,
        args.mc_sample_size,
        args.mc_seed,
        args.feature_names,
    )
    elapsed = time.time() - start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    print(f"Output: {os.path.join(args.output_dir, OUTPUT_SUBDIR)}/")
    return 0


if __name__ == "__main__":
    def _init_logging():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"error_propagation_{timestamp}_{os.getpid()}.log"
        setup_logging(log_filename=log_filename)
        global logger
        logger = get_logger(__name__)

    _init_logging()
    try:
        exit_code = main()
    except Exception:
        logger.exception("Error-propagation run failed")
        raise
    sys.exit(exit_code)
