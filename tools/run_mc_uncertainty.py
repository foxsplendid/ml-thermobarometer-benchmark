#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MC 不确定性测试工具

使用方法:
    python tools/run_mc_uncertainty.py --model-module ert --corr-module segmented
    python tools/run_mc_uncertainty.py --data-module augmented --model-module catboost --feature-set Liquid

说明:
- 使用固定的 hold-out 测试集（由 prepare_splits 生成）
- 在完整 OOF 上拟合校正器（与主协议一致），然后在测试集子集上运行 MC 不确定性估计
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config_dict
from main import load_data, prepare_splits
from src.correction_modules import get_correction_module
from src.data_modules import get_data_module
from src.model_modules import get_model_module
from src.protocol import ExperimentConfig, Pipeline, StratifiedCVProtocol, compute_all_metrics
from src.uncertainty_modules import MCUncertaintyEstimator

# 从集中配置获取
BASE_CONFIG = get_config_dict()


def _build_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    model_params = {"random_seed": random_seed}
    data_params = {"random_seed": random_seed}

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


def _run_mc(config: ExperimentConfig,
            X_train: np.ndarray,
            y_T_train: np.ndarray,
            y_P_train: np.ndarray,
            groups_train: np.ndarray,
            tp_bins_train: np.ndarray,
            X_test: np.ndarray,
            y_T_test: np.ndarray,
            y_P_test: np.ndarray,
            output_dir: str,
            n_splits: int,
            random_seed: int,
            n_mc: int,
            mc_sample_size: int,
            mc_seed: int) -> None:
    rng = np.random.default_rng(mc_seed)
    sample_size = min(mc_sample_size, len(X_test))
    sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)

    X_mc = X_test[sample_idx]
    y_T_mc = y_T_test[sample_idx]
    y_P_mc = y_P_test[sample_idx]

    unc_dir = os.path.join(output_dir, "uncertainty")
    _ensure_dir(unc_dir)

    meta = {
        "exp_id": config.exp_id,
        "n_mc": n_mc,
        "mc_sample_size": int(sample_size),
        "mc_seed": int(mc_seed),
        "test_indices_sampled": sample_idx.tolist(),
    }
    _save_json(os.path.join(unc_dir, f"{config.exp_id}_mc_meta.json"), meta)

    mc = MCUncertaintyEstimator(
        n_mc=n_mc,
        error_model="epma",
        rel_err_high=0.03,
        rel_err_low=0.08,
        error_threshold=1.0,
        clip_min=0.0,
        random_seed=mc_seed,
    )

    def make_pipeline_factory(base_seed: int):
        def factory(seed: Optional[int] = None):
            seed_value = base_seed if seed is None else seed
            data_params = dict(config.data_params)
            model_params = dict(config.model_params)
            if 'random_seed' not in data_params:
                data_params['random_seed'] = seed_value
            if 'random_seed' not in model_params:
                model_params['random_seed'] = seed_value
            data_mod = get_data_module(config.data_module_name, **data_params)
            model_mod = get_model_module(config.model_module_name, **model_params)
            corr_mod = get_correction_module(config.corr_module_name, **config.corr_params)
            return Pipeline(data_mod, model_mod, corr_mod)
        return factory

    pipeline_factory = make_pipeline_factory(random_seed)

    def run_target(tag: str, y_true: np.ndarray, y_train: np.ndarray) -> None:
        corr_module = get_correction_module(config.corr_module_name, **config.corr_params)
        protocol = StratifiedCVProtocol(n_splits=n_splits, random_seed=random_seed)
        cv_results = protocol.run(
            X_train, y_train, groups_train,
            pipeline_factory,
            uncertainty_module=None,
            corr_module=corr_module,
            stratify_labels=tp_bins_train,
            verbose=False
        )
        corr_model = cv_results['corr_model']

        pipeline = pipeline_factory(random_seed)
        pipeline.fit(X_train, y_train, groups_train, stratify_labels=tp_bins_train)
        pipeline.set_correction(corr_module, corr_model)

        dist = mc.predict_distribution(pipeline, X_mc)
        calib = mc.compute_calibration_metrics(y_true, dist)
        metrics = compute_all_metrics(y_true, dist["median"])

        summary = {
            "exp_id": config.exp_id,
            "target": tag,
            "n_mc": n_mc,
            "mc_sample_size": int(sample_size),
        }
        summary.update(metrics)
        summary.update(calib)

        df = pd.DataFrame({
            "sample_idx": sample_idx,
            "y_true": y_true,
            "y_pred_median": dist["median"],
            "y_pred_p16": dist["p16"],
            "y_pred_p84": dist["p84"],
            "y_pred_p5": dist["p5"],
            "y_pred_p95": dist["p95"],
            "interval_width_68": dist["p84"] - dist["p16"],
            "interval_width_90": dist["p95"] - dist["p5"],
        })

        df.to_csv(os.path.join(unc_dir, f"{config.exp_id}_mc_{tag}_samples.csv"), index=False)
        pd.DataFrame([summary]).to_csv(
            os.path.join(unc_dir, f"{config.exp_id}_mc_{tag}_summary.csv"),
            index=False
        )

    run_target("T", y_T_mc, y_T_train)
    run_target("P", y_P_mc, y_P_train)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MC uncertainty tool for any experiment config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MC uncertainty with default settings
  python tools/run_mc_uncertainty.py --model-module ert

  # Run MC uncertainty with segmented correction
  python tools/run_mc_uncertainty.py --model-module stacking --corr-module segmented
"""
    )
    parser.add_argument("--exp-id", default="mc_test", help="experiment id prefix")
    parser.add_argument("--data-module", default="augmented",
                        choices=["raw", "balanced", "augmented"],
                        help="data module")
    parser.add_argument("--model-module", default="ert",
                        choices=["ert", "extratrees", "catboost", "rf", "randomforest", "stacking"],
                        help="model module")
    parser.add_argument("--corr-module", default="none",
                        choices=["none", "residual", "segmented"],
                        help="correction module")
    parser.add_argument("--feature-set", default="Liquid",
                        choices=["NoLiquid", "Liquid"],
                        help="feature set")

    parser.add_argument("--data-path", default=BASE_CONFIG["data_path"])
    parser.add_argument("--output-dir", default=BASE_CONFIG["output_dir"])
    parser.add_argument("--n-splits", type=int, default=BASE_CONFIG["n_splits"])
    parser.add_argument("--random-seed", type=int, default=BASE_CONFIG["random_seed"])

    parser.add_argument("--n-mc", type=int, default=1000, help="MC samples")
    parser.add_argument("--mc-sample-size", type=int, default=10, help="MC test sample size")
    parser.add_argument("--mc-seed", type=int, default=42)

    args = parser.parse_args()

    config = _build_config(
        exp_id=args.exp_id,
        data_module=args.data_module,
        model_module=args.model_module,
        corr_module=args.corr_module,
        feature_set=args.feature_set,
        random_seed=args.random_seed,
    )

    print(f"Experiment: {config.exp_id}")
    print(f"  data_module: {config.data_module_name}")
    print(f"  model_module: {config.model_module_name}")
    print(f"  corr_module: {config.corr_module_name}")
    print(f"  feature_set: {config.feature_set}")

    load_config = BASE_CONFIG.copy()
    load_config["data_path"] = args.data_path
    load_config["output_dir"] = args.output_dir

    X, y_T, y_P, groups = load_data(load_config, feature_set=args.feature_set)
    split = prepare_splits(X, y_T, y_P, groups, {"random_seed": args.random_seed})

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    tp_bins_train = split["tp_bins_train"]

    X_train = X[train_idx]
    y_T_train = y_T[train_idx]
    y_P_train = y_P[train_idx]
    groups_train = groups[train_idx]

    X_test = X[test_idx]
    y_T_test = y_T[test_idx]
    y_P_test = y_P[test_idx]

    _ensure_dir(args.output_dir)

    start = time.time()
    _run_mc(
        config,
        X_train, y_T_train, y_P_train, groups_train, tp_bins_train,
        X_test, y_T_test, y_P_test,
        args.output_dir,
        args.n_splits,
        args.random_seed,
        args.n_mc,
        args.mc_sample_size,
        args.mc_seed,
    )
    elapsed = time.time() - start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
