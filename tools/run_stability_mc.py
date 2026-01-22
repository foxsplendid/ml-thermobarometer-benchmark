#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run stability repeats and MC uncertainty for any experiment configuration.

Usage:
    python tools/run_stability_mc.py --exp-id E02 --model-module catboost --n-repeats 100
    python tools/run_stability_mc.py --exp-id E08 --data-module augmented --model-module catboost --feature-set Liquid
    python tools/run_stability_mc.py --model-module stacking --skip-mc

Notes:
- This script can run any model configuration, not just E08.
- Stability uses the fixed hold-out test set from prepare_splits.
- MC uncertainty runs on a random subset of the fixed test set.
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

from main import CONFIG as BASE_CONFIG
from main import load_data, prepare_splits
from src.correction_modules import get_correction_module
from src.data_modules import get_data_module
from src.model_modules import get_model_module
from src.protocol import ExperimentConfig, ExperimentMatrix, Pipeline, compute_all_metrics
from src.uncertainty_modules import MCUncertaintyEstimator


def _build_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    """构建实验配置"""
    # 统一使用 random_seed（模型内部自行转换为 sklearn 的 random_state）
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


def _run_stability(config: ExperimentConfig,
                   X_train: np.ndarray,
                   y_T_train: np.ndarray,
                   y_P_train: np.ndarray,
                   groups_train: np.ndarray,
                   tp_bins_train: np.ndarray,
                   X_test: np.ndarray,
                   y_T_test: np.ndarray,
                   y_P_test: np.ndarray,
                   groups_test: np.ndarray,
                   output_dir: str,
                   n_splits: int,
                   n_repeats: int,
                   test_size: float,
                   checkpoint_interval: int,
                   seed_start: int) -> None:
    matrix = ExperimentMatrix(
        X=X_train,
        y_T=y_T_train,
        y_P=y_P_train,
        groups=groups_train,
        output_dir=output_dir,
    )
    matrix.run_stability_repeats(
        configs=[config],
        X_test=X_test,
        y_T_test=y_T_test,
        y_P_test=y_P_test,
        groups_test=groups_test,
        stratify_labels=tp_bins_train,
        n_splits=n_splits,
        test_size=test_size,
        n_repeats=n_repeats,
        checkpoint_interval=checkpoint_interval,
        random_seed=seed_start,
        verbose=True,
    )


def _fit_pipeline(config: ExperimentConfig,
                  X_train: np.ndarray,
                  y_train: np.ndarray,
                  groups_train: np.ndarray,
                  stratify_labels: Optional[np.ndarray]) -> Pipeline:
    data_mod = get_data_module(config.data_module_name, **config.data_params)
    model_mod = get_model_module(config.model_module_name, **config.model_params)
    corr_mod = get_correction_module(config.corr_module_name, **config.corr_params)

    pipeline = Pipeline(data_mod, model_mod, corr_mod)
    pipeline.fit(X_train, y_train, groups_train, stratify_labels=stratify_labels)
    return pipeline


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

    # Train separate pipelines for T and P
    pipeline_T = _fit_pipeline(config, X_train, y_T_train, groups_train, tp_bins_train)
    pipeline_P = _fit_pipeline(config, X_train, y_P_train, groups_train, tp_bins_train)

    def run_target(tag: str, y_true: np.ndarray, pipeline: Pipeline) -> None:
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

    run_target("T", y_T_mc, pipeline_T)
    run_target("P", y_P_mc, pipeline_P)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stability repeats + MC uncertainty for any experiment config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CatBoost stability test with default settings
  python tools/run_stability_mc.py --model-module catboost --n-repeats 100

  # Run ExtraTrees with NoLiquid feature set
  python tools/run_stability_mc.py --model-module ert --feature-set NoLiquid

  # Run Stacking with segmented correction, skip MC
  python tools/run_stability_mc.py --model-module stacking --corr-module segmented --skip-mc
"""
    )
    # 实验配置
    parser.add_argument("--exp-id", default="stability_test", help="实验ID（输出文件名前缀）")
    parser.add_argument("--data-module", default="augmented", 
                        choices=["raw", "balanced", "augmented"],
                        help="数据模块")
    parser.add_argument("--model-module", default="catboost",
                        choices=["ert", "extratrees", "catboost", "rf", "randomforest", "stacking"],
                        help="模型模块")
    parser.add_argument("--corr-module", default="none",
                        choices=["none", "residual", "segmented"],
                        help="校正模块")
    parser.add_argument("--feature-set", default="Liquid",
                        choices=["NoLiquid", "Liquid"],
                        help="特征集")
    
    # 通用参数
    parser.add_argument("--data-path", default=BASE_CONFIG["data_path"])
    parser.add_argument("--output-dir", default=BASE_CONFIG["output_dir"])
    parser.add_argument("--n-splits", type=int, default=BASE_CONFIG["n_splits"])
    parser.add_argument("--random-seed", type=int, default=BASE_CONFIG["random_seed"])
    
    # 稳定性测试参数（使用硬编码默认值，因main.py已移除相关配置）
    parser.add_argument("--n-repeats", type=int, default=1000,
                        help="重复次数（默认1000）")
    parser.add_argument("--stability-test-size", type=float, default=0.3,
                        help="稳定性测试集比例（默认0.3）")
    parser.add_argument("--stability-seed-start", type=int, default=0,
                        help="稳定性测试起始种子（默认0）")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="分批保存间隔（默认100次保存一次checkpoint）")
    
    # MC 不确定性参数
    parser.add_argument("--n-mc", type=int, default=1000, help="MC采样次数")
    parser.add_argument("--mc-sample-size", type=int, default=10, help="MC测试样本数")
    parser.add_argument("--mc-seed", type=int, default=42)
    
    # 跳过选项
    parser.add_argument("--skip-stability", action="store_true", help="跳过稳定性测试")
    parser.add_argument("--skip-mc", action="store_true", help="跳过MC不确定性")
    
    args = parser.parse_args()

    # 构建实验配置
    config = _build_config(
        exp_id=args.exp_id,
        data_module=args.data_module,
        model_module=args.model_module,
        corr_module=args.corr_module,
        feature_set=args.feature_set,
        random_seed=args.random_seed,
    )
    
    print(f"实验配置: {config.exp_id}")
    print(f"  数据模块: {config.data_module_name}")
    print(f"  模型模块: {config.model_module_name}")
    print(f"  校正模块: {config.corr_module_name}")
    print(f"  特征集: {config.feature_set}")

    # 加载数据（使用完整的BASE_CONFIG，覆盖命令行参数）
    load_config = BASE_CONFIG.copy()
    load_config['data_path'] = args.data_path
    load_config['output_dir'] = args.output_dir

    X, y_T, y_P, groups = load_data(load_config, feature_set=args.feature_set)

    split_config = {'random_seed': args.random_seed}
    split = prepare_splits(X, y_T, y_P, groups, split_config)

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
    groups_test = groups[test_idx]

    _ensure_dir(args.output_dir)

    start = time.time()
    
    if not args.skip_stability:
        print(f"\n运行稳定性测试 ({args.n_repeats} 次重复)...")
        _run_stability(
            config,
            X_train, y_T_train, y_P_train, groups_train, tp_bins_train,
            X_test, y_T_test, y_P_test, groups_test,
            args.output_dir,
            args.n_splits,
            args.n_repeats,
            args.stability_test_size,
            args.checkpoint_interval,
            args.stability_seed_start,
        )

    if not args.skip_mc:
        print(f"\n运行MC不确定性估计 ({args.n_mc} 次采样)...")
        _run_mc(
            config,
            X_train, y_T_train, y_P_train, groups_train, tp_bins_train,
            X_test, y_T_test, y_P_test,
            args.output_dir,
            args.n_mc,
            args.mc_sample_size,
            args.mc_seed,
        )

    elapsed = time.time() - start
    print(f"\n完成，耗时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
