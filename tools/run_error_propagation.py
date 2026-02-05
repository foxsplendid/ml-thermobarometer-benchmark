#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析误差传播工具 (Analysis Error Propagation)

设计理念：
    评估 "EPMA 分析误差经固定模型放大后的输出离散度"（analysis-limited precision），
    而非模型的总体预测误差。

    核心原则：
    1. 模型固定：在全部训练数据上拟合一个固定模型，不做 CV、不重新训练
    2. 样本固定：对固定的测试样本进行多次输入扰动
    3. 仅扰动输入：对输入组成（氧化物 wt%）添加 EPMA 误差模型噪声
    4. 评估输出离散度：计算输出分布的标准差、区间宽度等

    EPMA 误差模型说明：
    - 采用按氧化物列名映射的相对误差（而非按数值阈值动态切换）
    - 主量元素（SiO2, Al2O3, FeO, MgO, CaO）：3% 相对误差
    - 低含量元素（TiO2, MnO, Na2O, Cr2O3, K2O）：8% 相对误差
    - 设计依据：Ágreda-López et al. (2024) ML_PT_Pyworkflow
    - 详见 src/perturbation.py::DEFAULT_OXIDE_REL_ERR

    扰动策略：
    - 不做负值截断（clip），保留完整正态分布以体现真实误差传播
    - 不做闭合约束（closure），与训练数据预处理保持一致

    输出指标含义：
    - analysis_std: 由分析误差导致的输出标准差（核心指标）
    - analysis_interval_68/90: 输出分布的 68%/90% 区间宽度
    - total_rmse/mae: 相对真值的总误差（包含模型误差）
    - analysis_contribution_ratio: 分析误差对总误差的近似贡献比例

使用方法:
    python tools/run_error_propagation.py --model-module ert
    python tools/run_error_propagation.py --data-module augmented --model-module catboost --feature-set Liquid

说明:
- 使用固定的 hold-out 测试集（由 prepare_splits 生成）
- 在全部训练数据上拟合一个固定模型（无 CV）
- 支持 --corr-module；若启用校正器，则在全训练集上拟合（非 OOF）
"""
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
from src.protocol import (
    ExperimentConfig,
    Pipeline,
    _derive_target_seed,
)
from src.metrics import compute_all_metrics
from src.uncertainty_modules import MCUncertaintyEstimator
from src.logger import setup_logging, get_logger

logger = logging.getLogger(__name__)

# 从集中配置获取
BASE_CONFIG = get_config_dict()

# 输出子目录
OUTPUT_SUBDIR = "error_propagation"


def _build_model_params(base_config: dict, model_module: str, random_seed: int) -> dict:
    model_defaults = base_config["model_defaults"]
    name = model_module.lower()

    if name in {"ert", "extratrees"}:
        params = dict(model_defaults["ert"])
    elif name in {"rf", "randomforest"}:
        params = dict(model_defaults["rf"])
    elif name in {"catboost", "cb"}:
        params = dict(model_defaults["catboost"])
    elif name == "stacking":
        stacking_params = dict(model_defaults.get("stacking", {}))
        base_params = {
            "ert": dict(model_defaults["ert"]),
            "catboost": dict(model_defaults["catboost"]),
            "rf": dict(model_defaults["rf"]),
        }
        for key, override in model_defaults.get("stacking_base_defaults", {}).items():
            if key in base_params and isinstance(override, dict):
                base_params[key].update(override)
        params = {"base_model_params": base_params}
        if stacking_params:
            params.update({
                "inner_cv": stacking_params.get("inner_cv"),
                "use_meta_scaler": stacking_params.get("use_meta_scaler"),
            })
    else:
        params = {}

    params["random_seed"] = random_seed
    return params


def _build_data_params(base_config: dict, data_module: str, feature_set: str, random_seed: int) -> dict:
    params = {"random_seed": random_seed}
    if data_module.lower() == "augmented":
        params["feature_names"] = base_config["feature_sets"][feature_set]
        params["n_aug"] = base_config["augmentation"]["n_aug"]
    return params


def _load_main_experiment_test_rmse(output_dir: str, exp_id: str, target: str) -> Optional[float]:
    """从主实验 results/metrics_summary.csv 读取独立测试集 RMSE（统一对比口径）"""
    summary_path = os.path.join(output_dir, "metrics_summary.csv")
    if not os.path.exists(summary_path):
        logger.warning(f"主实验 metrics_summary.csv 不存在: {summary_path}")
        return None

    try:
        df = pd.read_csv(summary_path)
        if "exp_id" not in df.columns:
            logger.warning(f"metrics_summary.csv 缺少 exp_id 列: {summary_path}")
            return None

        row = df[df["exp_id"] == exp_id]
        if row.empty:
            logger.warning(f"metrics_summary.csv 中未找到 exp_id={exp_id}")
            return None

        col = f"{target}_test_rmse"
        if col not in row.columns:
            logger.warning(f"metrics_summary.csv 缺少 {col} 列")
            return None

        test_rmse = float(row.iloc[0][col])
        logger.info(f"[{target}] 读取主实验 test_RMSE: {test_rmse:.2f} (from {exp_id})")
        return test_rmse
    except Exception as e:
        logger.warning(f"读取主实验 test_RMSE 失败: {e}")
        return None


def _build_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    """构建实验配置（允许选择校正模块）"""
    model_params = _build_model_params(BASE_CONFIG, model_module, random_seed)
    data_params = _build_data_params(BASE_CONFIG, data_module, feature_set, random_seed)

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
    """
    运行分析误差传播实验

    核心逻辑：
    1. 在全量训练数据上拟合一个固定模型（无 CV）
    2. 计算无扰动基线预测 y_pred_base
    3. 对测试集样本进行 n_mc 次输入扰动
    4. 统计输出分布的离散度（std, interval_width 等）
    5. 分离报告：总误差指标 vs 分析误差传播指标
    """
    rng = np.random.default_rng(mc_seed)

    # mc_sample_size <= 0 表示使用全部测试集
    if mc_sample_size <= 0:
        sample_size = len(X_test)
        sample_idx = np.arange(len(X_test))
        logger.info(f"使用全部测试集: {sample_size} 个样本")
    else:
        sample_size = min(mc_sample_size, len(X_test))
        sample_idx = rng.choice(len(X_test), size=sample_size, replace=False)
        logger.info(f"随机采样测试集: {sample_size} / {len(X_test)} 个样本")

    X_mc = X_test[sample_idx]
    y_T_mc = y_T_test[sample_idx]
    y_P_mc = y_P_test[sample_idx]

    # 输出目录：results/error_propagation/
    ep_dir = os.path.join(output_dir, OUTPUT_SUBDIR)
    _ensure_dir(ep_dir)

    # 获取特征列名（用于按列名映射 rel_err）
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
                f"无法根据特征数推断 feature_names，n_features={X_test.shape[1]}，"
                f"请通过 --feature-names 提供自定义列名"
            )
    if len(feature_names) != X_test.shape[1]:
        raise ValueError(f"feature_names 长度必须与特征数一致，len={len(feature_names)} n_features={X_test.shape[1]}")

    from src.perturbation import get_rel_err_vector
    rel_err_vec = get_rel_err_vector(feature_names, strict=True)

    # 保存元数据
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
            "analysis_std": "仅反映输入扰动引起的预测离散度",
            "total_error": "total_* 为未扰动基线预测的误差，包含模型误差",
            "ratio_note": "analysis_contribution_ratio 为近似贡献比例，非严格误差分解",
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

    # 为每个目标单独训练与评估，隔离随机性
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
            y_pred_train_raw = pipeline.predict_raw(X_train, apply_correction=False)
            corr_model = corr_mod.fit(y_train, y_pred_train_raw)
            pipeline.set_correction(corr_mod, corr_model)
        else:
            pipeline.set_correction(corr_mod, None)

        y_pred_base_raw = pipeline.predict_raw(X_mc, apply_correction=False)
        y_pred_base = pipeline.predict_raw(X_mc, apply_correction=True)

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
        logger.info(f"[{tag}] 完成: analysis_std_mean={summary['analysis_std_mean']:.2f}, "
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

  # Run with different model and feature set
  python tools/run_error_propagation.py --model-module catboost --feature-set NoLiquid
  
  # Use custom exp_id
  python tools/run_error_propagation.py --exp-id my_custom_exp --model-module stacking

Output:
  results/error_propagation/
  ├── {exp_id}_ep_meta.json           # Experiment metadata
  ├── {exp_id}_ep_T_summary.csv       # Temperature summary statistics
  ├── {exp_id}_ep_T_samples.csv       # Per-sample T predictions and uncertainty
  ├── {exp_id}_ep_P_summary.csv       # Pressure summary statistics
  └── {exp_id}_ep_P_samples.csv       # Per-sample P predictions and uncertainty
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
                        choices=["none", "residual", "segmented"],
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

    # 自动生成 exp_id（与主实验命名原则完全一致）
    # 主实验命名规则：
    # E01-E03: raw + ert/catboost/stacking + none
    # E04-E06: balanced + ert/catboost/stacking + none
    # E07-E09: augmented + ert/catboost/stacking + none
    # E10-E12: augmented + ert/catboost/stacking + segmented
    if args.exp_id is None:
        # 根据 data_module 和 model_module 确定实验编号
        data_module = args.data_module.lower()
        model_module = args.model_module.lower()

        # 模型到偏移量的映射
        model_offset = {'ert': 0, 'extratrees': 0, 'catboost': 1, 'rf': 0, 'randomforest': 0, 'stacking': 2}
        offset = model_offset.get(model_module, 0)

        # 数据模块到基准编号的映射（corr=none / segmented）
        data_base_none = {'raw': 1, 'balanced': 4, 'augmented': 7}
        if args.corr_module.lower() == "segmented" and data_module == "augmented":
            base_num = 10
        else:
            base_num = data_base_none.get(data_module, 7)  # 默认 augmented

        exp_num = base_num + offset
        suffix = 'noliq' if args.feature_set == 'NoLiquid' else 'liq'
        # 格式：E07_ert_augmented_none_liq / E10_ert_augmented_segmented_liq
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
        logger.exception("分析误差传播运行异常")
        raise
    sys.exit(exit_code)
