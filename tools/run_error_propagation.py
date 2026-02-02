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
- 默认不使用校正器（分析误差传播实验关注输入扰动，不涉及模型系统误差校正）
"""
import argparse
import json
import logging
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
from src.data_modules import get_data_module
from src.model_modules import get_model_module
from src.protocol import (
    ExperimentConfig,
    Pipeline,
    compute_all_metrics,
)
from src.uncertainty_modules import MCUncertaintyEstimator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 从集中配置获取
BASE_CONFIG = get_config_dict()

# 输出子目录
OUTPUT_SUBDIR = "error_propagation"


def _load_main_experiment_rmse(output_dir: str, exp_id: str, target: str) -> Optional[float]:
    """
    从主实验结果中读取 CV-RMSE 均值

    Parameters
    ----------
    output_dir : str
        结果目录（如 results/）
    exp_id : str
        实验 ID（如 E07_ert_augmented_none_liq）
    target : str
        目标变量（T 或 P）

    Returns
    -------
    float or None
        CV-RMSE 均值，如果文件不存在则返回 None
    """
    # 主实验结果文件路径
    fold_metrics_path = os.path.join(output_dir, f"{exp_id}_{target}_fold_metrics.csv")

    if not os.path.exists(fold_metrics_path):
        logger.warning(f"主实验结果文件不存在: {fold_metrics_path}")
        return None

    try:
        df = pd.read_csv(fold_metrics_path)
        if 'rmse' in df.columns:
            cv_rmse_mean = df['rmse'].mean()
            logger.info(f"[{target}] 读取主实验 CV-RMSE: {cv_rmse_mean:.2f} (from {exp_id})")
            return float(cv_rmse_mean)
        else:
            logger.warning(f"主实验结果文件中无 rmse 列: {fold_metrics_path}")
            return None
    except Exception as e:
        logger.warning(f"读取主实验结果失败: {e}")
        return None


def _build_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    """构建实验配置（无校正模块）"""
    model_params = {"random_seed": random_seed}
    data_params = {"random_seed": random_seed}

    return ExperimentConfig(
        exp_id=exp_id,
        data_module_name=data_module,
        model_module_name=model_module,
        corr_module_name="none",  # 分析误差传播不使用校正
        feature_set=feature_set,
        data_params=data_params,
        model_params=model_params,
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


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
    if X_test.shape[1] == 18:
        feature_names = data_config.feature_sets['Liquid']
    elif X_test.shape[1] == 9:
        feature_names = data_config.feature_sets['NoLiquid']
    else:
        feature_names = [f'feature_{i}' for i in range(X_test.shape[1])]

    # 获取按列名映射的 rel_err
    from src.perturbation import get_rel_err_vector
    rel_err_vec = get_rel_err_vector(feature_names)

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
        "n_mc": n_mc,
        "mc_sample_size": int(sample_size),
        "mc_seed": int(mc_seed),
        "test_indices_sampled": sample_idx.tolist(),
        "epma_error_model": {
            "type": "per_oxide_mapping",
            "description": "Relative error mapped by oxide column name (3% for major, 8% for trace)",
            "feature_names": feature_names,
            "rel_err_vec": rel_err_vec.tolist(),
        },
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }
    _save_json(os.path.join(ep_dir, f"{config.exp_id}_ep_meta.json"), meta)

    # 创建 MC 不确定性估计器（使用按列名映射的 rel_err）
    mc = MCUncertaintyEstimator(
        n_mc=n_mc,
        feature_names=feature_names,
        random_seed=mc_seed,
    )

    def run_target(tag: str, y_true_mc: np.ndarray, y_train: np.ndarray) -> None:
        """对单个目标（T 或 P）运行分析误差传播"""
        logger.info(f"[{tag}] 拟合固定模型（全量训练数据）...")

        # 构建 Pipeline（无校正）
        data_params = dict(config.data_params)
        model_params = dict(config.model_params)
        data_params['random_seed'] = random_seed
        model_params['random_seed'] = random_seed

        data_mod = get_data_module(config.data_module_name, **data_params)
        model_mod = get_model_module(config.model_module_name, **model_params)

        # 使用无校正的 Pipeline
        from src.correction_modules import get_correction_module
        corr_mod = get_correction_module("none")
        pipeline = Pipeline(data_mod, model_mod, corr_mod)

        # 在全量训练数据上拟合（固定模型）
        pipeline.fit(X_train, y_train)

        # === 关键：计算无扰动基线预测 ===
        y_pred_base = pipeline.predict_raw(X_mc)

        logger.info(f"[{tag}] 运行 MC 误差传播 (n_mc={n_mc})...")

        # 运行 MC 扰动
        dist = mc.predict_distribution(pipeline, X_mc)

        # === 分离两类指标 ===

        # 1. 总误差指标（相对真值，包含模型误差）
        total_metrics = compute_all_metrics(y_true_mc, y_pred_base)

        # 2. 分析误差传播指标（输出离散度，仅反映输入扰动）
        output_std = dist["std"]  # 每个样本的输出标准差
        interval_68 = dist["p84"] - dist["p16"]  # 68% 区间宽度
        interval_90 = dist["p95"] - dist["p5"]  # 90% 区间宽度

        # 3. MAD (Median Absolute Deviation) - 稳健统计量，适用于非正态分布
        # 对每个样本计算其 MC 预测分布的 MAD
        samples = dist["samples"]  # shape: (n_mc, n_samples)
        sample_medians = dist["median"]  # shape: (n_samples,)
        # MAD = median(|x - median(x)|) for each sample
        output_mad = np.median(np.abs(samples - sample_medians), axis=0)
        # 传播误差 = median ± 2*MAD（Li & Zhang 推荐形式）
        propagated_error_2mad = 2 * output_mad

        # 4. 扰动引起的偏移（用于自检）
        delta_median = dist["median"] - y_pred_base
        delta_mean = dist["mean"] - y_pred_base

        # 5. 读取主实验 CV-RMSE 用于对比
        main_exp_rmse = _load_main_experiment_rmse(output_dir, config.exp_id, tag)

        # 汇总统计（明确区分两类指标）
        summary = {
            "exp_id": config.exp_id,
            "target": tag,
            "n_mc": n_mc,
            "mc_sample_size": int(sample_size),

            # === 总误差指标（模型 + 数据，相对真值）===
            "total_rmse": total_metrics["rmse"],
            "total_mae": total_metrics["mae"],
            "total_mbe": total_metrics["mbe"],
            "total_r2": total_metrics["r2"],

            # === 主实验 CV-RMSE（用于对比）===
            "main_cv_rmse": main_exp_rmse,

            # === 分析误差传播指标（输出离散度）===
            "analysis_std_mean": float(np.mean(output_std)),
            "analysis_std_median": float(np.median(output_std)),
            "analysis_std_max": float(np.max(output_std)),
            "analysis_interval_68_mean": float(np.mean(interval_68)),
            "analysis_interval_68_median": float(np.median(interval_68)),
            "analysis_interval_90_mean": float(np.mean(interval_90)),
            "analysis_interval_90_median": float(np.median(interval_90)),

            # === MAD 统计量（稳健，适用于非正态分布）===
            "analysis_mad_mean": float(np.mean(output_mad)),
            "analysis_mad_median": float(np.median(output_mad)),
            "analysis_2mad_mean": float(np.mean(propagated_error_2mad)),
            "analysis_2mad_median": float(np.median(propagated_error_2mad)),

            # === 自检指标（扰动引起的系统偏移，应接近 0）===
            "delta_median_mean": float(np.mean(delta_median)),
            "delta_median_std": float(np.std(delta_median)),
            "delta_mean_mean": float(np.mean(delta_mean)),

            # === 与主实验对比（传播误差 vs CV误差）===
            # 按方差分解：total^2 ≈ model^2 + analysis^2
            "analysis_contribution_ratio": float(np.mean(output_std)**2 / (total_metrics["rmse"]**2 + 1e-10)),
            # 传播误差占 CV-RMSE 的比例
            "propagated_vs_cv_ratio": float(np.mean(propagated_error_2mad) / (main_exp_rmse + 1e-10)) if main_exp_rmse else None,
        }

        # 逐样本结果
        df = pd.DataFrame({
            "sample_idx": sample_idx,
            "y_true": y_true_mc,
            # 基线预测（无扰动）
            "y_pred_base": y_pred_base,
            # MC 统计量
            "y_pred_mean": dist["mean"],
            "y_pred_median": dist["median"],
            "y_pred_std": output_std,
            "y_pred_mad": output_mad,
            "y_pred_2mad": propagated_error_2mad,
            "y_pred_p5": dist["p5"],
            "y_pred_p16": dist["p16"],
            "y_pred_p84": dist["p84"],
            "y_pred_p95": dist["p95"],
            # 区间宽度
            "interval_68": interval_68,
            "interval_90": interval_90,
            # 扰动引起的偏移（自检用）
            "delta_median": delta_median,
            "delta_mean": delta_mean,
        })

        # 保存结果
        df.to_csv(os.path.join(ep_dir, f"{config.exp_id}_ep_{tag}_samples.csv"), index=False)
        pd.DataFrame([summary]).to_csv(
            os.path.join(ep_dir, f"{config.exp_id}_ep_{tag}_summary.csv"),
            index=False
        )

        # 打印对比信息
        cv_info = f", main_cv_rmse={main_exp_rmse:.2f}" if main_exp_rmse else ""
        ratio_info = f", propagated/cv={summary['propagated_vs_cv_ratio']:.1%}" if summary.get('propagated_vs_cv_ratio') else ""

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
    parser.add_argument("--feature-set", default="Liquid",
                        choices=["NoLiquid", "Liquid"],
                        help="feature set (default: Liquid)")

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

        # 数据模块到基准编号的映射（corr=none）
        data_base = {'raw': 1, 'balanced': 4, 'augmented': 7}
        base_num = data_base.get(data_module, 7)  # 默认 augmented

        exp_num = base_num + offset
        suffix = 'noliq' if args.feature_set == 'NoLiquid' else 'liq'
        # 格式：E07_ert_augmented_none_liq（与主实验一致，不加 _ep 后缀）
        exp_id = f"E{exp_num:02d}_{model_module}_{data_module}_none_{suffix}"
    else:
        exp_id = args.exp_id

    config = _build_config(
        exp_id=exp_id,
        data_module=args.data_module,
        model_module=args.model_module,
        feature_set=args.feature_set,
        random_seed=args.random_seed,
    )

    print(f"="*60)
    print(f"Analysis Error Propagation Experiment")
    print(f"="*60)
    print(f"Experiment ID: {config.exp_id}")
    print(f"  data_module: {config.data_module_name}")
    print(f"  model_module: {config.model_module_name}")
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
    )
    elapsed = time.time() - start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    print(f"Output: {os.path.join(args.output_dir, OUTPUT_SUBDIR)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
