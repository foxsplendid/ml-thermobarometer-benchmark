#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
稳定性测试工具 - 评估模型在不同随机划分下的性能稳定性

功能：
    对固定测试集进行多次随机训练集划分，评估模型性能的稳定性
    输出 RMSE/R² 的均值、标准差、置信区间

使用方法：

    1. 快速测试（100次重复，约30分钟）：
       python tools/run_stability.py --n-repeats 100

    2. 完整运行（1000次重复，约5小时）：
       python tools/run_stability.py --n-repeats 1000

    3. 分段运行（推荐，支持断点续跑）：
       # 第一段（0-199）
       python tools/run_stability.py --repeat-start 0 --repeat-end 199
       # 第二段（200-399）
       python tools/run_stability.py --repeat-start 200 --repeat-end 399
       # ...更多段...
       # 合并
       python tools/run_stability.py --merge-dir results

    4. 断点续跑（自动从最新检查点继续）：
       python tools/run_stability.py --n-repeats 1000 --resume

    5. 自定义配置：
       python tools/run_stability.py --exp-id E07_stability --model-module ert --data-module augmented

输出：
    results/stability/
    ├── {exp_id}_T_test_metrics.csv       # T 目标的所有重复结果
    ├── {exp_id}_P_test_metrics.csv       # P 目标的所有重复结果
    ├── {exp_id}_*_checkpoint_*.csv       # 检查点文件（每100次）
    └── stability_summary.csv             # 汇总统计

Notes:
- 使用固定的 hold-out 测试集（由 prepare_splits 生成）
- 每次重复使用不同的随机种子划分训练集
- 检查点默认每 100 次保存，支持中断后恢复
"""
import argparse
import glob
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
from src.metrics import summarize_folds
from src.protocol import ExperimentConfig, ExperimentMatrix

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



def _resolve_stability_dir(base_dir: str) -> str:
    if os.path.basename(base_dir) == "stability":
        return base_dir
    return os.path.join(base_dir, "stability")

def _collect_segment_files(stability_dir: str, exp_id: str, target: str) -> list:
    pattern = os.path.join(stability_dir, f"{exp_id}_{target}_test_metrics_rep_*.csv")
    return sorted(glob.glob(pattern))

def _merge_stability_segments(merge_dir: str, output_dir: str, exp_id: str, verbose: bool = True) -> None:
    stability_dir = _resolve_stability_dir(merge_dir)
    out_stability_dir = _resolve_stability_dir(output_dir)
    _ensure_dir(out_stability_dir)

    summary_rows = []
    for target in ["T", "P"]:
        segment_files = _collect_segment_files(stability_dir, exp_id, target)
        if not segment_files:
            raise ValueError(f"no segment files found for {exp_id} {target} in {stability_dir}")
        if verbose:
            print(f"  {target}: {len(segment_files)} segment files")

        frames = [pd.read_csv(p) for p in segment_files]
        merged = pd.concat(frames, ignore_index=True)

        if "repeat_id" in merged.columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=["repeat_id"]).sort_values("repeat_id")
            after = len(merged)
            if verbose and after != before:
                print(f"  {target}: dropped {before - after} duplicated rows by repeat_id")

        merged_path = os.path.join(out_stability_dir, f"{exp_id}_{target}_test_metrics.csv")
        merged.to_csv(merged_path, index=False)

        summary = summarize_folds(merged.to_dict("records"), compute_ci=True)
        summary_row = {"exp_id": exp_id, "target": target}
        summary_row.update(summary)
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_stability_dir, "stability_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    if verbose:
        print(f"  summary: {summary_path}")
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
                   repeat_start: int,
                   repeat_end: int,
                   segment_tag: Optional[str],
                   write_summary: bool,
                   test_size: float,
                   checkpoint_interval: int,
                   seed_start: int,
                   resume: bool) -> None:
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
        repeat_start=repeat_start,
        repeat_end=repeat_end,
        segment_tag=segment_tag,
        write_summary=write_summary,
        checkpoint_interval=checkpoint_interval,
        random_seed=seed_start,
        resume=resume,
        verbose=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stability repeats for any experiment config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CatBoost stability test with default settings
  python tools/run_stability.py --model-module catboost --n-repeats 100

  # Run ExtraTrees with NoLiquid feature set
  python tools/run_stability.py --model-module ert --feature-set NoLiquid

  # Run Stacking with segmented correction
  python tools/run_stability.py --model-module stacking --corr-module segmented
"""
    )
    # 实验配置
    parser.add_argument("--exp-id", default="stability_test", help="实验ID（输出文件名前缀）")
    parser.add_argument("--data-module", default="augmented", 
                        choices=["raw", "balanced", "augmented"],
                        help="数据模块")
    parser.add_argument("--model-module", default="ert",
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
    parser.add_argument("--repeat-start", type=int, default=None,
                        help="repeat start (inclusive), for segmented runs")
    parser.add_argument("--repeat-end", type=int, default=None,
                        help="repeat end (inclusive), for segmented runs")
    parser.add_argument("--merge-dir", type=str, default=None,
                        help="merge segments and finalize summary")
    parser.add_argument("--stability-test-size", type=float, default=0.3,
                        help="stability test size (default: 0.3)")
    parser.add_argument("--stability-seed-start", type=int, default=0,
                        help="seed start for stability repeats (default: 0)")
    parser.add_argument("--checkpoint-interval", type=int, default=100,
                        help="checkpoint interval in repeats (default: 100)")
    parser.add_argument("--resume", action="store_true",
                        help="resume from latest checkpoint/test_metrics")

    # skip options
    parser.add_argument("--skip-stability", action="store_true", help="skip stability test")

    args = parser.parse_args()

    if (args.repeat_start is None) ^ (args.repeat_end is None):
        raise ValueError("repeat-start and repeat-end must be provided together")
    if args.repeat_start is None and args.repeat_end is None:
        repeat_start = 0
        repeat_end = args.n_repeats - 1
        segmented = False
    else:
        repeat_start = args.repeat_start
        repeat_end = args.repeat_end
        if repeat_end < repeat_start:
            raise ValueError("repeat-end must be >= repeat-start")
        segmented = True

    segment_tag = f"rep_{repeat_start:03d}_{repeat_end:03d}" if segmented else None

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
    if args.merge_dir:
        _ensure_dir(args.output_dir)
        print("=" * 70)
        print("Stability - finalize from segments")
        print("=" * 70)
        print(f"  merge_dir: {args.merge_dir}")
        print(f"  output_dir: {args.output_dir}")
        _merge_stability_segments(args.merge_dir, args.output_dir, args.exp_id, verbose=True)
        return 0

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
        if segmented:
            print(f"\nRunning stability segment: {repeat_start}-{repeat_end}")
        else:
            print(f"\nRunning stability repeats: {args.n_repeats}")
        _run_stability(
            config,
            X_train, y_T_train, y_P_train, groups_train, tp_bins_train,
            X_test, y_T_test, y_P_test, groups_test,
            args.output_dir,
            args.n_splits,
            args.n_repeats,
            repeat_start,
            repeat_end,
            segment_tag,
            (not segmented),
            args.stability_test_size,
            args.checkpoint_interval,
            args.stability_seed_start,
            args.resume,
        )

    elapsed = time.time() - start
    print(f"\n完成，耗时 {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
