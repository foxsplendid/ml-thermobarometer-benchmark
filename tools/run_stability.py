# -*- coding: utf-8 -*-
"""Run stability repeats (bootstrapped train/test splits) for any experiment."""

import argparse
import glob
import logging
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

# Bootstrap sys.path so tools/_common.py can find the repo root.
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tools._common import BASE_CONFIG, add_common_args, build_experiment_config, init_tool_logging
from main import load_data, prepare_splits
from src.metrics import summarize_folds
from src.protocol import ExperimentConfig, ExperimentMatrix

logger = logging.getLogger(__name__)


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
                   tp_bins_train: np.ndarray,
                   X_test: np.ndarray,
                   y_T_test: np.ndarray,
                   y_P_test: np.ndarray,
                   output_dir: str,
                   n_splits: int,
                   n_repeats: int,
                   repeat_start: int,
                   repeat_end: int,
                   segment_tag: Optional[str],
                   write_summary: bool,
                   test_size: float,
                   random_seed: int,
                   repeat_workers: int = 1) -> None:
    matrix = ExperimentMatrix(
        X=X_train,
        y_T=y_T_train,
        y_P=y_P_train,
        output_dir=output_dir,
    )
    matrix.run_stability_repeats(
        configs=[config],
        X_test=X_test,
        y_T_test=y_T_test,
        y_P_test=y_P_test,
        stratify_labels=tp_bins_train,
        n_splits=n_splits,
        test_size=test_size,
        n_repeats=n_repeats,
        repeat_start=repeat_start,
        repeat_end=repeat_end,
        segment_tag=segment_tag,
        write_summary=write_summary,
        random_seed=random_seed,
        verbose=True,
        repeat_workers=repeat_workers,
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
    add_common_args(parser)
    parser.set_defaults(exp_id="stability_test")
    
    parser.add_argument("--n-repeats", type=int, default=1000,
                        help="Number of repeats (default: 1000)")
    parser.add_argument("--repeat-start", type=int, default=None,
                        help="repeat start (inclusive), for segmented runs")
    parser.add_argument("--repeat-end", type=int, default=None,
                        help="repeat end (inclusive), for segmented runs")
    parser.add_argument("--merge-dir", type=str, default=None,
                        help="merge segments and finalize summary")
    parser.add_argument("--stability-test-size", type=float, default=0.3,
                        help="stability test size (default: 0.3)")
    parser.add_argument("--repeat-workers", type=int, default=None,
                        help="parallel repeat workers (default: auto = max(1, cpu_count // n_jobs));"
                             " set to 1 to force sequential")


    args = parser.parse_args()

    # Auto-detect repeat_workers: same formula as fold_workers (cpu_count // n_jobs)
    if args.repeat_workers is None:
        from src.runtime import get_n_jobs
        cpu_count = os.cpu_count() or 1
        n_jobs = get_n_jobs()
        n_per_worker = n_jobs if n_jobs > 0 else cpu_count
        repeat_workers = max(1, cpu_count // n_per_worker)
    else:
        repeat_workers = max(1, args.repeat_workers)

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

    config = build_experiment_config(
        exp_id=args.exp_id,
        data_module=args.data_module,
        model_module=args.model_module,
        corr_module=args.corr_module,
        feature_set=args.feature_set,
        random_seed=args.random_seed,
    )
    
    print(f"Experiment config: {config.exp_id}")
    print(f"  Data module: {config.data_module_name}")
    print(f"  Model module: {config.model_module_name}")
    print(f"  Correction module: {config.corr_module_name}")
    print(f"  Feature set: {config.feature_set}")

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

    X, y_T, y_P = load_data(load_config, feature_set=args.feature_set)

    split_config = {'random_seed': args.random_seed}
    split = prepare_splits(X, y_T, y_P, split_config)

    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    tp_bins_train = split["tp_bins_train"]

    X_train = X[train_idx]
    y_T_train = y_T[train_idx]
    y_P_train = y_P[train_idx]

    X_test = X[test_idx]
    y_T_test = y_T[test_idx]
    y_P_test = y_P[test_idx]

    _ensure_dir(args.output_dir)

    start = time.time()
    
    if segmented:
        print(f"\nRunning stability segment: {repeat_start}-{repeat_end}"
              f" (repeat_workers={repeat_workers})")
    else:
        print(f"\nRunning stability repeats: {args.n_repeats}"
              f" (repeat_workers={repeat_workers})")
    _run_stability(
        config,
        X_train, y_T_train, y_P_train, tp_bins_train,
        X_test, y_T_test, y_P_test,
        args.output_dir,
        args.n_splits,
        args.n_repeats,
        repeat_start,
        repeat_end,
        segment_tag,
        (not segmented),
        args.stability_test_size,
        args.random_seed,
        repeat_workers=repeat_workers,
    )

    elapsed = time.time() - start
    print(f"\nDone. Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    logger = init_tool_logging("stability")
    try:
        exit_code = main()
    except Exception:
        logger.exception("Stability test run failed")
        raise
    sys.exit(exit_code)
