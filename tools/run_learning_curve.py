# -*- coding: utf-8 -*-
"""Learning-curve tool: varies training-set size to measure data efficiency."""

import argparse
import glob
import logging
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Bootstrap sys.path so tools/_common.py can find the repo root.
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from tools._common import BASE_CONFIG, init_tool_logging
from main import load_data, prepare_splits
from src.correction_modules import get_correction_module
from src.data_modules import get_data_module
from src.utils import build_data_params, build_model_params
from src.model_modules import get_model_module
from src.protocol import (
    Pipeline,
    StratifiedCVProtocol,
    merge_sparse_bins,
    get_effective_n_splits,
    derive_target_seed,
)

OUTPUT_SUBDIR = "learning_curve"
logger = logging.getLogger(__name__)


def summarize_runs(runs_df: pd.DataFrame, requested_n_splits: int) -> pd.DataFrame:
    summary_records = []

    for (fraction, model, target), group in runs_df.groupby(['fraction', 'model', 'target']):
        rmse_values = group['rmse_mean'].dropna().values if 'rmse_mean' in group else np.array([])
        r2_values = group['r2_mean'].dropna().values if 'r2_mean' in group else np.array([])

        n_valid = len(rmse_values)
        if n_valid == 0:
            continue

        rmse_mean_of_repeats = np.mean(rmse_values)
        rmse_std_of_repeats = np.std(rmse_values, ddof=1) if n_valid > 1 else np.nan
        r2_mean_of_repeats = np.mean(r2_values) if len(r2_values) > 0 else np.nan
        r2_std_of_repeats = np.std(r2_values, ddof=1) if len(r2_values) > 1 else np.nan

        if n_valid > 2:
            from scipy import stats
            se = stats.sem(rmse_values, ddof=1)
            ci = stats.t.interval(0.95, n_valid - 1, loc=rmse_mean_of_repeats, scale=se)
            rmse_ci_low, rmse_ci_high = ci
        else:
            rmse_ci_low, rmse_ci_high = np.nan, np.nan

        if 'n_splits_requested' in group:
            req_vals = group['n_splits_requested'].dropna().unique()
            n_splits_requested = int(req_vals.max()) if len(req_vals) > 0 else requested_n_splits
        else:
            n_splits_requested = requested_n_splits

        n_splits_used_unique = group['n_splits_used'].unique() if 'n_splits_used' in group else np.array([])
        n_effective_bins_mean = group['n_effective_bins'].mean() if 'n_effective_bins' in group else np.nan
        n_splits_used_min = np.nan
        n_splits_used_max = np.nan
        n_splits_used_mean = np.nan
        bins_merged_ratio = np.nan
        notes = ""
        if len(n_splits_used_unique) > 0:
            min_splits_used = int(np.min(n_splits_used_unique))
            if len(n_splits_used_unique) > 1 or min_splits_used < n_splits_requested:
                notes = f"n_splits_downgraded_to_{min_splits_used}"
            n_splits_used_min = min_splits_used
            n_splits_used_max = int(np.max(n_splits_used_unique))
            n_splits_used_mean = float(group['n_splits_used'].mean())
        if 'bins_merged' in group:
            bins_merged_ratio = float(group['bins_merged'].mean())

        summary_records.append({
            'fraction': fraction,
            'model': model,
            'target': target,
            'n_train_sub_mean': group['n_train_sub'].mean() if 'n_train_sub' in group else np.nan,
            'n_splits_requested': n_splits_requested,
            'n_splits_used_min': n_splits_used_min,
            'n_splits_used_max': n_splits_used_max,
            'n_splits_used_mean': n_splits_used_mean,
            'n_valid_repeats': n_valid,
            'rmse_mean_of_repeats': rmse_mean_of_repeats,
            'rmse_std_of_repeats': rmse_std_of_repeats,
            'rmse_ci_low': rmse_ci_low,
            'rmse_ci_high': rmse_ci_high,
            'r2_mean_of_repeats': r2_mean_of_repeats,
            'r2_std_of_repeats': r2_std_of_repeats,
            'n_effective_bins_mean': n_effective_bins_mean,
            'bins_merged_ratio': bins_merged_ratio,
            'notes': notes,
        })

    return pd.DataFrame(summary_records)


def collect_runs_files(merge_dir: str) -> List[str]:
    pattern = os.path.join(merge_dir, "**", "learning_curve_runs*.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        return []
    segment_paths = [p for p in paths if os.path.basename(p) != "learning_curve_runs.csv"]
    return sorted(segment_paths) if segment_paths else sorted(paths)


def merge_runs_files(merge_dir: str) -> pd.DataFrame:
    paths = collect_runs_files(merge_dir)
    if not paths:
        raise ValueError(f"no runs files found under: {merge_dir}")
    frames = [pd.read_csv(p) for p in paths]
    runs_df = pd.concat(frames, ignore_index=True)

    key_cols = ["fraction", "repeat", "model", "target", "feature_set", "data_module", "correction_module"]
    if all(c in runs_df.columns for c in key_cols):
        dup_count = runs_df.duplicated(subset=key_cols).sum()
        if dup_count > 0:
            print(f"Warning: {dup_count} duplicated rows detected in merged runs")

    return runs_df


# ============================================================
# Per-task helpers
# ============================================================

def run_cv_for_subsample(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    strat_labels_sub: np.ndarray,
    pipeline_factory,
    corr_module,
    n_splits: int,
    random_seed: int,
) -> Dict[str, Any]:
    n_bins_raw = len(np.unique(strat_labels_sub)) if strat_labels_sub is not None else 0
    merged_labels = merge_sparse_bins(strat_labels_sub, min_samples_per_bin=n_splits)
    n_bins_merged = len(np.unique(merged_labels)) if merged_labels is not None else 0
    bins_merged = int(n_bins_raw > 0 and n_bins_merged < n_bins_raw)
    effective_n_splits = get_effective_n_splits(merged_labels, n_splits, len(X_sub))
    protocol = StratifiedCVProtocol(n_splits=effective_n_splits, random_seed=random_seed)

    try:
        results = protocol.run(
            X_sub, y_sub, pipeline_factory,
            uncertainty_module=None, corr_module=corr_module,
            stratify_labels=merged_labels, verbose=False,
        )
        summary = results['summary']
        n_splits_used = effective_n_splits
    except Exception as e:
        print(f"    warning: CV failed ({e}), returning NaN")
        summary = {'rmse_mean': np.nan, 'rmse_std': np.nan, 'r2_mean': np.nan, 'r2_std': np.nan}
        n_splits_used = 0

    return {
        'summary': summary,
        'n_splits_used': n_splits_used,
        'n_effective_bins': len(np.unique(merged_labels)),
        'n_bins_raw': n_bins_raw,
        'n_bins_merged': n_bins_merged,
        'bins_merged': bins_merged,
    }


def create_pipeline_factory(
    data_module_name: str,
    model_module_name: str,
    corr_module_name: str,
    base_seed: int,
    feature_set: str,
    base_config: dict,
):
    def factory(seed: Optional[int] = None):
        seed_value = base_seed if seed is None else seed
        data_params = build_data_params(base_config, data_module_name, feature_set, seed_value)
        model_params = build_model_params(base_config, model_module_name, seed_value)
        data_mod = get_data_module(data_module_name, **data_params)
        model_mod = get_model_module(model_module_name, **model_params)
        corr_mod = get_correction_module(corr_module_name)
        return Pipeline(data_mod, model_mod, corr_mod)
    return factory


def _run_single_lc_task(
    repeat_id: int,
    fraction: float,
    model_name: str,
    target_name: str,
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    strat_labels_sub: np.ndarray,
    n_train_full: int,
    n_train_sub: int,
    data_module_name: str,
    corr_module_name: str,
    n_splits: int,
    target_seed: int,
    feature_set: str,
    base_config: dict,
    is_worker: bool = False,
) -> Dict[str, Any]:
    """Run one (repeat × fraction × model × target) task in an isolated worker.

    When dispatched as a parallel worker (``is_worker=True``), forces
    ``ML_FOLD_WORKERS=1`` to prevent nested process-pool conflicts.
    Sequential callers must leave ``is_worker`` at its default (``False``) to
    preserve the parent-process parallelism settings.
    """
    if is_worker:
        import os as _os
        _os.environ["ML_PARALLEL_WORKER"] = "1"
        _os.environ["ML_FOLD_WORKERS"] = "1"

    pipeline_factory = create_pipeline_factory(
        data_module_name, model_name, corr_module_name, target_seed, feature_set, base_config
    )
    corr_module = get_correction_module(corr_module_name)
    cv_result = run_cv_for_subsample(
        X_sub, y_sub, strat_labels_sub, pipeline_factory, corr_module, n_splits, target_seed
    )
    summary = cv_result['summary']

    return {
        'fraction': fraction,
        'repeat': repeat_id,
        'model': model_name,
        'target': target_name,
        'feature_set': feature_set,
        'data_module': data_module_name,
        'correction_module': corr_module_name,
        'n_splits_used': cv_result['n_splits_used'],
        'n_splits_requested': n_splits,
        'n_effective_bins': cv_result['n_effective_bins'],
        'n_bins_raw': cv_result['n_bins_raw'],
        'n_bins_merged': cv_result['n_bins_merged'],
        'bins_merged': cv_result['bins_merged'],
        'n_train_full': n_train_full,
        'n_train_sub': n_train_sub,
        'rmse_mean': summary.get('rmse_mean', np.nan),
        'rmse_std': summary.get('rmse_std', np.nan),
        'r2_mean': summary.get('r2_mean', np.nan),
        'r2_std': summary.get('r2_std', np.nan),
        'mae_mean': summary.get('mae_mean', np.nan),
        'mbe_mean': summary.get('mbe_mean', np.nan),
        'slope_mean': summary.get('slope_mean', np.nan),
    }


# ============================================================
# Nested subsampling
# ============================================================

def create_nested_subsamples(
    indices: np.ndarray,
    strat_labels: np.ndarray,
    fractions: List[float],
    seed: int,
) -> Dict[float, np.ndarray]:
    """Create nested stratified subsamples at each fraction of the full dataset."""
    rng = np.random.RandomState(seed)
    bin_to_indices = {}
    for bin_id in np.unique(strat_labels):
        local_idx = np.where(strat_labels == bin_id)[0]
        if local_idx.size == 0:
            continue
        bin_to_indices[int(bin_id)] = indices[rng.permutation(local_idx)]

    nested_indices = {}
    for frac in sorted(fractions):
        if frac <= 0:
            raise ValueError(f"fraction must be > 0, got {frac}")
        selected = []
        for bin_id in sorted(bin_to_indices.keys()):
            bin_indices = bin_to_indices[bin_id]
            n_take = max(1, min(int(np.ceil(len(bin_indices) * frac)), len(bin_indices)))
            selected.extend(bin_indices[:n_take].tolist())
        nested_indices[frac] = np.array(selected, dtype=int)

    return nested_indices


# ============================================================
# Main learning-curve runner
# ============================================================

def run_learning_curve(
    X_train_full: np.ndarray,
    y_train_full: np.ndarray,
    strat_labels_train_full: np.ndarray,
    fractions: List[float],
    repeat_ids: List[int],
    models: List[str],
    targets: List[Tuple[str, np.ndarray]],
    data_module_name: str,
    corr_module_name: str,
    n_splits: int,
    base_seed: int,
    feature_set: str,
    task_workers: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    n_train_full = len(X_train_full)
    train_full_indices = np.arange(n_train_full)

    # Pre-compute all tasks: subsample X/y arrays so each worker receives
    # only its own small slice (avoids serializing X_train_full per task).
    tasks = []
    for repeat_id in repeat_ids:
        repeat_seed = base_seed + repeat_id
        nested_indices = create_nested_subsamples(
            train_full_indices, strat_labels_train_full, fractions, seed=repeat_seed
        )
        for fraction in fractions:
            sub_indices = nested_indices[fraction]
            X_sub = X_train_full[sub_indices]
            strat_labels_sub = strat_labels_train_full[sub_indices]
            n_train_sub = len(sub_indices)
            for model_name in models:
                for target_name, y_full in targets:
                    target_seed = derive_target_seed(repeat_seed, target_name)
                    y_sub = y_full[sub_indices]
                    tasks.append(dict(
                        repeat_id=repeat_id,
                        fraction=fraction,
                        model_name=model_name,
                        target_name=target_name,
                        X_sub=X_sub,
                        y_sub=y_sub,
                        strat_labels_sub=strat_labels_sub,
                        n_train_full=n_train_full,
                        n_train_sub=n_train_sub,
                        data_module_name=data_module_name,
                        corr_module_name=corr_module_name,
                        n_splits=n_splits,
                        target_seed=target_seed,
                        feature_set=feature_set,
                        base_config=BASE_CONFIG,
                    ))

    total = len(tasks)
    if verbose:
        print(f"  Total tasks: {total}"
              f" ({len(repeat_ids)} repeats × {len(fractions)} fractions"
              f" × {len(models)} models × {len(targets)} targets)")

    if task_workers > 1:
        from joblib import Parallel, delayed
        if verbose:
            print(f"  Running in parallel (workers={task_workers}, backend=loky) ...")
        records = Parallel(n_jobs=task_workers, backend="loky")(
            delayed(_run_single_lc_task)(**t, is_worker=True) for t in tasks
        )
    else:
        records = []
        for i, t in enumerate(tasks):
            records.append(_run_single_lc_task(**t))
            if verbose and (i + 1) % 50 == 0:
                print(f"\r  Progress: {(i + 1) / total * 100:5.1f}%  ({i+1}/{total})", end="")
        if verbose:
            print()

    return pd.DataFrame(records)


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Learning curve tool for model-complexity benefit analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (ERT + CatBoost + Stacking, Liquid, 30 repeats, auto workers)
  python tools/run_learning_curve.py

  # Use NoLiquid feature set
  python tools/run_learning_curve.py --feature-set noliq

  # ERT only, quick test
  python tools/run_learning_curve.py --models ert --repeats 5 --n-splits 5

  # Force sequential
  python tools/run_learning_curve.py --task-workers 1
"""
    )

    parser.add_argument('--feature-set', type=str, default='liq',
                        choices=['liq', 'noliq', 'Liquid', 'NoLiquid'],
                        help='Feature set (default: liq = Liquid)')
    parser.add_argument('--data-module', type=str, default='augmented',
                        choices=['raw', 'balanced', 'augmented'],
                        help='Data module (default: augmented)')
    parser.add_argument('--corr-module', type=str, default='none',
                        choices=['none', 'segmented'],
                        help='Correction module (default: none)')
    parser.add_argument('--models', nargs='+', type=str,
                        default=['ert', 'catboost', 'stacking'],
                        choices=['ert', 'catboost', 'stacking'],
                        help='Model list (default: ert catboost stacking)')
    parser.add_argument('--fractions', nargs='+', type=float,
                        default=[0.2, 0.4, 0.6, 0.8, 1.0],
                        help='Sampling fractions (default: 0.2 0.4 0.6 0.8 1.0)')
    parser.add_argument('--repeats', type=int, default=30,
                        help='Number of repeats per fraction (default: 30)')
    parser.add_argument('--repeat-start', type=int, default=None,
                        help='repeat start (inclusive), for segmented runs')
    parser.add_argument('--repeat-end', type=int, default=None,
                        help='repeat end (inclusive), for segmented runs')
    parser.add_argument('--merge-dir', type=str, default=None,
                        help='merge runs from directory and finalize summary')
    parser.add_argument('--n-splits', type=int, default=10,
                        help='Number of CV folds (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: results/learning_curve)')
    parser.add_argument('--task-workers', type=int, default=None,
                        help='Parallel task workers (default: auto = max(1, cpu_count // n_jobs)).'
                             ' Set to 1 to force sequential.')

    args = parser.parse_args()

    # Auto-detect task_workers using the same formula as fold/repeat workers
    if args.task_workers is None:
        from src.runtime import get_n_jobs
        cpu_count = os.cpu_count() or 1
        n_jobs = get_n_jobs()
        n_per_worker = n_jobs if n_jobs > 0 else cpu_count
        task_workers = max(1, cpu_count // n_per_worker)
    else:
        task_workers = max(1, args.task_workers)

    if (args.repeat_start is None) ^ (args.repeat_end is None):
        raise ValueError("repeat-start and repeat-end must be provided together")
    if args.repeat_start is None and args.repeat_end is None:
        repeat_start = 0
        repeat_end = args.repeats - 1
        segmented = False
    else:
        repeat_start = args.repeat_start
        repeat_end = args.repeat_end
        if repeat_end < repeat_start:
            raise ValueError("repeat-end must be >= repeat-start")
        segmented = True

    repeat_ids = list(range(repeat_start, repeat_end + 1))
    feature_set = 'Liquid' if args.feature_set.lower() in ['liq', 'liquid'] else 'NoLiquid'
    output_dir = args.output_dir if args.output_dir else os.path.join(BASE_CONFIG['output_dir'], OUTPUT_SUBDIR)
    os.makedirs(output_dir, exist_ok=True)

    if args.merge_dir:
        print("=" * 70)
        print("Learning Curve - finalize from runs")
        print("=" * 70)
        runs_df = merge_runs_files(args.merge_dir)
        summary_df = summarize_runs(runs_df, args.n_splits)
        runs_df.to_csv(os.path.join(output_dir, "learning_curve_runs.csv"), index=False)
        summary_df.to_csv(os.path.join(output_dir, "learning_curve_summary.csv"), index=False)
        print(f"  runs:    {os.path.join(output_dir, 'learning_curve_runs.csv')}")
        print(f"  summary: {os.path.join(output_dir, 'learning_curve_summary.csv')}")
        return runs_df, summary_df

    print("=" * 70)
    print("Learning Curve Tool - Model-Complexity Benefit Analysis")
    print("=" * 70)
    print(f"  Feature set:   {feature_set}")
    print(f"  Data module:   {args.data_module}")
    print(f"  Models:        {args.models}")
    print(f"  Fractions:     {args.fractions}")
    print(f"  Repeats:       {len(repeat_ids)} ({repeat_start}–{repeat_end})")
    print(f"  CV folds:      {args.n_splits}")
    print(f"  Task workers:  {task_workers}")
    print(f"  Output:        {output_dir}")
    print("=" * 70)

    print("\n[1/3] Loading data...")
    X, y_T, y_P = load_data(BASE_CONFIG, feature_set=feature_set)

    print("\n[2/3] Preparing data split...")
    split_data = prepare_splits(X, y_T, y_P, {'random_seed': args.seed})
    train_idx = split_data['train_idx']
    tp_bins_train = split_data['tp_bins_train']

    X_train_full = X[train_idx]
    y_T_train_full = y_T[train_idx]
    y_P_train_full = y_P[train_idx]
    print(f"  Training set: {len(train_idx)} samples, {len(np.unique(tp_bins_train))} P-T bins")

    print("\n[3/3] Running learning-curve experiments...")
    start_time = time.time()

    runs_df = run_learning_curve(
        X_train_full=X_train_full,
        y_train_full=y_T_train_full,
        strat_labels_train_full=tp_bins_train,
        fractions=args.fractions,
        repeat_ids=repeat_ids,
        models=args.models,
        targets=[('T', y_T_train_full), ('P', y_P_train_full)],
        data_module_name=args.data_module,
        corr_module_name=args.corr_module,
        n_splits=args.n_splits,
        base_seed=args.seed,
        feature_set=feature_set,
        task_workers=task_workers,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"  Finished in {elapsed:.1f}s")

    if segmented:
        runs_name = f"learning_curve_runs_rep_{repeat_start:03d}_{repeat_end:03d}.csv"
        runs_df.to_csv(os.path.join(output_dir, runs_name), index=False)
        print(f"  runs: {os.path.join(output_dir, runs_name)}")
        print("  segment mode: use --merge-dir to finalize")
        return runs_df, None

    runs_df.to_csv(os.path.join(output_dir, "learning_curve_runs.csv"), index=False)
    summary_df = summarize_runs(runs_df, args.n_splits)
    summary_df.to_csv(os.path.join(output_dir, "learning_curve_summary.csv"), index=False)
    print(f"  runs:    {os.path.join(output_dir, 'learning_curve_runs.csv')}")
    print(f"  summary: {os.path.join(output_dir, 'learning_curve_summary.csv')}")
    print("  Figures: python tools/plot_offline_figures.py")

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    for target in ['T', 'P']:
        print(f"\nTarget: {target}")
        target_df = summary_df[summary_df['target'] == target]
        if target_df.empty:
            continue
        display_cols = ['fraction', 'model', 'n_train_sub_mean',
                        'rmse_mean_of_repeats', 'rmse_std_of_repeats', 'r2_mean_of_repeats']
        available_cols = [c for c in display_cols if c in target_df.columns]
        print(target_df[available_cols].to_string(index=False))

    return runs_df, summary_df


if __name__ == '__main__':
    logger = init_tool_logging("learning_curve")
    try:
        main()
    except Exception:
        logger.exception("Learning-curve run failed")
        raise
