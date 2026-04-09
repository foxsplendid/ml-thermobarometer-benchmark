# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import get_config_dict
from main import load_data, prepare_splits
from src.correction_modules import get_correction_module
from src.data_modules import get_data_module
from src.experiment_params import build_data_params, build_model_params
from src.model_modules import get_model_module
from src.protocol import (
    Pipeline,
    StratifiedCVProtocol,
    _merge_sparse_bins,
    _get_effective_n_splits,
    _derive_target_seed,
)
from src.logger import setup_logging, get_logger

BASE_CONFIG = get_config_dict()
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
# ============================================================


def get_effective_n_splits(n_samples: int, strat_labels: np.ndarray, requested_n_splits: int) -> int:
    """get_effective_n_splits function."""
    return _get_effective_n_splits(strat_labels, requested_n_splits, n_samples)


def merge_sparse_bins(strat_labels: np.ndarray, min_samples_per_bin: int = 10) -> np.ndarray:
    """merge_sparse_bins function."""
    return _merge_sparse_bins(strat_labels, min_samples_per_bin)


def run_cv_for_subsample(
    X_sub: np.ndarray,
    y_sub: np.ndarray,
    strat_labels_sub: np.ndarray,
    pipeline_factory: Callable[..., Pipeline],
    corr_module,
    n_splits: int,
    random_seed: int,
) -> Dict[str, Any]:
    """run_cv_for_subsample function."""
    n_bins_raw = len(np.unique(strat_labels_sub)) if strat_labels_sub is not None else 0

    merged_labels = merge_sparse_bins(strat_labels_sub, min_samples_per_bin=n_splits)
    n_bins_merged = len(np.unique(merged_labels)) if merged_labels is not None else 0
    bins_merged = int(n_bins_raw > 0 and n_bins_merged < n_bins_raw)

    effective_n_splits = get_effective_n_splits(len(X_sub), merged_labels, n_splits)

    protocol = StratifiedCVProtocol(n_splits=effective_n_splits, random_seed=random_seed)

    try:
        results = protocol.run(
            X_sub, y_sub,
            pipeline_factory,
            uncertainty_module=None,
            corr_module=corr_module,
            stratify_labels=merged_labels,
            verbose=False
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
    """create_pipeline_factory function."""
    def factory(seed: Optional[int] = None):
        seed_value = base_seed if seed is None else seed
        data_params = build_data_params(base_config, data_module_name, feature_set, seed_value)
        model_params = build_model_params(base_config, model_module_name, seed_value)
        data_mod = get_data_module(data_module_name, **data_params)
        model_mod = get_model_module(model_module_name, **model_params)
        corr_mod = get_correction_module(corr_module_name)
        return Pipeline(data_mod, model_mod, corr_mod)
    return factory


# ============================================================
# ============================================================

def create_nested_subsamples(
    indices: np.ndarray,
    strat_labels: np.ndarray,
    fractions: List[float],
    seed: int
) -> Dict[float, np.ndarray]:
    """
    Create nested subsamples where fractions are absolute proportions
    of the full dataset, while keeping per-bin stratification.
    """
    rng = np.random.RandomState(seed)

    # Build per-bin shuffled index lists based on the full dataset
    bin_to_indices = {}
    for bin_id in np.unique(strat_labels):
        local_idx = np.where(strat_labels == bin_id)[0]
        if local_idx.size == 0:
            continue
        shuffled_local = rng.permutation(local_idx)
        bin_to_indices[int(bin_id)] = indices[shuffled_local]

    nested_indices = {}
    for frac in sorted(fractions):
        if frac <= 0:
            raise ValueError(f"fraction must be > 0, got {frac}")
        selected = []
        for bin_id in sorted(bin_to_indices.keys()):
            bin_indices = bin_to_indices[bin_id]
            n_full = len(bin_indices)
            n_take = int(np.ceil(n_full * frac))
            n_take = max(1, n_take)
            n_take = min(n_take, n_full)
            selected.extend(bin_indices[:n_take].tolist())
        nested_indices[frac] = np.array(selected, dtype=int)

    return nested_indices


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
    output_dir: Optional[str] = None,
    resume: bool = False,
    checkpoint_interval: int = 20,
    verbose: bool = True,
) -> pd.DataFrame:
    """run_learning_curve function."""
    runs_records = []
    n_train_full = len(X_train_full)
    train_full_indices = np.arange(n_train_full)

    completed_tasks = set()
    if output_dir and resume:
        import re
        checkpoint_files = []
        for fname in os.listdir(output_dir):
            if fname.startswith("learning_curve_checkpoint"):
                fpath = os.path.join(output_dir, fname)
                m = re.match(r"learning_curve_checkpoint_task(\d+)\.csv", fname)
                if m:
                    checkpoint_files.append((int(m.group(1)), fpath))
                elif fname == "learning_curve_checkpoint.csv":
                    checkpoint_files.append((0, fpath))

        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            latest_checkpoint = checkpoint_files[0][1]
            try:
                existing_df = pd.read_csv(latest_checkpoint)
                for _, row in existing_df.iterrows():
                    key = (row['repeat'], row['fraction'], row['model'], row['target'])
                    completed_tasks.add(key)
                    runs_records.append(row.to_dict())
                if verbose:
                    print(f"  [Resume] loaded {len(completed_tasks)} completed tasks from {os.path.basename(latest_checkpoint)}")
            except Exception as e:
                if verbose:
                    print(f"  [Resume] failed to load checkpoint: {e}")

    total_iterations = len(fractions) * len(repeat_ids) * len(models) * len(targets)
    current_iter = 0
    new_task_count = 0

    for repeat_id in repeat_ids:
        repeat_seed = base_seed + repeat_id

        nested_indices = create_nested_subsamples(
            train_full_indices,
            strat_labels_train_full,
            fractions,
            seed=repeat_seed
        )

        for fraction in fractions:
            sub_indices = nested_indices[fraction]

            X_sub = X_train_full[sub_indices]
            strat_labels_sub = strat_labels_train_full[sub_indices]
            n_train_sub = len(sub_indices)

            for model_name in models:
                corr_module = get_correction_module(corr_module_name)

                for target_name, y_full in targets:
                    target_seed = _derive_target_seed(repeat_seed, target_name)
                    pipeline_factory = create_pipeline_factory(
                        data_module_name,
                        model_name,
                        corr_module_name,
                        target_seed,
                        feature_set,
                        BASE_CONFIG,
                    )
                    current_iter += 1

                    task_key = (repeat_id, fraction, model_name, target_name)
                    if task_key in completed_tasks:
                        if verbose:
                            progress = current_iter / total_iterations * 100
                            print(f"\r  Progress: {progress:5.1f}% | [skip] rep={repeat_id:03d} frac={fraction:.1f} model={model_name:8s} target={target_name}", end="")
                        continue

                    y_sub = y_full[sub_indices]

                    if verbose:
                        progress = current_iter / total_iterations * 100
                        print(f"\r  Progress: {progress:5.1f}% | rep={repeat_id:03d} frac={fraction:.1f} model={model_name:8s} target={target_name}", end="")

                    cv_result = run_cv_for_subsample(
                        X_sub, y_sub, strat_labels_sub,
                        pipeline_factory, corr_module, n_splits, target_seed
                    )

                    summary = cv_result['summary']

                    record = {
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
                    runs_records.append(record)
                    new_task_count += 1

                    if output_dir and checkpoint_interval > 0 and new_task_count % checkpoint_interval == 0:
                        checkpoint_df = pd.DataFrame(runs_records)
                        checkpoint_file = os.path.join(output_dir, f"learning_curve_checkpoint_task{new_task_count:05d}.csv")
                        checkpoint_df.to_csv(checkpoint_file, index=False)
                        if verbose:
                            print(f"\n  [Checkpoint] task={new_task_count} saved {len(runs_records)} records -> {os.path.basename(checkpoint_file)}")

    if verbose:
        print()

    if output_dir and new_task_count > 0:
        checkpoint_df = pd.DataFrame(runs_records)
        checkpoint_file = os.path.join(output_dir, "learning_curve_checkpoint.csv")
        checkpoint_df.to_csv(checkpoint_file, index=False)

    runs_df = pd.DataFrame(runs_records)

    return runs_df


# ============================================================
# ============================================================

# ============================================================
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Learning curve tool for model-complexity benefit analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default run (ERT + Stacking, Liquid feature set, 30 repeats)
  python tools/run_learning_curve.py

  # Use NoLiquid feature set
  python tools/run_learning_curve.py --feature-set noliq

  # Add CatBoost model
  python tools/run_learning_curve.py --models ert stacking catboost

  # Reduce repeats for a quick test
  python tools/run_learning_curve.py --repeats 5 --n-splits 5

  # Custom sampling fractions
  python tools/run_learning_curve.py --fractions 0.1 0.3 0.5 0.7 0.9 1.0
"""
    )

    parser.add_argument('--feature-set', type=str, default='liq',
                        choices=['liq', 'noliq', 'Liquid', 'NoLiquid'],
                        help='Feature set: liq (Liquid) or noliq (NoLiquid)')
    parser.add_argument('--data-module', type=str, default='augmented',
                        choices=['raw', 'balanced', 'augmented'],
                        help='Data module (default: augmented, best-performing data strategy)')
    parser.add_argument('--corr-module', type=str, default='none',
                        choices=['none', 'segmented'],
                        help='Correction module (default: none)')

    parser.add_argument('--models', nargs='+', type=str, default=['ert', 'stacking'],
                        choices=['ert', 'catboost', 'stacking'],
                        help='Model list (default: ert stacking)')
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.2, 0.4, 0.6, 0.8, 1.0],
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

    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run (automatically skips completed tasks)')
    parser.add_argument('--checkpoint-interval', type=int, default=20,
                        help='Checkpoint interval (default: save every 20 tasks)')

    parser.add_argument('--no-plot', action='store_true',
                        help='Plot generation is handled by tools/plot_offline_figures.py')

    args = parser.parse_args()

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
        print(f"  merge_dir: {args.merge_dir}")
        print(f"  output_dir: {output_dir}")

        runs_df = merge_runs_files(args.merge_dir)
        summary_df = summarize_runs(runs_df, args.n_splits)

        runs_path = os.path.join(output_dir, "learning_curve_runs.csv")
        summary_path = os.path.join(output_dir, "learning_curve_summary.csv")
        runs_df.to_csv(runs_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        print(f"  runs: {runs_path}")
        print(f"  summary: {summary_path}")

        print("  Generate figures with tools/plot_offline_figures.py")

        return runs_df, summary_df

    print("=" * 70)
    print("Learning Curve Tool - Model-Complexity Benefit Analysis")
    print("=" * 70)
    print(f"  Feature set: {feature_set}")
    print(f"  Data module: {args.data_module}")
    print(f"  Correction module: {args.corr_module}")
    print(f"  Models: {args.models}")
    print(f"  Fractions: {args.fractions}")
    if segmented:
        print(f"  repeat range: {repeat_start}-{repeat_end}")
    else:
        print(f"  repeats: {args.repeats}")
    print(f"  CV folds: {args.n_splits}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output directory: {output_dir}")
    print("=" * 70)

    print("\n[1/4] Loading data...")
    X, y_T, y_P = load_data(BASE_CONFIG, feature_set=feature_set)

    print("\n[2/4] Preparing data split...")
    split_data = prepare_splits(X, y_T, y_P, {'random_seed': args.seed})
    train_idx = split_data['train_idx']
    tp_bins_train = split_data['tp_bins_train']

    X_train_full = X[train_idx]
    y_T_train_full = y_T[train_idx]
    y_P_train_full = y_P[train_idx]

    print(f"  Training set size: {len(train_idx)}")
    print(f"  Number of P-T bins: {len(np.unique(tp_bins_train))}")

    print("\n[3/4] Running learning-curve experiments...")
    start_time = time.time()

    targets = [('T', y_T_train_full), ('P', y_P_train_full)]

    runs_df = run_learning_curve(
        X_train_full=X_train_full,
        y_train_full=y_T_train_full,
        strat_labels_train_full=tp_bins_train,
        fractions=args.fractions,
        repeat_ids=repeat_ids,
        models=args.models,
        targets=targets,
        data_module_name=args.data_module,
        corr_module_name=args.corr_module,
        n_splits=args.n_splits,
        base_seed=args.seed,
        feature_set=feature_set,
        output_dir=output_dir,
        resume=args.resume,
        checkpoint_interval=args.checkpoint_interval,
        verbose=True,
    )

    elapsed = time.time() - start_time
    print(f"\n  Experiment finished in {elapsed:.1f} s")

    print("\n[4/4] Saving results...")
    if segmented:
        runs_name = f"learning_curve_runs_rep_{repeat_start:03d}_{repeat_end:03d}.csv"
    else:
        runs_name = "learning_curve_runs.csv"

    runs_path = os.path.join(output_dir, runs_name)
    runs_df.to_csv(runs_path, index=False)
    print(f"  runs: {runs_path}")

    if segmented:
        print("  segment mode: summary/plots skipped (use --merge-dir to finalize)")
        return runs_df, None

    summary_df = summarize_runs(runs_df, args.n_splits)
    summary_path = os.path.join(output_dir, "learning_curve_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"  summary: {summary_path}")

    print("  Generate figures with tools/plot_offline_figures.py")

    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)

    for target in ['T', 'P']:
        print(f"\nTarget: {target}")
        target_df = summary_df[summary_df['target'] == target]
        if target_df.empty:
            continue

        display_cols = ['fraction', 'model', 'n_train_sub_mean', 'rmse_mean_of_repeats', 'rmse_std_of_repeats', 'r2_mean_of_repeats']
        available_cols = [c for c in display_cols if c in target_df.columns]
        print(target_df[available_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("Done")
    print("=" * 70)

    return runs_df, summary_df


if __name__ == '__main__':
    def _init_logging():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f"learning_curve_{timestamp}_{os.getpid()}.log"
        setup_logging(log_filename=log_filename)
        global logger
        logger = get_logger(__name__)

    _init_logging()
    try:
        main()
    except Exception:
        logger.exception("Learning-curve run failed")
        raise
