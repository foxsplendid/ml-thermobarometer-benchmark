# -*- coding: utf-8 -*-
"""Run the benchmark experiment matrix and quick-test workflow."""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*divide by zero.*')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# ============================================================
from config import get_config_dict
from src.experiment_params import BASE_CONFIGS, build_exp_id, build_model_params, build_data_params
from src.logger import setup_logging, get_logger

CONFIG = get_config_dict()


# ============================================================
# ============================================================

def get_experiment_configs(tier='full'):
    """get_experiment_configs function.

    tier='fast' (H4) overlays smoke-test model params; its only caller is
    run_quick_test. main() has no tier parameter, so the fast tier is
    unreachable for the E01-E12 benchmark.
    """
    from src.protocol import ExperimentConfig

    final_configs = []
    for base in BASE_CONFIGS:
        for fset in ['NoLiquid', 'Liquid']:
            exp_id = build_exp_id(base['data'], base['model'], base['corr'], fset)
            final_configs.append(ExperimentConfig(
                exp_id=exp_id,
                data_module_name=base['data'],
                model_module_name=base['model'],
                corr_module_name=base['corr'],
                feature_set=fset,
                # random_seed omitted: protocol._apply_seed injects per-target seed at runtime
                data_params=build_data_params(CONFIG, base['data'], fset),
                model_params=build_model_params(CONFIG, base['model'], tier=tier),
                run_uncertainty=False,
            ))

    return final_configs


# ============================================================
# ============================================================

def load_data(config, feature_set='Liquid'):
    """load_data function."""
    print(f"Loading data (feature set: {feature_set})...")

    df = pd.read_csv(config['data_path'], encoding=config['data_encoding'])

    dirty_patterns = ['Unnamed:', '\ufeff']
    cols_to_drop = [col for col in df.columns if any(p in col for p in dirty_patterns)]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    feature_cols = config['feature_sets'][feature_set]
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    X = df[feature_cols].values.astype(np.float64)
    y_T = df[config['target_T']].values.astype(np.float64)
    y_P = df[config['target_P']].values.astype(np.float64)

    # S4 (V8): missing measurements must not be silently zero-filled (V7 used
    # np.nan_to_num here). Exact zeros in the source CSV are the upstream
    # convention for not-analyzed / below-detection oxides (e.g. Cr2O3.cpx is
    # 0 in 1102/2079 rows) and are deliberately kept as numeric zero; a NaN,
    # however, means the dataset changed and needs an explicit imputation
    # decision (trees/CatBoost tolerate NaN natively, Ridge and StandardScaler
    # semantics do not), so fail loudly instead of guessing.
    nan_counts = np.isnan(X).sum(axis=0)
    if nan_counts.any():
        bad = {feature_cols[i]: int(c) for i, c in enumerate(nan_counts) if c > 0}
        raise ValueError(
            f"NaN found in feature columns {bad}; this benchmark has no "
            "imputation policy — clean the data or add one explicitly."
        )
    if np.isnan(y_T).any() or np.isnan(y_P).any():
        raise ValueError(
            f"NaN found in targets: {config['target_T']}={int(np.isnan(y_T).sum())}, "
            f"{config['target_P']}={int(np.isnan(y_P).sum())}."
        )

    print(f"  Feature set: {feature_set} ({len(feature_cols)} features)")
    print(f"  Data shape: X={X.shape}")
    print(f"  Temperature range: {y_T.min():.0f} - {y_T.max():.0f} C")
    print(f"  Pressure range: {y_P.min():.2f} - {y_P.max():.2f} kbar")
    
    return X, y_T, y_P


# ============================================================
# ============================================================

def prepare_splits(X, y_T, y_P, config):
    """prepare_splits function."""
    from src.splitters import compute_pt_edges, assign_pt_bins, select_test_indices

    print("\nPreparing test split (P-T grid sampling)...")

    bins = compute_pt_edges(y_T, y_P)
    tp_bins = assign_pt_bins(y_T, y_P, bins)

    test_idx = select_test_indices(
        tp_bins,
        random_state=config['random_seed']
    )

    train_mask = np.ones(len(X), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    test_tp_bins = tp_bins[test_idx]
    unique_bins, bin_counts = np.unique(test_tp_bins, return_counts=True)

    print(f"  Test set size: {len(test_idx)} (sampled from {len(np.unique(tp_bins))} non-empty P-T bins)")
    print(f"  P-T bin coverage: {len(unique_bins)}/{len(np.unique(tp_bins))}")
    print(f"  Samples per bin: min={bin_counts.min()}, max={bin_counts.max()}, mean={bin_counts.mean():.2f}")

    split_info = {
        'test_indices': test_idx.tolist(),
        'test_size': int(len(test_idx)),
        'p_edges': bins.p_edges.tolist(),
        't_edges': bins.t_edges.tolist(),
        'n_pt_bins': len(unique_bins),
    }

    return {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'tp_bins_train': tp_bins[train_idx],
        'split_info': split_info,
    }


# ============================================================
# ============================================================

def main(reuse_fits=False):
    """main function.

    reuse_fits=True (H6, opt-in) reuses outer-CV fits and the persisted
    full fit between experiments that differ only in correction module
    (E10-E12 consume E07-E09); training_time columns then carry the
    producer's measurements.
    """
    print("=" * 70)
    print("Chapter 3 Benchmark Protocol - Modular Validation Framework")
    print("=" * 70)

    X_liquid, y_T, y_P = load_data(CONFIG, feature_set='Liquid')

    split_data = prepare_splits(X_liquid, y_T, y_P, CONFIG)
    split_info = split_data['split_info']
    print(f"Test size: {split_info['test_size']}, P-T bins: {split_info['n_pt_bins']}")

    train_idx = split_data['train_idx']
    test_idx = split_data['test_idx']
    tp_bins_train = split_data['tp_bins_train']

    configs = get_experiment_configs()
    print(f"\nNumber of experiments: {len(configs)} (12 base configs x 2 feature sets)")

    all_results = []

    for feature_set in ['NoLiquid', 'Liquid']:
        print(f"\n{'='*70}")
        print(f"Running feature set: {feature_set}")
        print(f"{'='*70}")

        X, y_T, y_P = load_data(CONFIG, feature_set=feature_set)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_T_train = y_T[train_idx]
        y_T_test = y_T[test_idx]
        y_P_train = y_P[train_idx]
        y_P_test = y_P[test_idx]

        feature_configs = [c for c in configs if c.feature_set == feature_set]
        print(f"Experiments in this feature set: {len(feature_configs)}")

        from src.protocol import ExperimentMatrix

        matrix = ExperimentMatrix(
            X=X_train,
            y_T=y_T_train,
            y_P=y_P_train,
            output_dir=CONFIG['output_dir'],
        )

        summary_df = matrix.run_experiments(
            configs=feature_configs,
            n_splits=CONFIG['n_splits'],
            stratify_labels=tp_bins_train,
            X_test=X_test,
            y_T_test=y_T_test,
            y_P_test=y_P_test,
            random_seed=CONFIG['random_seed'],
            verbose=True,
            reuse_identical_fits=reuse_fits,
        )

        all_results.append(summary_df)

    summary_df = pd.concat(all_results, ignore_index=True)

    matrix.compute_effect_table(summary_df)

    matrix.save_config(configs, extra_info={
        'n_splits': CONFIG['n_splits'],
        'random_seed': CONFIG['random_seed'],
        'test_split': split_info,
        'feature_sets': list(CONFIG['feature_sets'].keys()),
        'n_features_by_feature_set': {k: len(v) for k, v in CONFIG['feature_sets'].items()},
    })

    print("\n" + "=" * 70)
    print("Experiments completed. Summary:")
    print("=" * 70)

    display_cols = ['exp_id', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

    print(f"\nOutput directory: {CONFIG['output_dir']}")
    print("=" * 70)

    return summary_df


def run_quick_test(tier='fast'):
    """run_quick_test function.

    tier (H4): 'fast' (default) uses FAST_TIER_OVERRIDES for ~10x cheaper
    smoke fits; 'full' reproduces the pre-H4 quick-test behavior.
    """
    print("=" * 70)
    print(f"Quick-test mode (2 folds, 4 experiments, model tier: {tier})")
    print("=" * 70)

    test_config = CONFIG.copy()
    test_config['output_dir'] = CONFIG['test_output_dir']

    X_liquid, y_T, y_P = load_data(test_config, feature_set='Liquid')

    split_data = prepare_splits(X_liquid, y_T, y_P, test_config)
    split_info = split_data['split_info']
    print(f"Test size: {split_info['test_size']}, P-T bins: {split_info['n_pt_bins']}")

    train_idx = split_data['train_idx']
    test_idx = split_data['test_idx']
    tp_bins_train = split_data['tp_bins_train']

    all_configs = get_experiment_configs(tier=tier)
    test_configs = [c for c in all_configs if any(c.exp_id.startswith(f"E{i:02d}") for i in [1, 2])]
    print(f"\nNumber of quick-test experiments: {len(test_configs)}")

    all_results = []

    for feature_set in ['NoLiquid', 'Liquid']:
        print(f"\n{'='*70}")
        print(f"Running feature set: {feature_set}")
        print(f"{'='*70}")

        X, y_T, y_P = load_data(test_config, feature_set=feature_set)

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_T_train = y_T[train_idx]
        y_T_test = y_T[test_idx]
        y_P_train = y_P[train_idx]
        y_P_test = y_P[test_idx]

        feature_configs = [c for c in test_configs if c.feature_set == feature_set]
        print(f"Experiments in this feature set: {len(feature_configs)}")

        from src.protocol import ExperimentMatrix

        matrix = ExperimentMatrix(
            X=X_train,
            y_T=y_T_train,
            y_P=y_P_train,
            output_dir=test_config['output_dir'],
        )

        summary_df = matrix.run_experiments(
            configs=feature_configs,
            n_splits=2,
            stratify_labels=tp_bins_train,
            X_test=X_test,
            y_T_test=y_T_test,
            y_P_test=y_P_test,
            random_seed=test_config['random_seed'],
            verbose=True,
        )

        all_results.append(summary_df)

    summary_df = pd.concat(all_results, ignore_index=True)

    matrix.save_config(test_configs, extra_info={
        'mode': 'quick_test',
        'model_tier': tier,
        'n_splits': 2,
        'random_seed': test_config['random_seed'],
        'test_split': split_info,
    })

    print("\n" + "=" * 70)
    print("Quick test completed.")
    print("=" * 70)

    display_cols = ['exp_id', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

    print(f"\nOutput directory: {test_config['output_dir']}")
    print("=" * 70)

    return summary_df


# ============================================================
# ============================================================

def _init_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"main_{timestamp}_{os.getpid()}.log"
    setup_logging(log_filename=log_filename)
    return get_logger(__name__)


def _print_runtime_banner():
    """Print hardware probe + effective parallel budget (M3.startup).

    Helps diagnose oversubscription when several CC / experiment processes
    run on the same machine. Set ML_RESERVE_CORES and/or ML_OUTER_PROCS
    to leave headroom for siblings.
    """
    try:
        from src.runtime import runtime_summary_str, suggest_n_jobs
        print(runtime_summary_str())
        print(
            f"  effective n_jobs: model={suggest_n_jobs('model')} "
            f"inner_loop={suggest_n_jobs('inner_loop')} "
            f"cross_proc={suggest_n_jobs('cross_proc')}"
        )
        reserve = os.environ.get('ML_RESERVE_CORES', '0')
        cap = os.environ.get('ML_N_JOBS', '(none)')
        outer = os.environ.get('ML_OUTER_PROCS', '1')
        stacking_par = os.environ.get('ML_STACKING_PARALLEL', '(none)')
        print(f"  env: ML_RESERVE_CORES={reserve} ML_N_JOBS={cap} "
              f"ML_OUTER_PROCS={outer} ML_STACKING_PARALLEL={stacking_par}")
    except Exception as exc:
        print(f"  (runtime banner failed: {exc})")


if __name__ == '__main__':
    import argparse

    logger = _init_logging()
    try:
        parser = argparse.ArgumentParser(description='Benchmark Protocol')
        parser.add_argument('--test', action='store_true', help='Run quick test')
        parser.add_argument('--tier', choices=['fast', 'full'], default=None,
                            help='Model parameter tier for --test mode (default: fast). '
                                 'H4: fast tier never applies to the full benchmark.')
        parser.add_argument('--reuse-fits', action='store_true',
                            help='H6 (opt-in): reuse outer-CV fits between experiments that '
                                 'differ only in correction module (E10-E12 reuse E07-E09). '
                                 'training_time columns then carry producer measurements.')
        args = parser.parse_args()

        if args.tier is not None and not args.test:
            parser.error('--tier is only valid together with --test')
        if args.reuse_fits and args.test:
            parser.error('--reuse-fits applies to the full benchmark only (E01/E02 have no '
                         'correction-only siblings, so the quick test never reuses fits)')

        if args.test:
            # SPEC §11.4: quick test caps threads at 4 unless the user chose.
            os.environ.setdefault('ML_N_JOBS', '4')

        _print_runtime_banner()

        if args.test:
            run_quick_test(tier=args.tier or 'fast')
        else:
            main(reuse_fits=args.reuse_fits)
    except Exception:
        logger.exception("Main entrypoint failed")
        raise

