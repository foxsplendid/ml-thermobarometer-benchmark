# -*- coding: utf-8 -*-
"""Cross-validation protocols and experiment orchestration utilities."""

import os
import time
import logging
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .interfaces import DataModule, ModelModule, CorrectionModule, UncertaintyModule, DataModuleState
from .metrics import summarize_folds, compute_all_metrics

logger = logging.getLogger(__name__)

P_SEED_OFFSET = 1000

def _derive_target_seed(base_seed: int, target_name: str) -> int:
    """_derive_target_seed function."""
    return base_seed + (P_SEED_OFFSET if str(target_name).upper() == "P" else 0)


# ============================================================
# ============================================================

def _apply_seed(params: Dict[str, Any], keys: List[str], seed: int, force: bool = False) -> Dict[str, Any]:
    """_apply_seed function."""
    updated = dict(params)
    for key in keys:
        if force or key not in updated:
            updated[key] = seed
    return updated


def _call_pipeline_factory(factory: Callable, seed: int, fold_idx: int = 0) -> 'Pipeline':
    """_call_pipeline_factory function."""
    effective_seed = seed + fold_idx
    try:
        return factory(effective_seed)
    except TypeError:
        return factory()

# ============================================================
# ============================================================

class Pipeline:
    """Pipeline class."""
    
    def __init__(self,
                 data_module: DataModule,
                 model_module: ModelModule,
                 corr_module: CorrectionModule):
        self.data_module = data_module
        self.model_module = model_module
        self.corr_module = corr_module
        
        self._state: Optional[DataModuleState] = None
        self._model: Optional[Any] = None
        self._corr_model: Optional[Any] = None
        self._is_fitted = False
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            stratify_labels: Optional[np.ndarray] = None) -> 'Pipeline':
        """fit function."""
        X2, y2, weights, self._state = self.data_module.fit_transform(
            X_train, y_train
        )
        
        if stratify_labels is not None:
            if len(y2) > len(stratify_labels):
                n_aug = len(y2) // len(stratify_labels)
                stratify2 = np.tile(stratify_labels, n_aug)
            else:
                stratify2 = stratify_labels
        else:
            stratify2 = None
        
        self._model = self.model_module.fit(X2, y2, weights, stratify_labels=stratify2)
        
        self._is_fitted = True
        return self
    
    def predict(self,
                X: np.ndarray,
                apply_correction: bool = True) -> np.ndarray:
        """predict function."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")
        
        y_pred = self.model_module.predict(self._model, X)
        
        if apply_correction:
            y_pred = self.corr_module.apply(self._corr_model, y_pred)
        
        return y_pred
    
    def predict_raw(self,
                    X_raw: np.ndarray,
                    apply_correction: bool = True) -> np.ndarray:
        """predict_raw function."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")
        
        X_scaled, _ = self.data_module.transform(X_raw, self._state)
        
        return self.predict(X_scaled, apply_correction)
    
    def get_model(self) -> Any:
        """get_model function."""
        return self._model
    
    def get_correction_params(self) -> Dict[str, float]:
        """get_correction_params function."""
        return self.corr_module.get_correction_params(self._corr_model)
    
    def get_name(self) -> str:
        """get_name function."""
        return f"{self.data_module.get_name()}_{self.model_module.get_name()}_{self.corr_module.get_name()}"
    
    def set_correction(self, corr_module: CorrectionModule, corr_model: Any) -> None:
        """set_correction function."""
        self.corr_module = corr_module
        self._corr_model = corr_model

# ============================================================
# ============================================================



def _merge_sparse_bins(labels: np.ndarray,
                       min_samples_per_bin: int,
                       verbose: bool = False) -> np.ndarray:
    """_merge_sparse_bins function."""
    unique_bins, bin_counts = np.unique(labels, return_counts=True)
    merged = labels.copy()

    sparse_bins = unique_bins[bin_counts < min_samples_per_bin]
    if sparse_bins.size == 0:
        if verbose:
            logger.debug(f"Bin merge: no sparse bins (min_samples={min_samples_per_bin}), "
                        f"{len(unique_bins)} bins unchanged")
        return merged

    non_sparse_bins = unique_bins[bin_counts >= min_samples_per_bin]
    if non_sparse_bins.size == 0:
        if verbose:
            logger.warning(f"Bin merge: all {len(unique_bins)} bins are sparse, "
                          f"collapsing to single bin")
        return np.zeros_like(labels)

    merge_map = {}
    for sparse_bin in sparse_bins:
        distances = np.abs(non_sparse_bins - sparse_bin)
        nearest_bin = non_sparse_bins[np.argmin(distances)]
        merged[labels == sparse_bin] = nearest_bin
        merge_map[int(sparse_bin)] = int(nearest_bin)

    if verbose:
        n_merged = len(sparse_bins)
        n_remaining = len(non_sparse_bins)
        logger.info(f"Bin merge: {n_merged} sparse bins -> {n_remaining} bins "
                   f"(threshold={min_samples_per_bin})")
        if n_merged <= 10:
            for src, dst in merge_map.items():
                logger.debug(f"  bin {src} -> {dst}")

    return merged


def _get_effective_n_splits(labels: Optional[np.ndarray], requested: int, n_samples: int) -> int:
    """_get_effective_n_splits function."""
    if n_samples <= 1:
        return 2
    if labels is None:
        return max(2, min(requested, n_samples))
    _, bin_counts = np.unique(labels, return_counts=True)
    min_bin = int(bin_counts.min()) if bin_counts.size > 0 else 1
    effective = min(requested, min_bin, n_samples)
    return max(2, effective)


# ============================================================
# ============================================================

class StratifiedCVProtocol:
    """StratifiedCVProtocol class."""
    
    def __init__(self,
                 n_splits: int = 10,
                 random_seed: int = 42):
        self.n_splits = n_splits
        self.random_seed = random_seed
    
    def run(self,
            X: np.ndarray,
            y: np.ndarray,
            pipeline_factory: Callable[..., Pipeline],
            uncertainty_module: Optional[UncertaintyModule] = None,
            corr_module: Optional[CorrectionModule] = None,
            stratify_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """run function."""
        if stratify_labels is None:
            logger.warning(
                "stratify_labels=None: using plain KFold instead of StratifiedKFold; "
                "this may cause imbalanced folds."
            )
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X)
        else:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X, stratify_labels)

        oof_pred_raw = np.full(len(y), np.nan, dtype=np.float64)
        fold_records = []
        training_times = []

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{self.n_splits}: ", end="")

            start_time = time.time()

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            stratify_train = stratify_labels[train_idx] if stratify_labels is not None else None

            pipeline = _call_pipeline_factory(pipeline_factory, self.random_seed, fold_idx)

            pipeline.fit(X_train, y_train, stratify_labels=stratify_train)

            X_val_scaled, _ = pipeline.data_module.transform(X_val, pipeline._state)
            y_pred_raw = pipeline.predict(X_val_scaled, apply_correction=False)

            oof_pred_raw[val_idx] = y_pred_raw

            fold_time = time.time() - start_time
            training_times.append(fold_time)

            fold_records.append({
                'fold_id': fold_idx,
                'val_idx': val_idx,
                'y_val': y_val,
                'y_pred_raw': y_pred_raw,
                'pipeline': pipeline,
                'X_val': X_val,
                'training_time': fold_time,
            })

        if np.any(np.isnan(oof_pred_raw)):
            raise RuntimeError("OOF prediction contains NaN values.")

        if corr_module is None:
            from .correction_modules import NoCorrection
            corr_module = NoCorrection()

        corr_model = corr_module.fit(y, oof_pred_raw)

        fold_metrics = []
        all_predictions = []
        unc_fold_metrics = [] if uncertainty_module is not None else None

        if uncertainty_module is not None and verbose:
            print("  Running MC uncertainty across folds...")

        for record in fold_records:
            y_pred_corr = corr_module.apply(corr_model, record['y_pred_raw'])
            dist = None

            if uncertainty_module is not None:
                pipeline = record['pipeline']
                pipeline.set_correction(corr_module, corr_model)

                dist = uncertainty_module.predict_distribution(
                    pipeline, record['X_val'], fold_idx=record['fold_id']
                )
                y_pred_corr = dist.get('median', y_pred_corr)

                calib_metrics = uncertainty_module.compute_calibration_metrics(record['y_val'], dist)
                calib_metrics['fold_id'] = record['fold_id']
                unc_fold_metrics.append(calib_metrics)

            metrics = compute_all_metrics(record['y_val'], y_pred_corr, record['y_pred_raw'])
            metrics['fold_id'] = record['fold_id']
            metrics['training_time'] = record['training_time']
            fold_metrics.append(metrics)

            if verbose:
                print(f"RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.4f}")

            preds_payload = {
                'fold_id': record['fold_id'],
                'sample_idx': record['val_idx'],
                'y_true': record['y_val'],
                'y_pred_raw': record['y_pred_raw'],
                'y_pred_corr': y_pred_corr,
                'residual': record['y_val'] - y_pred_corr,
                'y_pred_p16': dist.get('p16') if dist is not None else np.nan,
                'y_pred_p84': dist.get('p84') if dist is not None else np.nan,
                'y_pred_median': dist.get('median', y_pred_corr) if dist is not None else np.nan,
            }
            all_predictions.append(pd.DataFrame(preds_payload))

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        summary = summarize_folds(fold_metrics)
        summary['total_training_time'] = sum(training_times)

        uncertainty_results = None
        if unc_fold_metrics is not None:
            unc_summary = summarize_folds(unc_fold_metrics)
            for k, v in unc_summary.items():
                summary[f"unc_{k}"] = v

            uncertainty_results = {
                'fold_metrics': pd.DataFrame(unc_fold_metrics),
                'summary': unc_summary,
            }

        return {
            'fold_metrics': pd.DataFrame(fold_metrics),
            'predictions': predictions_df,
            'summary': summary,
            'uncertainty': uncertainty_results,
            'corr_module': corr_module,
            'corr_model': corr_model,
            'fold_records': fold_records,
        }

# ============================================================
# ============================================================

@dataclass
class ExperimentConfig:
    """ExperimentConfig class."""
    exp_id: str
    data_module_name: str
    model_module_name: str
    corr_module_name: str
    feature_set: str = 'Liquid'
    data_params: Dict = field(default_factory=dict)
    model_params: Dict = field(default_factory=dict)
    corr_params: Dict = field(default_factory=dict)
    uncertainty_params: Dict = field(default_factory=dict)
    run_uncertainty: bool = False

class ExperimentMatrix:
    """ExperimentMatrix class."""
    
    def __init__(self,
                 X: np.ndarray,
                 y_T: np.ndarray,
                 y_P: np.ndarray,
                 output_dir: str = 'results',
                 target_names: Tuple[str, str] = ('T', 'P')):
        """__init__ function."""
        self.X = X
        self.y_T = y_T
        self.y_P = y_P
        self.output_dir = output_dir
        self.target_names = target_names
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_experiments(self,
                        configs: List[ExperimentConfig],
                        n_splits: int = 10,
                        stratify_labels: Optional[np.ndarray] = None,
                        X_test: Optional[np.ndarray] = None,
                        y_T_test: Optional[np.ndarray] = None,
                        y_P_test: Optional[np.ndarray] = None,
                        random_seed: int = 42,
                        verbose: bool = True) -> pd.DataFrame:
        """run_experiments function."""
        from .data_modules import get_data_module
        from .model_modules import get_model_module
        from .correction_modules import get_correction_module
        from .uncertainty_modules import MCUncertaintyEstimator
        
        all_results = []
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Experiment: {config.exp_id}")
            print(f"Config: {config.data_module_name} + {config.model_module_name} + {config.corr_module_name}")
            print(f"{'='*60}")

            exp_result = {
                'exp_id': config.exp_id,
                'data_module': config.data_module_name,
                'model_module': config.model_module_name,
                'corr_module': config.corr_module_name,
                'feature_set': config.feature_set,
            }
            
            def make_pipeline_factory(cfg):
                def factory(seed: Optional[int] = None):
                    seed_value = random_seed if seed is None else seed

                    data_params = _apply_seed(cfg.data_params, ['random_seed'], seed_value)
                    model_params = _apply_seed(cfg.model_params, ['random_seed'], seed_value)

                    data_mod = get_data_module(cfg.data_module_name, **data_params)
                    model_mod = get_model_module(cfg.model_module_name, **model_params)
                    corr_mod = get_correction_module(cfg.corr_module_name, **cfg.corr_params)
                    return Pipeline(data_mod, model_mod, corr_mod)
                return factory

            pipeline_factory = make_pipeline_factory(config)
            
            unc_params_base = None
            if config.run_uncertainty:
                unc_params = {}
                try:
                    from config import CONFIG as APP_CONFIG
                    unc_params.update({
                        'n_mc': APP_CONFIG.uncertainty.n_mc,
                        'percentiles': APP_CONFIG.uncertainty.percentiles,
                    })
                    feature_names = APP_CONFIG.data.feature_sets.get(config.feature_set)
                    if feature_names:
                        unc_params.setdefault('feature_names', list(feature_names))
                except Exception:
                    pass

                if config.uncertainty_params:
                    unc_params.update(config.uncertainty_params)
                unc_params_base = unc_params
            for target_name, y in [('T', self.y_T), ('P', self.y_P)]:
                target_seed = _derive_target_seed(random_seed, target_name)
                unc_module = None
                if config.run_uncertainty and unc_params_base is not None:
                    unc_params = dict(unc_params_base)
                    unc_params.setdefault('random_seed', target_seed)
                    unc_module = MCUncertaintyEstimator(**unc_params)
                print(f"\n--- Target: {target_name} ---")
                
                protocol = StratifiedCVProtocol(n_splits=n_splits, random_seed=target_seed)
                corr_module = get_correction_module(config.corr_module_name, **config.corr_params)
                results = protocol.run(
                    self.X, y,
                    pipeline_factory,
                    uncertainty_module=unc_module,
                    corr_module=corr_module,
                    stratify_labels=stratify_labels,
                    verbose=verbose
                )
                
                results['fold_metrics'].to_csv(
                    os.path.join(self.output_dir, f'{config.exp_id}_{target_name}_fold_metrics.csv'),
                    index=False
                )
                
                results['predictions'].to_parquet(
                    os.path.join(self.output_dir, f'{config.exp_id}_{target_name}_predictions.parquet'),
                    index=False
                )
                
                import joblib
                models_dir = os.path.join(self.output_dir, 'models')
                os.makedirs(models_dir, exist_ok=True)

                full_pipeline = _call_pipeline_factory(pipeline_factory, target_seed)
                full_pipeline.fit(self.X, y, stratify_labels=stratify_labels)
                full_pipeline.set_correction(results['corr_module'], results['corr_model'])

                model_path = os.path.join(models_dir, f'{config.exp_id}_{target_name}_model.joblib')
                joblib.dump({
                    'model': full_pipeline.get_model(),
                    'model_module': full_pipeline.model_module,
                    'corr_model': results['corr_model'],
                    'data_state': full_pipeline._state,
                    'config': {
                        'exp_id': config.exp_id,
                        'data_module': config.data_module_name,
                        'model_module': config.model_module_name,
                        'corr_module': config.corr_module_name,
                        'feature_set': config.feature_set,
                    },
                }, model_path)

                y_test = y_T_test if target_name == 'T' else y_P_test
                if X_test is not None and y_test is not None:
                    X_test_scaled, _ = full_pipeline.data_module.transform(X_test, full_pipeline._state)
                    y_test_pred_raw = full_pipeline.predict(X_test_scaled, apply_correction=False)
                    y_test_pred_corr = full_pipeline.corr_module.apply(full_pipeline._corr_model, y_test_pred_raw)
                    test_metrics = compute_all_metrics(y_test, y_test_pred_corr, y_test_pred_raw)
                    for k, v in test_metrics.items():
                        exp_result[f'{target_name}_test_{k}'] = v
                
                for k, v in results['summary'].items():
                    exp_result[f'{target_name}_{k}'] = v

            all_results.append(exp_result)
        
        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(self.output_dir, 'metrics_summary.csv')
        if os.path.exists(summary_path):
            existing = pd.read_csv(summary_path)
            combined = pd.concat([existing, summary_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['exp_id'], keep='last')
            combined.to_csv(summary_path, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)
        
        return summary_df
    
    def run_stability_repeats(self,
                              configs: List[ExperimentConfig],
                              X_test: np.ndarray,
                              y_T_test: np.ndarray,
                              y_P_test: np.ndarray,
                              stratify_labels: Optional[np.ndarray] = None,
                              n_splits: int = 10,
                              test_size: float = 0.3,
                              n_repeats: int = 1000,
                              checkpoint_interval: int = 20,
                              random_seed: int = 0,
                              resume: bool = False,
                              repeat_start: int = 0,
                              repeat_end: Optional[int] = None,
                              segment_tag: Optional[str] = None,
                              write_summary: bool = True,
                              verbose: bool = True) -> pd.DataFrame:
        from .data_modules import get_data_module
        from .model_modules import get_model_module
        from .correction_modules import get_correction_module

        stability_dir = os.path.join(self.output_dir, 'stability')
        os.makedirs(stability_dir, exist_ok=True)

        if repeat_end is None:
            repeat_end = repeat_start + n_repeats - 1
        if repeat_start < 0 or repeat_end < repeat_start:
            raise ValueError("repeat_end must be >= repeat_start")

        segment_suffix = f"_{segment_tag}" if segment_tag else ""

        summary_rows = []

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Stability: {config.exp_id}")
            print(f"Config: {config.data_module_name} + {config.model_module_name} + {config.corr_module_name}")
            print(f"{'='*60}")

            def make_pipeline_factory(cfg):
                def factory(seed: Optional[int] = None):
                    seed_value = random_seed if seed is None else seed

                    data_params = _apply_seed(cfg.data_params, ['random_seed'], seed_value, force=True)
                    model_params = _apply_seed(cfg.model_params, ['random_seed'], seed_value, force=True)

                    data_mod = get_data_module(cfg.data_module_name, **data_params)
                    model_mod = get_model_module(cfg.model_module_name, **model_params)
                    corr_mod = get_correction_module(cfg.corr_module_name, **cfg.corr_params)
                    return Pipeline(data_mod, model_mod, corr_mod)
                return factory

            pipeline_factory = make_pipeline_factory(config)
            corr_module = get_correction_module(config.corr_module_name, **config.corr_params)

            for target_name, y_full, y_test in [('T', self.y_T, y_T_test), ('P', self.y_P, y_P_test)]:
                target_seed_base = _derive_target_seed(random_seed, target_name)
                repeat_metrics = []
                idx_all = np.arange(len(self.X))
            
                test_metrics_path = os.path.join(
                    stability_dir, f'{config.exp_id}_{target_name}_test_metrics{segment_suffix}.csv'
                )
            
                start_repeat = repeat_start
                if resume:
                    import re
                    latest_path = None
                    if os.path.exists(test_metrics_path):
                        latest_path = test_metrics_path
                    else:
                        pattern_new = re.compile(
                            rf"{re.escape(config.exp_id)}_{target_name}_checkpoint{re.escape(segment_suffix)}_task(\d+)\.csv"
                        )
                        checkpoints = []
                        for fname in os.listdir(stability_dir):
                            m = pattern_new.match(fname)
                            if m:
                                checkpoints.append((int(m.group(1)), fname))
                        if checkpoints:
                            checkpoints.sort()
                            latest_path = os.path.join(stability_dir, checkpoints[-1][1])
            
                    if latest_path is not None:
                        existing_df = pd.read_csv(latest_path)
                        if not existing_df.empty:
                            if 'repeat_id' in existing_df.columns:
                                existing_df = existing_df.drop_duplicates(subset=['repeat_id'])
                                existing_df = existing_df[
                                    (existing_df["repeat_id"] >= repeat_start) &
                                    (existing_df["repeat_id"] <= repeat_end)
                                ]
                                if not existing_df.empty:
                                    start_repeat = int(existing_df["repeat_id"].max()) + 1
                                else:
                                    start_repeat = repeat_start
                            else:
                                start_repeat = repeat_start + int(len(existing_df))
                            repeat_metrics = existing_df.to_dict("records")
                            if verbose:
                                print(f"  Resume {target_name}: start from repeat {start_repeat}")
            
                total_repeats = repeat_end - repeat_start + 1
                if start_repeat > repeat_end and verbose:
                    print(f"  Resume {target_name}: segment already complete")

                if stratify_labels is not None:
                    subsample_stratify = _merge_sparse_bins(stratify_labels, min_samples_per_bin=2)
                else:
                    subsample_stratify = None

                for i in range(start_repeat, repeat_end + 1):
                    seed = target_seed_base + i

                    if subsample_stratify is not None:
                        train_idx, _ = train_test_split(
                            idx_all,
                            test_size=test_size,
                            random_state=seed,
                            stratify=subsample_stratify
                        )
                    else:
                        train_idx, _ = train_test_split(
                            idx_all,
                            test_size=test_size,
                            random_state=seed
                        )

                    X_train = self.X[train_idx]
                    y_train = y_full[train_idx]
                    if stratify_labels is not None:
                        stratify_raw = stratify_labels[train_idx]
                        n_bins_raw = len(np.unique(stratify_raw))
                        merged_labels = _merge_sparse_bins(stratify_raw, min_samples_per_bin=n_splits)
                        n_bins_merged = len(np.unique(merged_labels))
                        bins_merged = int(n_bins_merged < n_bins_raw)
                        stratify_train = merged_labels
                    else:
                        merged_labels = None
                        stratify_train = None
                        n_bins_raw = 0
                        n_bins_merged = 0
                        bins_merged = 0
            
                    effective_n_splits = _get_effective_n_splits(merged_labels, n_splits, len(X_train))
            
                    protocol = StratifiedCVProtocol(n_splits=effective_n_splits, random_seed=seed)
                    cv_results = protocol.run(
                        X_train,
                        y_train,
                        pipeline_factory,
                        uncertainty_module=None,
                        corr_module=corr_module,
                        stratify_labels=merged_labels,
                        verbose=False
                    )
                    corr_model = cv_results["corr_model"]
            
                    pipeline = pipeline_factory(seed)
                    pipeline.fit(X_train, y_train, stratify_labels=stratify_train)
                    pipeline.set_correction(corr_module, corr_model)
            
                    X_test_scaled, _ = pipeline.data_module.transform(X_test, pipeline._state)
                    y_pred_raw = pipeline.predict(X_test_scaled, apply_correction=False)
                    y_pred_corr = corr_module.apply(corr_model, y_pred_raw)
            
                    metrics = compute_all_metrics(y_test, y_pred_corr, y_pred_raw)
                    metrics["repeat_id"] = i
                    metrics["n_splits_requested"] = n_splits
                    metrics["n_splits_used"] = effective_n_splits
                    metrics["n_bins_raw"] = n_bins_raw
                    metrics["n_bins_merged"] = n_bins_merged
                    metrics["bins_merged"] = bins_merged
                    repeat_metrics.append(metrics)
            
                    current_idx = i - repeat_start + 1
                    if verbose and (current_idx % 50) == 0:
                        print(f"  Repeat {current_idx}/{total_repeats}")
            
                    if checkpoint_interval > 0 and (current_idx % checkpoint_interval) == 0:
                        checkpoint_df = pd.DataFrame(repeat_metrics)
                        checkpoint_path = os.path.join(
                            stability_dir, f'{config.exp_id}_{target_name}_checkpoint{segment_suffix}_task{current_idx:05d}.csv'
                        )
                        checkpoint_df.to_csv(checkpoint_path, index=False)
                        if verbose:
                            print(f"  Checkpoint saved at task {current_idx} -> {os.path.basename(checkpoint_path)}")

                repeat_df = pd.DataFrame(repeat_metrics)
                repeat_df.to_csv(test_metrics_path, index=False)
            
                if write_summary:
                    summary = summarize_folds(repeat_metrics, compute_ci=True)
                    summary_row = {"exp_id": config.exp_id, "target": target_name}
                    summary_row.update(summary)
                    summary_rows.append(summary_row)
        if write_summary:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(
                os.path.join(stability_dir, 'stability_summary.csv'),
                index=False
            )
            return summary_df

        return pd.DataFrame()
    def compute_effect_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute effect table relative to a baseline within each feature_set.
        """
        def select_baseline(df: pd.DataFrame) -> pd.Series:
            if {'data_module', 'model_module', 'corr_module'}.issubset(df.columns):
                mask = (
                    (df['data_module'] == 'raw') &
                    (df['model_module'] == 'ert') &
                    (df['corr_module'] == 'none')
                )
                if mask.any():
                    return df[mask].iloc[0]
            return df.sort_values('exp_id').iloc[0]

        effects = []
        if 'feature_set' in summary_df.columns:
            grouped = summary_df.groupby('feature_set', dropna=False)
        else:
            grouped = [(None, summary_df)]

        for feature_set, df in grouped:
            if df.empty:
                continue
            baseline = select_baseline(df)
            for _, row in df.iterrows():
                effect = {'exp_id': row['exp_id']}
                if feature_set is not None:
                    effect['feature_set'] = feature_set

                for target in ['T', 'P']:
                    if f'{target}_rmse_mean' in row and f'{target}_rmse_mean' in baseline:
                        effect[f'{target}_delta_rmse'] = row[f'{target}_rmse_mean'] - baseline[f'{target}_rmse_mean']
                        effect[f'{target}_pct_rmse'] = (
                            effect[f'{target}_delta_rmse'] / baseline[f'{target}_rmse_mean']
                        ) * 100
                    if f'{target}_mbe_mean' in row:
                        effect[f'{target}_delta_mbe'] = abs(row[f'{target}_mbe_mean']) - abs(
                            baseline.get(f'{target}_mbe_mean', 0)
                        )

                effects.append(effect)

        effect_df = pd.DataFrame(effects)
        effect_df.to_csv(
            os.path.join(self.output_dir, 'effect_table.csv'),
            index=False
        )

        return effect_df

    def save_config(self, configs: List[ExperimentConfig], extra_info: Dict = None):
        """save_config function."""
        from config import get_version_info
        version_info = get_version_info()

        config_data = {
            'experiments': [
                {
                    'exp_id': c.exp_id,
                    'feature_set': c.feature_set,
                    'data_module': c.data_module_name,
                    'model_module': c.model_module_name,
                    'corr_module': c.corr_module_name,
                    'data_params': c.data_params,
                    'model_params': c.model_params,
                    'corr_params': c.corr_params,
                }
                for c in configs
            ],
            'data_shape': {
                'n_samples': len(self.X),
                'n_features': self.X.shape[1],
            },
            'version_info': version_info,
        }
        
        n_features_by_feature_set = None
        if extra_info and 'n_features_by_feature_set' in extra_info:
            n_features_by_feature_set = extra_info.pop('n_features_by_feature_set')

        if n_features_by_feature_set is not None:
            config_data['data_shape']['n_features_by_feature_set'] = n_features_by_feature_set

        if extra_info:
            config_data.update(extra_info)
        
        with open(os.path.join(self.output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)

