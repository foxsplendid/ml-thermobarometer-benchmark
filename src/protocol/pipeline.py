# -*- coding: utf-8 -*-
"""Fold-level execution: Pipeline, StratifiedCVProtocol, and seeding helpers."""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from ..interfaces import (
    CorrectionModule,
    DataModule,
    DataModuleState,
    ModelModule,
    UncertaintyModule,
)
from ..metrics import compute_all_metrics, summarize_folds
from ..runtime import get_fold_backend, get_fold_workers

logger = logging.getLogger(__name__)

# Offset applied to the random seed for the pressure (P) target so that T and
# P training runs use statistically independent random sequences while sharing
# the same base seed. 1000 is large enough to avoid overlap with per-fold seed
# increments (max expected n_splits <= 50).
P_SEED_OFFSET = 1000


# --- Seeding helpers ------------------------------------------------------

def derive_target_seed(base_seed: int, target_name: str) -> int:
    """Return the per-target base seed (adds ``P_SEED_OFFSET`` for the P target)."""
    return base_seed + (P_SEED_OFFSET if str(target_name).upper() == "P" else 0)


def apply_seed(
    params: Dict[str, Any],
    keys: List[str],
    seed: int,
    force: bool = False,
) -> Dict[str, Any]:
    """Return a copy of ``params`` with the given seed applied to ``keys``."""
    updated = dict(params)
    for key in keys:
        if force or key not in updated:
            updated[key] = seed
    return updated


def call_pipeline_factory(
    factory: Callable,
    seed: int,
    fold_idx: int = 0,
) -> "Pipeline":
    """Invoke a pipeline factory, passing ``seed + fold_idx`` when supported."""
    effective_seed = seed + fold_idx
    try:
        return factory(effective_seed)
    except TypeError:
        return factory()


# --- Stratification helpers ----------------------------------------------

def merge_sparse_bins(
    labels: np.ndarray,
    min_samples_per_bin: int,
    verbose: bool = False,
) -> np.ndarray:
    """Merge bins with fewer than ``min_samples_per_bin`` into the nearest dense bin."""
    unique_bins, bin_counts = np.unique(labels, return_counts=True)
    merged = labels.copy()

    sparse_bins = unique_bins[bin_counts < min_samples_per_bin]
    if sparse_bins.size == 0:
        if verbose:
            logger.debug(
                f"Bin merge: no sparse bins (min_samples={min_samples_per_bin}), "
                f"{len(unique_bins)} bins unchanged"
            )
        return merged

    non_sparse_bins = unique_bins[bin_counts >= min_samples_per_bin]
    if non_sparse_bins.size == 0:
        if verbose:
            logger.warning(
                f"Bin merge: all {len(unique_bins)} bins are sparse, collapsing to single bin"
            )
        return np.zeros_like(labels)

    merge_map: Dict[int, int] = {}
    for sparse_bin in sparse_bins:
        distances = np.abs(non_sparse_bins - sparse_bin)
        nearest_bin = non_sparse_bins[np.argmin(distances)]
        merged[labels == sparse_bin] = nearest_bin
        merge_map[int(sparse_bin)] = int(nearest_bin)

    if verbose:
        n_merged = len(sparse_bins)
        n_remaining = len(non_sparse_bins)
        logger.info(
            f"Bin merge: {n_merged} sparse bins -> {n_remaining} bins "
            f"(threshold={min_samples_per_bin})"
        )
        if n_merged <= 10:
            for src, dst in merge_map.items():
                logger.debug(f"  bin {src} -> {dst}")

    return merged


def get_effective_n_splits(
    labels: Optional[np.ndarray],
    requested: int,
    n_samples: int,
) -> int:
    """Clamp ``requested`` to the largest feasible fold count given the bin sizes."""
    if n_samples <= 1:
        raise ValueError(
            f"Cannot perform cross-validation: n_samples={n_samples}. "
            "At least 2 samples are required."
        )
    if labels is None:
        return max(2, min(requested, n_samples))
    _, bin_counts = np.unique(labels, return_counts=True)
    min_bin = int(bin_counts.min()) if bin_counts.size > 0 else 1
    effective = min(requested, min_bin, n_samples)
    return max(2, effective)


# --- Pipeline -------------------------------------------------------------

class Pipeline:
    """Pipeline class."""

    def __init__(
        self,
        data_module: DataModule,
        model_module: ModelModule,
        corr_module: CorrectionModule,
    ):
        self.data_module = data_module
        self.model_module = model_module
        self.corr_module = corr_module

        self._state: Optional[DataModuleState] = None
        self._model: Optional[Any] = None
        self._corr_model: Optional[Any] = None
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        stratify_labels: Optional[np.ndarray] = None,
        fold_seed: Optional[int] = None,
    ) -> "Pipeline":
        """fit function."""
        X2, y2, weights, self._state = self.data_module.fit_transform(
            X_train, y_train, fold_seed=fold_seed
        )

        if stratify_labels is not None:
            if len(y2) > len(stratify_labels):
                if len(y2) % len(stratify_labels) != 0:
                    raise ValueError(
                        f"Augmented dataset size ({len(y2)}) is not an exact multiple of "
                        f"stratify_labels size ({len(stratify_labels)}); cannot broadcast labels."
                    )
                n_aug = len(y2) // len(stratify_labels)
                stratify2 = np.tile(stratify_labels, n_aug)
            else:
                stratify2 = stratify_labels
        else:
            stratify2 = None

        self._model = self.model_module.fit(X2, y2, weights, stratify_labels=stratify2)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        apply_correction: bool = True,
    ) -> np.ndarray:
        """predict function."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

        y_pred = self.model_module.predict(self._model, X)

        if apply_correction:
            y_pred = self.corr_module.apply(self._corr_model, y_pred)

        return y_pred

    def predict_from_raw_input(
        self,
        X_raw: np.ndarray,
        apply_correction: bool = True,
    ) -> np.ndarray:
        """Scale ``X_raw`` with the fitted scaler, then predict.

        Parameters
        ----------
        X_raw:
            Unscaled input features in original (physical) units.
        apply_correction:
            Whether to apply the post-hoc correction module (default ``True``).
            Pass ``False`` to obtain raw model output before correction.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

        X_scaled, _ = self.data_module.transform(X_raw, self._state)

        return self.predict(X_scaled, apply_correction)

    def get_model(self) -> Any:
        """get_model function."""
        return self._model

    def get_data_state(self) -> Any:
        """Return the fitted data-module state (scaler, feature names, etc.)."""
        return self._state

    def get_corr_model(self) -> Any:
        """Return the fitted correction model."""
        return self._corr_model

    def get_correction_params(self) -> Dict[str, float]:
        """get_correction_params function."""
        return self.corr_module.get_correction_params(self._corr_model)

    def get_name(self) -> str:
        """get_name function."""
        return (
            f"{self.data_module.get_name()}_"
            f"{self.model_module.get_name()}_"
            f"{self.corr_module.get_name()}"
        )

    def set_correction(self, corr_module: CorrectionModule, corr_model: Any) -> None:
        """set_correction function."""
        self.corr_module = corr_module
        self._corr_model = corr_model


# --- Fold worker ---------------------------------------------------------

def _run_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    stratify_labels: Optional[np.ndarray],
    pipeline_factory: Callable,
    random_seed: int,
) -> Dict[str, Any]:
    """Train and validate one fold; returns a record dict."""
    start = time.time()

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    stratify_train = (
        stratify_labels[train_idx] if stratify_labels is not None else None
    )

    pipeline = call_pipeline_factory(pipeline_factory, random_seed, fold_idx)
    fold_seed = random_seed + fold_idx
    pipeline.fit(X_train, y_train, stratify_labels=stratify_train, fold_seed=fold_seed)

    X_val_scaled, _ = pipeline.data_module.transform(X_val, pipeline.get_data_state())
    y_pred_raw = pipeline.predict(X_val_scaled, apply_correction=False)

    return {
        "fold_id": fold_idx,
        "val_idx": val_idx,
        "y_val": y_val,
        "y_pred_raw": y_pred_raw,
        "pipeline": pipeline,
        "X_val": X_val,
        "training_time": time.time() - start,
    }


# --- Stratified cross-validation protocol --------------------------------

class StratifiedCVProtocol:
    """StratifiedCVProtocol class."""

    def __init__(self, n_splits: int = 10, random_seed: int = 42):
        self.n_splits = n_splits
        self.random_seed = random_seed

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pipeline_factory: Callable[..., Pipeline],
        uncertainty_module: Optional[UncertaintyModule] = None,
        corr_module: Optional[CorrectionModule] = None,
        stratify_labels: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """run function."""
        if stratify_labels is None:
            logger.warning(
                "stratify_labels=None: using plain KFold instead of StratifiedKFold; "
                "this may cause imbalanced folds."
            )
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed,
            )
            split_iter = splitter.split(X)
        else:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed,
            )
            split_iter = splitter.split(X, stratify_labels)

        # Collect split indices up-front so we know the total count.
        all_splits = list(split_iter)
        n_folds = len(all_splits)

        fold_workers = get_fold_workers()
        fold_backend = get_fold_backend()

        if fold_workers > 1:
            # Parallel fold execution.
            if verbose:
                print(f"  Running {n_folds} folds in parallel "
                      f"(workers={fold_workers}, backend={fold_backend}) ...")
            from joblib import Parallel, delayed
            fold_records = Parallel(n_jobs=fold_workers, backend=fold_backend)(
                delayed(_run_single_fold)(
                    fold_idx, train_idx, val_idx,
                    X, y, stratify_labels,
                    pipeline_factory, self.random_seed,
                )
                for fold_idx, (train_idx, val_idx) in enumerate(all_splits)
            )
            # Sort by fold_id to preserve deterministic order.
            fold_records = sorted(fold_records, key=lambda r: r["fold_id"])
        else:
            # Sequential fold execution with progress printing.
            fold_records = []
            for fold_idx, (train_idx, val_idx) in enumerate(all_splits):
                if verbose:
                    print(f"  Fold {fold_idx + 1}/{n_folds}: ", end="")
                fold_records.append(
                    _run_single_fold(
                        fold_idx, train_idx, val_idx,
                        X, y, stratify_labels,
                        pipeline_factory, self.random_seed,
                    )
                )

        oof_pred_raw = np.full(len(y), np.nan, dtype=np.float64)
        for record in fold_records:
            oof_pred_raw[record["val_idx"]] = record["y_pred_raw"]

        if np.any(np.isnan(oof_pred_raw)):
            raise RuntimeError("OOF prediction contains NaN values.")

        if corr_module is None:
            from ..correction_modules import NoCorrection
            corr_module = NoCorrection()

        # Global corrector: fitted on all OOF predictions; returned to the caller
        # for use when fitting the final full-dataset pipeline.
        corr_model = corr_module.fit(y, oof_pred_raw)

        fold_metrics = []
        all_predictions = []
        unc_fold_metrics = [] if uncertainty_module is not None else None

        if uncertainty_module is not None and verbose:
            print("  Running MC uncertainty across folds...")

        for record in fold_records:
            # Leave-one-fold-out corrector: exclude this fold's validation
            # indices so the per-fold metrics are free of secondary leakage.
            other_mask = np.ones(len(y), dtype=bool)
            other_mask[record["val_idx"]] = False
            corr_model_fold = corr_module.fit(
                y[other_mask], oof_pred_raw[other_mask]
            )
            y_pred_corr = corr_module.apply(corr_model_fold, record["y_pred_raw"])
            dist = None

            if uncertainty_module is not None:
                pipeline = record["pipeline"]
                pipeline.set_correction(corr_module, corr_model_fold)

                dist = uncertainty_module.predict_distribution(
                    pipeline, record["X_val"], fold_idx=record["fold_id"]
                )
                y_pred_corr = dist.get("median", y_pred_corr)

                calib_metrics = uncertainty_module.compute_calibration_metrics(
                    record["y_val"], dist
                )
                calib_metrics["fold_id"] = record["fold_id"]
                unc_fold_metrics.append(calib_metrics)

            metrics = compute_all_metrics(record["y_val"], y_pred_corr, record["y_pred_raw"])
            metrics["fold_id"] = record["fold_id"]
            metrics["training_time"] = record["training_time"]
            fold_metrics.append(metrics)

            if verbose:
                print(f"RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.4f}")

            preds_payload = {
                "fold_id": record["fold_id"],
                "sample_idx": record["val_idx"],
                "y_true": record["y_val"],
                "y_pred_raw": record["y_pred_raw"],
                "y_pred_corr": y_pred_corr,
                "residual": record["y_val"] - y_pred_corr,
                "y_pred_p16": dist.get("p16") if dist is not None else np.nan,
                "y_pred_p84": dist.get("p84") if dist is not None else np.nan,
                "y_pred_median": dist.get("median", y_pred_corr) if dist is not None else np.nan,
            }
            all_predictions.append(pd.DataFrame(preds_payload))

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        summary = summarize_folds(fold_metrics)
        summary["total_training_time"] = sum(r["training_time"] for r in fold_records)

        uncertainty_results = None
        if unc_fold_metrics is not None:
            unc_summary = summarize_folds(unc_fold_metrics)
            for k, v in unc_summary.items():
                summary[f"unc_{k}"] = v

            uncertainty_results = {
                "fold_metrics": pd.DataFrame(unc_fold_metrics),
                "summary": unc_summary,
            }

        return {
            "fold_metrics": pd.DataFrame(fold_metrics),
            "predictions": predictions_df,
            "summary": summary,
            "uncertainty": uncertainty_results,
            "corr_module": corr_module,
            "corr_model": corr_model,
            "fold_records": fold_records,
        }
