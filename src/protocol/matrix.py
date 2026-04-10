# -*- coding: utf-8 -*-
"""Multi-experiment orchestration: ExperimentConfig and ExperimentMatrix."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from ..metrics import compute_all_metrics, summarize_folds
from .pipeline import (
    Pipeline,
    StratifiedCVProtocol,
    apply_seed,
    call_pipeline_factory,
    derive_target_seed,
    get_effective_n_splits,
    merge_sparse_bins,
)


def _run_single_stability_repeat(
    repeat_id: int,
    X: np.ndarray,
    y_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    idx_all: np.ndarray,
    subsample_stratify: Optional[np.ndarray],
    stratify_labels: Optional[np.ndarray],
    data_module_name: str,
    data_params: Dict[str, Any],
    model_module_name: str,
    model_params: Dict[str, Any],
    corr_module_name: str,
    corr_params: Dict[str, Any],
    test_size: float,
    n_splits: int,
    target_seed_base: int,
    random_seed: int,
) -> Dict[str, Any]:
    """Run one stability repeat in an isolated worker process.

    Designed to be picklable by joblib loky so it can be dispatched to a
    worker process without carrying closure state.  Each worker forces
    ``ML_FOLD_WORKERS=1`` to avoid nested process-pool conflicts.
    """
    import os as _os
    # Signal to CatBoost that this is a parallel worker process — GPU must not
    # be used as multiple workers would contend for the same device.
    _os.environ["ML_PARALLEL_WORKER"] = "1"
    # Disable nested fold-level parallelism to avoid process over-subscription.
    _os.environ["ML_FOLD_WORKERS"] = "1"

    from sklearn.model_selection import train_test_split as _tts
    from ..data_modules import get_data_module
    from ..model_modules import get_model_module
    from ..correction_modules import get_correction_module

    seed = target_seed_base + repeat_id

    if subsample_stratify is not None:
        train_idx, _ = _tts(
            idx_all, test_size=test_size, random_state=seed, stratify=subsample_stratify
        )
    else:
        train_idx, _ = _tts(idx_all, test_size=test_size, random_state=seed)

    X_train = X[train_idx]
    y_train = y_full[train_idx]

    if stratify_labels is not None:
        stratify_raw = stratify_labels[train_idx]
        n_bins_raw = len(np.unique(stratify_raw))
        merged_labels = merge_sparse_bins(stratify_raw, min_samples_per_bin=n_splits)
        n_bins_merged = len(np.unique(merged_labels))
        bins_merged = int(n_bins_merged < n_bins_raw)
        stratify_train = merged_labels
    else:
        merged_labels = None
        stratify_train = None
        n_bins_raw = 0
        n_bins_merged = 0
        bins_merged = 0

    effective_n_splits = get_effective_n_splits(merged_labels, n_splits, len(X_train))

    corr_mod = get_correction_module(corr_module_name, **corr_params)

    def _factory(s=None):
        s = seed if s is None else s
        dm = get_data_module(
            data_module_name, **apply_seed(data_params, ["random_seed"], s, force=True)
        )
        mm = get_model_module(
            model_module_name, **apply_seed(model_params, ["random_seed"], s, force=True)
        )
        cm = get_correction_module(corr_module_name, **corr_params)
        return Pipeline(dm, mm, cm)

    protocol = StratifiedCVProtocol(n_splits=effective_n_splits, random_seed=seed)
    cv_results = protocol.run(
        X_train, y_train, _factory,
        uncertainty_module=None,
        corr_module=corr_mod,
        stratify_labels=merged_labels,
        verbose=False,
    )
    corr_model = cv_results["corr_model"]

    pipeline = _factory(seed)
    pipeline.fit(X_train, y_train, stratify_labels=stratify_train, fold_seed=seed)
    pipeline.set_correction(corr_mod, corr_model)

    X_test_scaled, _ = pipeline.data_module.transform(X_test, pipeline._state)
    y_pred_raw = pipeline.predict(X_test_scaled, apply_correction=False)
    y_pred_corr = corr_mod.apply(corr_model, y_pred_raw)

    metrics = compute_all_metrics(y_test, y_pred_corr, y_pred_raw)
    metrics["repeat_id"] = repeat_id
    metrics["n_splits_requested"] = n_splits
    metrics["n_splits_used"] = effective_n_splits
    metrics["n_bins_raw"] = n_bins_raw
    metrics["n_bins_merged"] = n_bins_merged
    metrics["bins_merged"] = bins_merged
    return metrics


@dataclass
class ExperimentConfig:
    """ExperimentConfig class."""
    exp_id: str
    data_module_name: str
    model_module_name: str
    corr_module_name: str
    feature_set: str = "Liquid"
    data_params: Dict = field(default_factory=dict)
    model_params: Dict = field(default_factory=dict)
    corr_params: Dict = field(default_factory=dict)
    uncertainty_params: Dict = field(default_factory=dict)
    run_uncertainty: bool = False


class ExperimentMatrix:
    """ExperimentMatrix class."""

    def __init__(
        self,
        X: np.ndarray,
        y_T: np.ndarray,
        y_P: np.ndarray,
        output_dir: str = "results",
        target_names: Tuple[str, str] = ("T", "P"),
    ):
        """__init__ function."""
        self.X = X
        self.y_T = y_T
        self.y_P = y_P
        self.output_dir = output_dir
        self.target_names = target_names

        os.makedirs(output_dir, exist_ok=True)

    def run_experiments(
        self,
        configs: List[ExperimentConfig],
        n_splits: int = 10,
        stratify_labels: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_T_test: Optional[np.ndarray] = None,
        y_P_test: Optional[np.ndarray] = None,
        random_seed: int = 42,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """run_experiments function."""
        from ..data_modules import get_data_module
        from ..model_modules import get_model_module
        from ..correction_modules import get_correction_module
        from ..uncertainty_modules import MCUncertaintyEstimator

        all_results = []

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Experiment: {config.exp_id}")
            print(
                f"Config: {config.data_module_name} + "
                f"{config.model_module_name} + {config.corr_module_name}"
            )
            print(f"{'='*60}")

            exp_result = {
                "exp_id": config.exp_id,
                "data_module": config.data_module_name,
                "model_module": config.model_module_name,
                "corr_module": config.corr_module_name,
                "feature_set": config.feature_set,
            }

            def make_pipeline_factory(cfg):
                def factory(seed: Optional[int] = None):
                    seed_value = random_seed if seed is None else seed

                    data_params = apply_seed(cfg.data_params, ["random_seed"], seed_value)
                    model_params = apply_seed(cfg.model_params, ["random_seed"], seed_value)

                    data_mod = get_data_module(cfg.data_module_name, **data_params)
                    model_mod = get_model_module(cfg.model_module_name, **model_params)
                    corr_mod = get_correction_module(cfg.corr_module_name, **cfg.corr_params)
                    return Pipeline(data_mod, model_mod, corr_mod)
                return factory

            pipeline_factory = make_pipeline_factory(config)

            unc_params_base = None
            if config.run_uncertainty:
                unc_params: Dict[str, Any] = {}
                try:
                    from config import CONFIG as APP_CONFIG
                    unc_params.update({
                        "n_mc": APP_CONFIG.uncertainty.n_mc,
                        "percentiles": APP_CONFIG.uncertainty.percentiles,
                    })
                    feature_names = APP_CONFIG.data.feature_sets.get(config.feature_set)
                    if feature_names:
                        unc_params.setdefault("feature_names", list(feature_names))
                except Exception:
                    pass

                if config.uncertainty_params:
                    unc_params.update(config.uncertainty_params)
                unc_params_base = unc_params

            for target_name, y in [("T", self.y_T), ("P", self.y_P)]:
                target_seed = derive_target_seed(random_seed, target_name)
                unc_module = None
                if config.run_uncertainty and unc_params_base is not None:
                    unc_params = dict(unc_params_base)
                    unc_params.setdefault("random_seed", target_seed)
                    unc_module = MCUncertaintyEstimator(**unc_params)
                print(f"\n--- Target: {target_name} ---")

                protocol = StratifiedCVProtocol(n_splits=n_splits, random_seed=target_seed)
                corr_module = get_correction_module(
                    config.corr_module_name, **config.corr_params
                )
                results = protocol.run(
                    self.X, y,
                    pipeline_factory,
                    uncertainty_module=unc_module,
                    corr_module=corr_module,
                    stratify_labels=stratify_labels,
                    verbose=verbose,
                )

                results["fold_metrics"].to_csv(
                    os.path.join(
                        self.output_dir, f"{config.exp_id}_{target_name}_fold_metrics.csv"
                    ),
                    index=False,
                )

                results["predictions"].to_parquet(
                    os.path.join(
                        self.output_dir, f"{config.exp_id}_{target_name}_predictions.parquet"
                    ),
                    index=False,
                )

                import joblib
                models_dir = os.path.join(self.output_dir, "models")
                os.makedirs(models_dir, exist_ok=True)

                full_pipeline = call_pipeline_factory(pipeline_factory, target_seed)
                full_pipeline.fit(self.X, y, stratify_labels=stratify_labels)
                full_pipeline.set_correction(results["corr_module"], results["corr_model"])

                model_path = os.path.join(
                    models_dir, f"{config.exp_id}_{target_name}_model.joblib"
                )
                joblib.dump({
                    "model": full_pipeline.get_model(),
                    "model_module": full_pipeline.model_module,
                    "corr_model": results["corr_model"],
                    "data_state": full_pipeline._state,
                    "config": {
                        "exp_id": config.exp_id,
                        "data_module": config.data_module_name,
                        "model_module": config.model_module_name,
                        "corr_module": config.corr_module_name,
                        "feature_set": config.feature_set,
                    },
                }, model_path)

                y_test = y_T_test if target_name == "T" else y_P_test
                if X_test is not None and y_test is not None:
                    X_test_scaled, _ = full_pipeline.data_module.transform(
                        X_test, full_pipeline._state
                    )
                    y_test_pred_raw = full_pipeline.predict(X_test_scaled, apply_correction=False)
                    y_test_pred_corr = full_pipeline.corr_module.apply(
                        full_pipeline._corr_model, y_test_pred_raw
                    )
                    test_metrics = compute_all_metrics(
                        y_test, y_test_pred_corr, y_test_pred_raw
                    )
                    for k, v in test_metrics.items():
                        exp_result[f"{target_name}_test_{k}"] = v

                for k, v in results["summary"].items():
                    exp_result[f"{target_name}_{k}"] = v

            all_results.append(exp_result)

        summary_df = pd.DataFrame(all_results)
        summary_path = os.path.join(self.output_dir, "metrics_summary.csv")
        if os.path.exists(summary_path):
            existing = pd.read_csv(summary_path)
            combined = pd.concat([existing, summary_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["exp_id"], keep="last")
            combined.to_csv(summary_path, index=False)
        else:
            summary_df.to_csv(summary_path, index=False)

        return summary_df

    def run_stability_repeats(
        self,
        configs: List[ExperimentConfig],
        X_test: np.ndarray,
        y_T_test: np.ndarray,
        y_P_test: np.ndarray,
        stratify_labels: Optional[np.ndarray] = None,
        n_splits: int = 10,
        test_size: float = 0.3,
        n_repeats: int = 1000,
        random_seed: int = 0,
        repeat_start: int = 0,
        repeat_end: Optional[int] = None,
        segment_tag: Optional[str] = None,
        write_summary: bool = True,
        verbose: bool = True,
        repeat_workers: int = 1,
    ) -> pd.DataFrame:
        from ..data_modules import get_data_module
        from ..model_modules import get_model_module
        from ..correction_modules import get_correction_module

        stability_dir = os.path.join(self.output_dir, "stability")
        os.makedirs(stability_dir, exist_ok=True)

        if repeat_end is None:
            repeat_end = repeat_start + n_repeats - 1
        if repeat_start < 0 or repeat_end < repeat_start:
            raise ValueError("repeat_end must be >= repeat_start")

        segment_suffix = f"_{segment_tag}" if segment_tag else ""
        total_repeats = repeat_end - repeat_start + 1
        summary_rows = []

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Stability: {config.exp_id}")
            print(
                f"Config: {config.data_module_name} + "
                f"{config.model_module_name} + {config.corr_module_name}"
            )
            print(f"{'='*60}")

            for target_name, y_full, y_test in [
                ("T", self.y_T, y_T_test),
                ("P", self.y_P, y_P_test),
            ]:
                target_seed_base = derive_target_seed(random_seed, target_name)
                idx_all = np.arange(len(self.X))

                test_metrics_path = os.path.join(
                    stability_dir,
                    f"{config.exp_id}_{target_name}_test_metrics{segment_suffix}.csv",
                )

                if stratify_labels is not None:
                    subsample_stratify = merge_sparse_bins(stratify_labels, min_samples_per_bin=2)
                else:
                    subsample_stratify = None

                repeat_ids = list(range(repeat_start, repeat_end + 1))

                _repeat_kwargs = dict(
                    X=self.X,
                    y_full=y_full,
                    X_test=X_test,
                    y_test=y_test,
                    idx_all=idx_all,
                    subsample_stratify=subsample_stratify,
                    stratify_labels=stratify_labels,
                    data_module_name=config.data_module_name,
                    data_params=config.data_params,
                    model_module_name=config.model_module_name,
                    model_params=config.model_params,
                    corr_module_name=config.corr_module_name,
                    corr_params=config.corr_params,
                    test_size=test_size,
                    n_splits=n_splits,
                    target_seed_base=target_seed_base,
                    random_seed=random_seed,
                )

                if repeat_workers > 1:
                    from joblib import Parallel, delayed
                    if verbose:
                        print(
                            f"  Running {total_repeats} repeats in parallel"
                            f" (workers={repeat_workers}, backend=loky) ..."
                        )
                    repeat_metrics = Parallel(n_jobs=repeat_workers, backend="loky")(
                        delayed(_run_single_stability_repeat)(i, **_repeat_kwargs)
                        for i in repeat_ids
                    )
                else:
                    repeat_metrics = []
                    for i in repeat_ids:
                        repeat_metrics.append(_run_single_stability_repeat(i, **_repeat_kwargs))
                        current_idx = i - repeat_start + 1
                        if verbose and (current_idx % 50) == 0:
                            print(f"  Repeat {current_idx}/{total_repeats}")

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
                os.path.join(stability_dir, "stability_summary.csv"),
                index=False,
            )
            return summary_df

        return pd.DataFrame()

    def compute_effect_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """Compute effect table relative to a baseline within each feature_set."""
        def select_baseline(df: pd.DataFrame) -> pd.Series:
            if {"data_module", "model_module", "corr_module"}.issubset(df.columns):
                mask = (
                    (df["data_module"] == "raw")
                    & (df["model_module"] == "ert")
                    & (df["corr_module"] == "none")
                )
                if mask.any():
                    return df[mask].iloc[0]
            return df.sort_values("exp_id").iloc[0]

        effects = []
        if "feature_set" in summary_df.columns:
            grouped = summary_df.groupby("feature_set", dropna=False)
        else:
            grouped = [(None, summary_df)]

        for feature_set, df in grouped:
            if df.empty:
                continue
            baseline = select_baseline(df)
            for _, row in df.iterrows():
                effect = {"exp_id": row["exp_id"]}
                if feature_set is not None:
                    effect["feature_set"] = feature_set

                for target in ["T", "P"]:
                    if f"{target}_rmse_mean" in row and f"{target}_rmse_mean" in baseline:
                        effect[f"{target}_delta_rmse"] = (
                            row[f"{target}_rmse_mean"] - baseline[f"{target}_rmse_mean"]
                        )
                        effect[f"{target}_pct_rmse"] = (
                            effect[f"{target}_delta_rmse"] / baseline[f"{target}_rmse_mean"]
                        ) * 100
                    if f"{target}_mbe_mean" in row:
                        effect[f"{target}_delta_mbe"] = abs(row[f"{target}_mbe_mean"]) - abs(
                            baseline.get(f"{target}_mbe_mean", 0)
                        )

                effects.append(effect)

        effect_df = pd.DataFrame(effects)
        effect_df.to_csv(
            os.path.join(self.output_dir, "effect_table.csv"),
            index=False,
        )

        return effect_df

    def save_config(
        self, configs: List[ExperimentConfig], extra_info: Optional[Dict] = None
    ):
        """save_config function."""
        from config import get_version_info
        version_info = get_version_info()

        config_data: Dict[str, Any] = {
            "experiments": [
                {
                    "exp_id": c.exp_id,
                    "feature_set": c.feature_set,
                    "data_module": c.data_module_name,
                    "model_module": c.model_module_name,
                    "corr_module": c.corr_module_name,
                    "data_params": c.data_params,
                    "model_params": c.model_params,
                    "corr_params": c.corr_params,
                }
                for c in configs
            ],
            "data_shape": {
                "n_samples": len(self.X),
                "n_features": self.X.shape[1],
            },
            "version_info": version_info,
        }

        n_features_by_feature_set = None
        if extra_info and "n_features_by_feature_set" in extra_info:
            n_features_by_feature_set = extra_info.pop("n_features_by_feature_set")

        if n_features_by_feature_set is not None:
            config_data["data_shape"]["n_features_by_feature_set"] = n_features_by_feature_set

        if extra_info:
            config_data.update(extra_info)

        with open(os.path.join(self.output_dir, "config_used.yaml"), "w") as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
