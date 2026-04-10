# -*- coding: utf-8 -*-
"""Monte Carlo uncertainty estimators and calibration metrics."""
import numpy as np
from typing import Any, Dict, List, Optional
from .interfaces import UncertaintyModule

# ============================================================
# ============================================================
class MCUncertaintyEstimator(UncertaintyModule):
    """MCUncertaintyEstimator class."""

    def __init__(self,
                 n_mc: int = 1000,
                 feature_names: Optional[List[str]] = None,
                 percentiles: tuple = (5, 16, 50, 84, 95),
                 random_seed: int = 42):
        """__init__ function."""
        self.n_mc = n_mc
        self.feature_names = feature_names
        self.percentiles = percentiles
        self.random_seed = random_seed

    def predict_distribution(self,
                             pipeline: Any,
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None,
                             fold_idx: int = 0
                             ) -> Dict[str, np.ndarray]:
        """predict_distribution function."""
        from .utils import get_rel_err_vector, epma_perturb

        seed_offset = mc_params.get('seed_offset', 0) if mc_params else 0
        effective_seed = self.random_seed + fold_idx + seed_offset
        rng = np.random.RandomState(effective_seed)

        n_mc = mc_params.get('n_mc', self.n_mc) if mc_params else self.n_mc
        percentiles = mc_params.get('percentiles', self.percentiles) if mc_params else self.percentiles
        feature_names = mc_params.get('feature_names', self.feature_names) if mc_params else self.feature_names

        if feature_names is None:
            raise ValueError(
                "feature_names must be provided explicitly to MCUncertaintyEstimator "
                f"(got n_features={X.shape[1]}); pass feature_names to the constructor "
                "or via mc_params['feature_names']."
            )

        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names length must match X feature dimension")
        rel_err_vec = get_rel_err_vector(feature_names, strict=True)

        n_samples = X.shape[0]
        predictions = np.zeros((n_mc, n_samples))

        for i in range(n_mc):
            X_perturbed = epma_perturb(X, rel_err_vec, rng)
            predictions[i] = pipeline.predict_from_raw_input(X_perturbed)

        percentiles = tuple(percentiles)
        pct_values = np.percentile(predictions, percentiles, axis=0)
        pct_map = {p: pct_values[i] for i, p in enumerate(percentiles)}

        p16 = pct_map.get(16, np.percentile(predictions, 16, axis=0))
        p84 = pct_map.get(84, np.percentile(predictions, 84, axis=0))
        median = pct_map.get(50, np.percentile(predictions, 50, axis=0))
        p5 = pct_map.get(5, np.percentile(predictions, 5, axis=0))
        p95 = pct_map.get(95, np.percentile(predictions, 95, axis=0))

        return {
            'samples': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'ci_lower': p16,
            'ci_upper': p84,
            'median': median,
            'p16': p16,
            'p84': p84,
            'p5': p5,
            'p95': p95,
        }

    def compute_calibration_metrics(self,
                                    y_true: np.ndarray,
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        lower = dist.get('p16', dist.get('ci_lower'))
        upper = dist.get('p84', dist.get('ci_upper'))
        if lower is None or upper is None:
            lower = np.percentile(dist['samples'], 16, axis=0)
            upper = np.percentile(dist['samples'], 84, axis=0)

        in_68_interval = (y_true >= lower) & (y_true <= upper)
        picp_68 = np.mean(in_68_interval)

        lower_90 = dist.get('p5')
        upper_90 = dist.get('p95')
        if lower_90 is None or upper_90 is None:
            lower_90 = np.percentile(dist['samples'], 5, axis=0)
            upper_90 = np.percentile(dist['samples'], 95, axis=0)
        in_90_interval = (y_true >= lower_90) & (y_true <= upper_90)
        picp_90 = np.mean(in_90_interval)

        widths = upper - lower
        mean_width = np.mean(widths)
        median_width = np.median(widths)

        center = dist.get('median', dist.get('mean'))
        abs_errors = np.abs(y_true - center)
        if np.std(abs_errors) > 1e-10 and np.std(widths) > 1e-10:
            corr = np.corrcoef(abs_errors, widths)[0, 1]
        else:
            corr = np.nan

        sharpness = np.std(widths)

        return {
            'picp_68': picp_68,
            'picp_90': picp_90,
            'mean_interval_width': mean_width,
            'median_interval_width': median_width,
            'error_uncertainty_corr': corr,
            'sharpness': sharpness,
        }

    def compute_reliability_diagram_data(self,
                                         y_true: np.ndarray,
                                         dist: Dict[str, np.ndarray],
                                         n_bins: int = 10
                                         ) -> Dict[str, np.ndarray]:
        quantiles = np.linspace(5, 95, n_bins)
        observed_coverages = []

        for q in quantiles:
            lower = np.percentile(dist['samples'], (100 - q) / 2, axis=0)
            upper = np.percentile(dist['samples'], 100 - (100 - q) / 2, axis=0)
            in_interval = (y_true >= lower) & (y_true <= upper)
            observed_coverages.append(np.mean(in_interval))

        return {
            'expected_coverage': quantiles / 100,
            'observed_coverage': np.array(observed_coverages),
        }


# ============================================================
# ============================================================
def get_uncertainty_module(name: str, **kwargs) -> UncertaintyModule:
    """get_uncertainty_module function."""
    modules = {
        'mc': MCUncertaintyEstimator,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"Unknown uncertainty module: {name}, supported: {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

