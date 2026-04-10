# -*- coding: utf-8 -*-
"""Prediction-bias correction modules used after model inference."""

import numpy as np
from typing import Any, Dict, Optional

from .interfaces import CorrectionModule

# ============================================================
# ============================================================

DEFAULT_N_SEGMENTS: int = 3
DEFAULT_SEGMENT_QUANTILES = [1 / 3, 2 / 3]
MIN_SEGMENT_SAMPLES: int = 5

# ============================================================
# ============================================================

class NoCorrection(CorrectionModule):
    """NoCorrection class."""

    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Any:
        """fit function."""
        return None
    
    def apply(self, corr_model: Any, y_pred: np.ndarray) -> np.ndarray:
        """apply function."""
        return y_pred.copy()
    
    def get_correction_params(self, corr_model: Any) -> Dict[str, float]:
        """get_correction_params function."""
        return {'method': 'none'}

# ============================================================
# ============================================================

class SegmentedLinearCorrector(CorrectionModule):
    """SegmentedLinearCorrector class."""

    def __init__(self,
                 n_segments: int = DEFAULT_N_SEGMENTS,
                 quantiles: Optional[list] = None,
                 clip_to_train_range: bool = True):
        """__init__ function."""
        self.n_segments = n_segments
        self.quantiles = quantiles if quantiles is not None else DEFAULT_SEGMENT_QUANTILES
        self.clip_to_train_range = clip_to_train_range

    def fit(self,
            y_true_train: np.ndarray,
            y_pred_train: np.ndarray) -> Dict[str, Any]:
        from sklearn.linear_model import LinearRegression

        boundaries = [y_pred_train.min()]
        for q in self.quantiles:
            boundaries.append(np.percentile(y_pred_train, q * 100))
        boundaries.append(y_pred_train.max())

        segment_models = []
        for i in range(len(boundaries) - 1):
            mask = (y_pred_train >= boundaries[i]) & (y_pred_train < boundaries[i + 1])
            if i == len(boundaries) - 2:
                mask = (y_pred_train >= boundaries[i]) & (y_pred_train <= boundaries[i + 1])

            if np.sum(mask) < MIN_SEGMENT_SAMPLES:
                segment_models.append(None)
                continue

            X_seg = y_pred_train[mask].reshape(-1, 1)
            y_seg = y_true_train[mask]

            model = LinearRegression()
            model.fit(X_seg, y_seg)
            segment_models.append(model)

        return {
            'boundaries': boundaries,
            'segment_models': segment_models,
            'y_min': float(np.min(y_true_train)),
            'y_max': float(np.max(y_true_train)),
            'clip_to_train_range': self.clip_to_train_range,
        }

    def apply(self, corr_model: Dict[str, Any], y_pred: np.ndarray) -> np.ndarray:
        boundaries = corr_model['boundaries']
        segment_models = corr_model['segment_models']

        y_corr = np.copy(y_pred)

        for i, model in enumerate(segment_models):
            if model is None:
                continue

            if i == len(segment_models) - 1:
                mask = (y_pred >= boundaries[i]) & (y_pred <= boundaries[i + 1])
            else:
                mask = (y_pred >= boundaries[i]) & (y_pred < boundaries[i + 1])

            if np.sum(mask) > 0:
                y_corr[mask] = model.predict(y_pred[mask].reshape(-1, 1))

        if corr_model.get('clip_to_train_range'):
            y_min = corr_model.get('y_min')
            y_max = corr_model.get('y_max')
            if y_min is not None and y_max is not None:
                y_corr = np.clip(y_corr, y_min, y_max)

        return y_corr

    def get_correction_params(self, corr_model: Dict[str, Any]) -> Dict[str, float]:
        if corr_model is None:
            return {}

        params = {
            'method': 'segmented_linear',
            'n_segments': len(corr_model.get('segment_models', [])),
            'clip_to_train_range': bool(corr_model.get('clip_to_train_range')),
            'y_min': corr_model.get('y_min'),
            'clip_max': corr_model.get('y_max'),
        }

        for i, model in enumerate(corr_model.get('segment_models', [])):
            if model is not None and hasattr(model, 'coef_'):
                params[f'segment_{i}_slope'] = float(model.coef_[0])
                params[f'segment_{i}_intercept'] = float(model.intercept_)

        return params

# ============================================================
# ============================================================

def get_correction_module(name: str, **kwargs) -> CorrectionModule:
    """get_correction_module function."""
    modules = {
        'none': NoCorrection,
        'segmented': SegmentedLinearCorrector,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"Unknown correction module: {name}, supported: {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

