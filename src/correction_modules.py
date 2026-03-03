# -*- coding: utf-8 -*-
"""Prediction-bias correction modules used after model inference."""

import numpy as np
from typing import Any, Dict, Optional

from .interfaces import CorrectionModule

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

class ResidualRegressionCorrector(CorrectionModule):
    """ResidualRegressionCorrector class."""
    
    def __init__(self, 
                 method: str = 'ridge',
                 alpha: float = 1.0,
                 use_polynomial: bool = False,
                 poly_degree: int = 2):
        """__init__ function."""
        self.method = method
        self.alpha = alpha
        self.use_polynomial = use_polynomial
        self.poly_degree = poly_degree
    
    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Dict[str, Any]:
        """fit function."""
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        residuals = y_true_train - y_pred_train
        
        X_pred = y_pred_train.reshape(-1, 1)
        
        poly_features = None
        if self.use_polynomial:
            poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_features = poly_features.fit_transform(X_pred)
        else:
            X_features = X_pred
        
        if self.method == 'ridge':
            residual_model = Ridge(alpha=self.alpha)
        else:
            residual_model = LinearRegression()
        
        residual_model.fit(X_features, residuals)
        
        from scipy.stats import linregress
        reg_result = linregress(y_pred_train, y_true_train)
        
        return {
            'residual_model': residual_model,
            'poly_features': poly_features,
            'slope': reg_result.slope,
            'intercept': reg_result.intercept,
            'method': self.method
        }
    
    def apply(self, corr_model: Dict[str, Any], y_pred: np.ndarray) -> np.ndarray:
        """apply function."""
        X_pred = y_pred.reshape(-1, 1)
        
        if corr_model['poly_features'] is not None:
            X_features = corr_model['poly_features'].transform(X_pred)
        else:
            X_features = X_pred
        
        predicted_residual = corr_model['residual_model'].predict(X_features)
        
        return y_pred + predicted_residual
    
    def get_correction_params(self, corr_model: Dict[str, Any]) -> Dict[str, float]:
        """get_correction_params function."""
        if corr_model is None:
            return {}
        
        result = {
            'method': corr_model.get('method', 'unknown'),
            'slope_before': corr_model.get('slope', np.nan),
            'intercept_before': corr_model.get('intercept', np.nan),
        }
        
        model = corr_model.get('residual_model')
        if model is not None and hasattr(model, 'coef_'):
            result['residual_coef'] = float(model.coef_[0]) if len(model.coef_) == 1 else 0.0
            result['residual_intercept'] = float(model.intercept_)
        
        return result

# ============================================================
# ============================================================

class SegmentedLinearCorrector(CorrectionModule):
    """SegmentedLinearCorrector class."""

    def __init__(self,
                 n_segments: int = 3,
                 quantiles: Optional[list] = None,
                 clip_to_train_range: bool = True):
        """__init__ function."""
        self.n_segments = n_segments
        self.quantiles = quantiles or [1/3, 2/3]
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

            if np.sum(mask) < 5:
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
        'residual': ResidualRegressionCorrector,
        'segmented': SegmentedLinearCorrector,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"Unknown correction module: {name}, supported: {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

