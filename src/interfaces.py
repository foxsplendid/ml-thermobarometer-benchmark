# -*- coding: utf-8 -*-
"""Abstract interfaces and shared state containers for pipeline modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from .protocol import Pipeline


# ============================================================
# ============================================================

@dataclass
class DataModuleState:
    """DataModuleState class."""
    scaler: Any = None
    bin_edges: Optional[np.ndarray] = None
    feature_std: Optional[np.ndarray] = None
    feature_names: Optional[List[str]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# ============================================================

class DataModule(ABC):
    """DataModule class."""

    def _infer_feature_names(self, n_features: int, feature_names: Optional[List[str]] = None) -> List[str]:
        """_infer_feature_names function."""
        if feature_names is not None:
            return feature_names
        raise ValueError(
            f"feature_names must be provided explicitly (got n_features={n_features}); "
            "pass feature_names to the data module constructor."
        )

    @abstractmethod
    def fit_transform(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      fold_seed: Optional[int] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """fit_transform function."""
        pass
    
    @abstractmethod
    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """transform function."""
        pass
    
    def get_name(self) -> str:
        """get_name function."""
        return self.__class__.__name__


# ============================================================
# ============================================================

class ModelModule(ABC):
    """ModelModule class."""
    
    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        pass
    
    def get_name(self) -> str:
        """get_name function."""
        return self.__class__.__name__
    
    def get_training_time(self) -> float:
        """get_training_time function."""
        return getattr(self, '_training_time', 0.0)


# ============================================================
# ============================================================

class CorrectionModule(ABC):
    """CorrectionModule class."""
    
    @abstractmethod
    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Any:
        """fit function."""
        pass
    
    @abstractmethod
    def apply(self, corr_model: Any, y_pred: np.ndarray) -> np.ndarray:
        """apply function."""
        pass
    
    def get_name(self) -> str:
        """get_name function."""
        return self.__class__.__name__
    
    def get_correction_params(self, corr_model: Any) -> Dict[str, float]:
        """get_correction_params function."""
        return {}


# ============================================================
# ============================================================

class UncertaintyModule(ABC):
    """UncertaintyModule class."""
    
    @abstractmethod
    def predict_distribution(self, 
                             pipeline: 'Pipeline',
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, np.ndarray]:
        """predict_distribution function."""
        pass
    
    @abstractmethod
    def compute_calibration_metrics(self, 
                                    y_true: np.ndarray, 
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        """compute_calibration_metrics function."""
        pass
    
    def get_name(self) -> str:
        """get_name function."""
        return self.__class__.__name__


