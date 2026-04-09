# -*- coding: utf-8 -*-
"""Data-module implementations for scaling, balancing, and augmentation."""

import numpy as np
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from .interfaces import DataModule, DataModuleState


# ============================================================
# ============================================================

class RawDataModule(DataModule):
    """RawDataModule class."""

    def __init__(self, random_seed: int = 42, feature_names: Optional[List[str]] = None):
        """__init__ function."""
        self.random_seed = random_seed
        self.feature_names = feature_names
    
    def fit_transform(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      fold_seed: Optional[int] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """fit_transform function."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        sample_weights = np.ones(len(y_train), dtype=np.float64)

        state = DataModuleState(
            scaler=scaler,
            feature_std=np.std(X_train, axis=0),
            feature_names=self.feature_names
        )

        return X_scaled, y_train.copy(), sample_weights, state

    def transform(self,
                  X_val: np.ndarray,
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """transform function."""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None


# ============================================================
# ============================================================

class BalancedDataModule(DataModule):
    """BalancedDataModule class."""
    
    def __init__(self, 
                 n_bins: int = 10, 
                 strategy: str = 'quantile',
                 random_seed: int = 42,
                 feature_names: Optional[List[str]] = None):
        """__init__ function."""
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_seed = random_seed
        self.feature_names = feature_names
    
    def fit_transform(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      fold_seed: Optional[int] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """fit_transform function."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        kbd = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy=self.strategy,
            subsample=None,
            random_state=self.random_seed if self.strategy == 'kmeans' else None
        )
        bins = kbd.fit_transform(y_train.reshape(-1, 1)).flatten().astype(int)
        
        bin_counts = np.bincount(bins, minlength=self.n_bins)
        bin_weights = 1.0 / (bin_counts + 1e-8)
        raw_weights = bin_weights[bins]
        sample_weights = raw_weights / raw_weights.sum() * len(y_train)
        
        state = DataModuleState(
            scaler=scaler,
            bin_edges=kbd.bin_edges_,
            feature_std=np.std(X_train, axis=0),
            feature_names=self.feature_names,
            extra={'kbd': kbd, 'bin_counts': bin_counts}
        )
        
        return X_scaled, y_train.copy(), sample_weights, state

    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """transform function."""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None


# ============================================================
# ============================================================

class AugmentedDataModule(DataModule):
    """AugmentedDataModule class."""

    def __init__(
                 self, 
                 n_aug: int = 15,
                 feature_names: Optional[List[str]] = None,
                 random_seed: int = 42):
        """__init__ function."""
        self.n_aug = n_aug
        self.feature_names = feature_names
        self.random_seed = random_seed
        self._fit_count = 0

    def fit_transform(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      fold_seed: Optional[int] = None,
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """fit_transform function."""
        from .perturbation import get_rel_err_vector, epma_perturb

        if fold_seed is not None:
            effective_seed = fold_seed
        else:
            effective_seed = self.random_seed + self._fit_count
        self._fit_count += 1
        rng = np.random.RandomState(effective_seed)

        feature_std = np.std(X_train, axis=0)

        if self.feature_names is None:
            self.feature_names = self._infer_feature_names(X_train.shape[1])
        if len(self.feature_names) != X_train.shape[1]:
            raise ValueError("feature_names length must match X_train feature dimension")
        rel_err_vec = get_rel_err_vector(self.feature_names, strict=True)

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_list = [X_train]
        y_list = [y_train]

        for _ in range(self.n_aug):
            X_augmented = epma_perturb(X_train, rel_err_vec, rng)
            X_list.append(X_augmented)
            y_list.append(y_train)

        X_all = np.vstack(X_list)
        y_all = np.concatenate(y_list)
        X_scaled = scaler.transform(X_all)

        sample_weights = np.ones(len(y_all), dtype=np.float64)

        state = DataModuleState(
            scaler=scaler,
            feature_std=feature_std,
            feature_names=self.feature_names,
            extra={
                'n_aug': self.n_aug,
                'original_size': len(y_train)
            }
        )

        return X_scaled, y_all, sample_weights, state

    def transform(
                  self,
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """transform function."""
        X_scaled = state.scaler.transform(X_val)
        return X_scaled, None

# ============================================================
# ============================================================

def get_data_module(name: str, **kwargs) -> DataModule:
    """get_data_module function."""
    modules = {
        'raw': RawDataModule,
        'balanced': BalancedDataModule,
        'augmented': AugmentedDataModule,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"Unknown data module: {name}, supported: {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

