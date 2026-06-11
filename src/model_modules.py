# -*- coding: utf-8 -*-
"""Model-module implementations for baseline and ensemble regressors."""

import logging
import os
import time
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .interfaces import ModelModule

logger = logging.getLogger(__name__)


# ============================================================
# ============================================================

def _get_default_n_jobs() -> int:
    """Return the default n_jobs for sklearn parallel estimators.

    Delegates to :func:`src.runtime.suggest_n_jobs` so that ML_N_JOBS,
    ML_RESERVE_CORES and ML_OUTER_PROCS are honoured consistently across
    model modules and cooperative with sibling processes.
    """
    from .runtime import suggest_n_jobs
    return suggest_n_jobs("model")


def _get_catboost_task_type(
    task_type: str = 'auto',
    gpu_devices: str = '0',
    n_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve CatBoost ``task_type`` / ``devices`` based on user preference,
    GPU availability and dataset size.

    Decision rules:
      - ``'CPU'`` -> always CPU.
      - ``'GPU'`` -> always GPU (errors out if no device).
      - ``'auto'`` ->
          * no CUDA device -> CPU.
          * dataset smaller than ``ML_CATBOOST_GPU_MIN_SAMPLES`` (default 5000)
            -> CPU. CatBoost GPU has non-trivial startup overhead and CPU
            is faster on small problems.
          * otherwise -> GPU.

    ``n_samples=None`` skips the size gate (back-compat with V7 callers that
    decide at init time).
    """
    from catboost.utils import get_gpu_device_count

    task_type_upper = task_type.upper().strip()

    if task_type_upper == 'CPU':
        return {}

    if task_type_upper == 'GPU':
        return {'task_type': 'GPU', 'devices': gpu_devices}

    # 'auto' path
    if get_gpu_device_count() < 1:
        return {}

    if n_samples is not None:
        threshold = os.environ.get('ML_CATBOOST_GPU_MIN_SAMPLES', '5000')
        try:
            threshold_int = int(threshold)
        except ValueError:
            threshold_int = 5000
        if n_samples < threshold_int:
            return {}  # small dataset: CPU is faster

    return {'task_type': 'GPU', 'devices': gpu_devices}


# ============================================================
# ============================================================

class ExtraTreesModel(ModelModule):
    """ExtraTreesModel class."""

    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 15,
                 min_samples_split: int = 5,
                 n_jobs: Optional[int] = None,
                 random_seed: int = 42,
                 **kwargs):
        if n_jobs is None:
            n_jobs = _get_default_n_jobs()
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'n_jobs': n_jobs,
            'random_state': random_seed,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        from sklearn.ensemble import ExtraTreesRegressor
        
        start_time = time.time()
        
        model = ExtraTreesRegressor(**self.params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        return model.predict(X)
    
    def get_feature_importance(self, model: Any) -> np.ndarray:
        """get_feature_importance function."""
        return model.feature_importances_


# ============================================================
# ============================================================

# Standalone defaults for direct CatBoostModel() construction.
# The experiment harness builds parameters via experiment_params.build_model_params,
# which reads config.CatBoostConfig — that dataclass is the single source of truth.
# Keep these values aligned with config.CatBoostConfig.
_CATBOOST_DEFAULTS: Dict[str, Any] = {
    "iterations": 1000,
    "depth": 6,
    "learning_rate": 0.03,
    "loss_function": "RMSE",
    "task_type": "auto",
    "gpu_devices": "0",
}


class CatBoostModel(ModelModule):
    """CatBoostModel class."""

    def __init__(self,
                 iterations: int = _CATBOOST_DEFAULTS["iterations"],
                 depth: int = _CATBOOST_DEFAULTS["depth"],
                 learning_rate: float = _CATBOOST_DEFAULTS["learning_rate"],
                 loss_function: str = _CATBOOST_DEFAULTS["loss_function"],
                 random_seed: int = 42,
                 silent: bool = True,
                 task_type: str = _CATBOOST_DEFAULTS["task_type"],
                 gpu_devices: str = _CATBOOST_DEFAULTS["gpu_devices"],
                 **kwargs):
        # Store preferences; resolve device + thread_count at fit() time
        # when we know the training set size (H3).
        self._task_type_pref = task_type
        self._gpu_devices_pref = gpu_devices
        self._user_kwargs = dict(kwargs)

        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'loss_function': loss_function,
            'random_seed': random_seed,
            'verbose': False if silent else 100,
            'allow_writing_files': False,
        }
        self._training_time = 0.0

    def _resolve_runtime_params(self, n_samples: int) -> Dict[str, Any]:
        """Compose final CatBoost params using runtime info (data size, GPU)."""
        gpu_params = _get_catboost_task_type(
            self._task_type_pref, self._gpu_devices_pref, n_samples=n_samples
        )
        thread_kw: Dict[str, Any] = {}
        if 'task_type' not in gpu_params and 'thread_count' not in self._user_kwargs:
            thread_kw['thread_count'] = _get_default_n_jobs()
        return {**self.params, **gpu_params, **thread_kw, **self._user_kwargs}

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        from catboost import CatBoostRegressor, Pool

        start_time = time.time()

        resolved = self._resolve_runtime_params(n_samples=len(y_train))
        train_pool = Pool(X_train, y_train, weight=sample_weights)
        model = CatBoostRegressor(**resolved)
        model.fit(train_pool)

        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        return model.predict(X)
    
    def get_feature_importance(self, model: Any) -> np.ndarray:
        """get_feature_importance function."""
        return model.get_feature_importance()


# ============================================================
# ============================================================

class RandomForestModel(ModelModule):
    """RandomForestModel class."""
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 15,
                 min_samples_split: int = 5,
                 n_jobs: Optional[int] = None,
                 random_seed: int = 42,
                 **kwargs):
        if n_jobs is None:
            n_jobs = _get_default_n_jobs()
        
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'n_jobs': n_jobs,
            'random_state': random_seed,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        from sklearn.ensemble import RandomForestRegressor
        
        start_time = time.time()
        
        model = RandomForestRegressor(**self.params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        return model.predict(X)

    def get_feature_importance(self, model: Any) -> np.ndarray:
        """get_feature_importance function."""
        return model.feature_importances_


# ============================================================
# ============================================================

# Registry mapping model keys to constructors, used by StrictOOFStacking to build base learners dynamically.
def _build_base_model_registry(random_seed: int) -> Dict[str, Any]:
    """_build_base_model_registry function."""
    return {
        'ert':      lambda p: ExtraTreesModel(random_seed=random_seed, **p),
        'catboost': lambda p: CatBoostModel(random_seed=random_seed, **p),
        'rf':       lambda p: RandomForestModel(random_seed=random_seed, **p),
        'ridge':    lambda p: RidgeModel(**p),
    }


# ============================================================
# ============================================================

class StrictOOFStacking(ModelModule):
    """StrictOOFStacking class."""
    
    def __init__(self,
                 base_models: Optional[List[ModelModule]] = None,
                 base_model_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 meta_model: Optional[ModelModule] = None,
                 inner_cv: int = 5,
                 use_meta_scaler: bool = True,
                 random_seed: int = 42,
                 inner_parallel: Optional[int] = None):
        """__init__ function.

        inner_parallel (H7): number of concurrent (fold x base-model) fit
        workers. None (default) reads ENV ML_STACKING_PARALLEL (default 1);
        <=1 keeps the V7/V8 sequential loop verbatim. When >1, the worker
        count is clamped to suggest_n_jobs('inner_loop') and each base
        model's thread budget defaults to budget // workers — but params
        that already carry a thread count are left untouched (notably
        config.ModelDefaults.ert pins n_jobs=4), so the no-oversubscription
        guarantee only holds for workers <= budget // <largest explicit
        thread count>; a warning is logged when an explicit value exceeds
        the per-worker share. CatBoost with task_type='auto' is pinned to
        CPU in parallel mode (concurrent workers must not share one GPU);
        an explicit 'GPU' is honored with a warning.
        Note: under parallel fits the per-base-module _training_time field
        is subject to a benign race and carries no meaning.
        """
        from .runtime import _env_int, suggest_n_jobs

        self.inner_cv = inner_cv
        self.use_meta_scaler = use_meta_scaler
        self.random_seed = random_seed

        if inner_parallel is None:
            inner_parallel = _env_int("ML_STACKING_PARALLEL", 1) or 1
        inner_parallel = max(1, int(inner_parallel))
        if inner_parallel > 1:
            # per_fit floors at 1 below, so cap the workers themselves to
            # keep workers x per_fit within the single-fit envelope.
            inner_parallel = min(inner_parallel, suggest_n_jobs("inner_loop"))
        self.inner_parallel = inner_parallel

        if base_models is not None:
            self.base_models = base_models
            if self.inner_parallel > 1:
                logger.warning(
                    "inner_parallel=%d with pre-built base_models: thread "
                    "budgets are not adjusted; the caller is responsible "
                    "for avoiding oversubscription.", self.inner_parallel
                )
        elif base_model_params is not None:
            if self.inner_parallel > 1:
                # H7: split the inner-loop budget across workers so that
                # workers x per-fit threads <= suggest_n_jobs('inner_loop').
                per_fit = max(1, suggest_n_jobs("inner_loop") // self.inner_parallel)
                adjusted: Dict[str, Dict[str, Any]] = {}
                for key, params in base_model_params.items():
                    p = dict(params)
                    if key in ("ert", "rf"):
                        kept = p.setdefault("n_jobs", per_fit)
                        if isinstance(kept, int) and (kept < 0 or kept > per_fit):
                            logger.warning(
                                "H7: base model %r keeps explicit n_jobs=%r above the "
                                "per-worker budget %d; %d workers may oversubscribe.",
                                key, kept, per_fit, self.inner_parallel,
                            )
                    elif key == "catboost":
                        # Lands in _user_kwargs and suppresses the fit-time
                        # suggest_n_jobs('model') injection.
                        kept = p.setdefault("thread_count", per_fit)
                        if isinstance(kept, int) and (kept < 0 or kept > per_fit):
                            logger.warning(
                                "H7: base model 'catboost' keeps explicit thread_count=%r "
                                "above the per-worker budget %d; %d workers may "
                                "oversubscribe.", kept, per_fit, self.inner_parallel,
                            )
                        # Concurrent workers must not auto-select one shared GPU.
                        task_pref = str(p.get("task_type", "auto")).strip().lower()
                        if task_pref == "auto":
                            p["task_type"] = "CPU"
                        elif task_pref == "gpu":
                            logger.warning(
                                "H7: inner_parallel=%d with explicit task_type='GPU': "
                                "concurrent trainings on one device are the caller's "
                                "responsibility.", self.inner_parallel,
                            )
                    adjusted[key] = p
                base_model_params = adjusted
            registry = _build_base_model_registry(random_seed)
            self.base_models = []
            for key, params in base_model_params.items():
                if key not in registry:
                    raise ValueError(
                        f"Unknown base model key: {key!r}. "
                        f"Supported keys: {list(registry.keys())}"
                    )
                self.base_models.append(registry[key](params))
        else:
            raise ValueError("Either base_models or base_model_params must be provided")
        
        if meta_model is None:
            self.meta_model = RidgeModel(alpha=1.0)
        else:
            self.meta_model = meta_model
        
        self._fitted_base_models: List[Any] = []
        self._meta_scaler: Optional[StandardScaler] = None
        self._oof_meta_features: Optional[np.ndarray] = None
        self._training_time = 0.0
        self._base_correlations: Optional[np.ndarray] = None
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """fit function."""
        start_time = time.time()
        
        n_samples = len(y_train)
        n_base = len(self.base_models)
        
        if stratify_labels is not None:
            splitter = StratifiedKFold(
                n_splits=self.inner_cv,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X_train, stratify_labels)
        else:
            splitter = KFold(
                n_splits=self.inner_cv,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X_train)

        oof_meta = np.zeros((n_samples, n_base))

        if self.inner_parallel <= 1:
            for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(split_iter):
                X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
                X_iv = X_train[inner_val_idx]

                w_it = sample_weights[inner_train_idx] if sample_weights is not None else None

                for j, base_module in enumerate(self.base_models):
                    model = base_module.fit(X_it, y_it, w_it)
                    oof_meta[inner_val_idx, j] = base_module.predict(model, X_iv)
        else:
            # H7: flatten (fold x base) into independent tasks. Each task
            # constructs its own estimator from a fixed seed, so execution
            # order cannot change any RNG stream; val_idx slices are pairwise
            # disjoint per column, so the gather is order-independent.
            from joblib import Parallel, delayed

            splits = list(split_iter)

            def _fit_one(fold_idx: int, j: int):
                inner_train_idx, inner_val_idx = splits[fold_idx]
                w_it = sample_weights[inner_train_idx] if sample_weights is not None else None
                base_module = self.base_models[j]
                model = base_module.fit(
                    X_train[inner_train_idx], y_train[inner_train_idx], w_it
                )
                preds = base_module.predict(model, X_train[inner_val_idx])
                return j, inner_val_idx, preds

            tasks = [(f, j) for f in range(len(splits)) for j in range(n_base)]
            task_results = Parallel(
                n_jobs=min(self.inner_parallel, len(tasks)), backend="threading"
            )(delayed(_fit_one)(f, j) for f, j in tasks)
            for j, inner_val_idx, preds in task_results:
                oof_meta[inner_val_idx, j] = preds
        
        self._oof_meta_features = oof_meta.copy()
        
        self._base_correlations = np.corrcoef(oof_meta.T)
        
        if self.use_meta_scaler:
            self._meta_scaler = StandardScaler()
            oof_meta_scaled = self._meta_scaler.fit_transform(oof_meta)
        else:
            oof_meta_scaled = oof_meta
        
        meta_fitted = self.meta_model.fit(oof_meta_scaled, y_train, sample_weights)

        if self.inner_parallel <= 1:
            self._fitted_base_models = []
            for base_module in self.base_models:
                model = base_module.fit(X_train, y_train, sample_weights)
                self._fitted_base_models.append(model)
        else:
            from joblib import Parallel, delayed
            self._fitted_base_models = list(Parallel(
                n_jobs=min(self.inner_parallel, n_base), backend="threading"
            )(
                delayed(base_module.fit)(X_train, y_train, sample_weights)
                for base_module in self.base_models
            ))
        
        self._training_time = time.time() - start_time
        
        return {
            'meta': meta_fitted,
            'base': self._fitted_base_models,
            'meta_scaler': self._meta_scaler
        }
    
    def predict(self, model_dict: Dict[str, Any], X: np.ndarray) -> np.ndarray:
        """predict function."""
        meta_features = np.column_stack([
            base_module.predict(fitted, X)
            for base_module, fitted in zip(self.base_models, model_dict['base'])
        ])
        
        if model_dict['meta_scaler'] is not None:
            meta_scaled = model_dict['meta_scaler'].transform(meta_features)
        else:
            meta_scaled = meta_features
        
        return self.meta_model.predict(model_dict['meta'], meta_scaled)
    
    def get_oof_predictions(self, model: Dict[str, Any]) -> np.ndarray:
        """get_oof_predictions function."""
        if self._oof_meta_features is None:
            raise RuntimeError("get_oof_predictions() called before fit(); call fit() first.")

        if self._meta_scaler is not None:
            oof_scaled = self._meta_scaler.transform(self._oof_meta_features)
        else:
            oof_scaled = self._oof_meta_features

        return self.meta_model.predict(model['meta'], oof_scaled)
    
    def get_base_correlations(self) -> Optional[np.ndarray]:
        """get_base_correlations function."""
        return self._base_correlations
    
    def get_meta_weights(self, model_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """get_meta_weights function."""
        return self.meta_model.get_weights(model_dict['meta'])

    def get_feature_importance(self, model_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """get_feature_importance function."""
        if not self._fitted_base_models:
            return None

        meta_weights = self.get_meta_weights(model_dict)
        if meta_weights is None:
            meta_weights = np.ones(len(self._fitted_base_models)) / len(self._fitted_base_models)
        else:
            meta_weights = np.abs(meta_weights)
            meta_weights = meta_weights / meta_weights.sum()

        importances_list = []
        for i, (base_mod, fitted_model) in enumerate(zip(self.base_models, self._fitted_base_models)):
            try:
                if hasattr(base_mod, 'get_feature_importance'):
                    imp = base_mod.get_feature_importance(fitted_model)
                    if imp is not None:
                        importances_list.append((meta_weights[i], imp))
            except Exception:
                continue

        if not importances_list:
            return None

        total_weight = sum(w for w, _ in importances_list)
        weighted_imp = sum(w * imp for w, imp in importances_list) / total_weight
        return weighted_imp


# ============================================================
# ============================================================

class RidgeModel(ModelModule):
    """RidgeModel class."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        from sklearn.linear_model import Ridge
        
        start_time = time.time()
        
        model = Ridge(alpha=self.alpha)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        return model.predict(X)
    
    def get_weights(self, model: Any) -> np.ndarray:
        """get_weights function."""
        return model.coef_


# ============================================================
# ============================================================

def get_model_module(name: str, **kwargs) -> ModelModule:
    """get_model_module function."""
    modules = {
        'ert': ExtraTreesModel,
        'extratrees': ExtraTreesModel,
        'catboost': CatBoostModel,
        'cb': CatBoostModel,
        'rf': RandomForestModel,
        'randomforest': RandomForestModel,
        'stacking': StrictOOFStacking,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"Unknown model module: {name}, supported: {list(set(modules.values()))}")

    if name_lower == 'stacking':
        if 'base_models' not in kwargs and 'base_model_params' not in kwargs:
            raise ValueError(
                "get_model_module('stacking') requires either base_models or base_model_params; "
                "pass base_model_params={'ert': {...}, 'catboost': {...}, 'rf': {...}} explicitly."
            )

    return modules[name_lower](**kwargs)

