# -*- coding: utf-8 -*-
"""Model-module implementations for baseline and ensemble regressors."""

import os
import time
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .interfaces import ModelModule


# ============================================================
# ============================================================

def _get_default_n_jobs() -> int:
    """Return the default n_jobs for sklearn parallel estimators.

    读取环境变量 ``ML_N_JOBS``（整数）以允许外部并行脚本精确控制每个模型
    占用的线程数，避免多实验并发时线程过度竞争。
    未设置时默认 -1（使用全部逻辑核心）。
    """
    env_val = os.environ.get('ML_N_JOBS')
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return -1  # 使用全部可用核心（移除原 Windows 限制）


def _get_catboost_task_type(task_type: str = 'auto', gpu_devices: str = '0') -> Dict[str, Any]:
    """_get_catboost_task_type function."""
    from catboost.utils import get_gpu_device_count

    task_type_upper = task_type.upper().strip()

    if task_type_upper == 'CPU':
        return {}

    if task_type_upper == 'GPU':
        return {'task_type': 'GPU', 'devices': gpu_devices}

    if get_gpu_device_count() >= 1:
        return {'task_type': 'GPU', 'devices': gpu_devices}
    return {}


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

class CatBoostModel(ModelModule):
    """CatBoostModel class."""
    
    def __init__(self,
                 iterations: int = 1000,
                 depth: int = 6,
                 learning_rate: float = 0.03,
                 loss_function: str = 'RMSE',
                 random_seed: int = 42,
                 silent: bool = True,
                 task_type: str = 'auto',
                 gpu_devices: str = '0',
                 **kwargs):
        gpu_params = _get_catboost_task_type(task_type, gpu_devices)

        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'loss_function': loss_function,
            'random_seed': random_seed,
            'verbose': False if silent else 100,
            'allow_writing_files': False,
            **gpu_params,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        from catboost import CatBoostRegressor, Pool
        
        start_time = time.time()
        
        train_pool = Pool(X_train, y_train, weight=sample_weights)
        model = CatBoostRegressor(**self.params)
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

class SVRModel(ModelModule):
    """SVRModel — 核方法回归，为 Stacking 提供与树模型正交的归纳偏置。

    输入应已标准化（Pipeline 通过 DataModule 在 fit_transform 中完成缩放）。
    SVR 本身无随机性，random_seed 参数仅为保持接口统一，不影响输出。

    自动规模切换策略：
      n_samples <= linear_threshold : 使用 SVR(RBF)，O(n²) 但核方法精度高
      n_samples >  linear_threshold : 自动切换到 LinearSVR，O(n)，避免增强
                                       数据集上的内存/时间爆炸（~30k 样本时
                                       RBF 核矩阵需 5GB+ 内存，耗时数小时）
    """

    # 样本数超过此阈值时自动切换 LinearSVR
    LINEAR_THRESHOLD: int = 5_000

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 10.0,
                 epsilon: float = 0.1,
                 gamma: str = 'scale',
                 linear_threshold: Optional[int] = None,
                 random_seed: int = 42,
                 **kwargs):
        """__init__ function."""
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.linear_threshold = linear_threshold if linear_threshold is not None \
            else self.LINEAR_THRESHOLD
        self.extra_kwargs = kwargs
        self._training_time = 0.0

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """fit function."""
        start_time = time.time()

        if X_train.shape[0] > self.linear_threshold:
            # 大数据集：LinearSVR，O(n) 复杂度
            from sklearn.svm import LinearSVR
            model = LinearSVR(
                C=self.C,
                epsilon=self.epsilon,
                max_iter=10000,
                **{k: v for k, v in self.extra_kwargs.items()
                   if k not in ('gamma',)}  # LinearSVR 不接受 gamma
            )
        else:
            # 小数据集：RBF-SVR，保留核方法归纳偏置
            from sklearn.svm import SVR
            model = SVR(
                kernel=self.kernel,
                C=self.C,
                epsilon=self.epsilon,
                gamma=self.gamma,
                **self.extra_kwargs,
            )

        model.fit(X_train, y_train, sample_weight=sample_weights)
        self._training_time = time.time() - start_time
        return model

    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """predict function."""
        return model.predict(X)

    def get_feature_importance(self, model: Any) -> np.ndarray:
        """get_feature_importance function."""
        # SVR/LinearSVR 无树结构，返回空数组
        return np.array([])


# ============================================================
# ============================================================

# 模型键到构造器的注册表，供 StrictOOFStacking 动态构建基学习器使用
def _build_base_model_registry(random_seed: int) -> Dict[str, Any]:
    """_build_base_model_registry function."""
    return {
        'ert':      lambda p: ExtraTreesModel(random_seed=random_seed, **p),
        'catboost': lambda p: CatBoostModel(random_seed=random_seed, **p),
        'rf':       lambda p: RandomForestModel(random_seed=random_seed, **p),
        'svr':      lambda p: SVRModel(random_seed=random_seed, **p),
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
                 random_seed: int = 42):
        """__init__ function."""
        self.inner_cv = inner_cv
        self.use_meta_scaler = use_meta_scaler
        self.random_seed = random_seed
        
        if base_models is not None:
            self.base_models = base_models
        elif base_model_params is not None:
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
        
        for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(split_iter):
            X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
            X_iv = X_train[inner_val_idx]
            
            w_it = sample_weights[inner_train_idx] if sample_weights is not None else None
            
            for j, base_module in enumerate(self.base_models):
                model = base_module.fit(X_it, y_it, w_it)
                oof_meta[inner_val_idx, j] = base_module.predict(model, X_iv)
        
        self._oof_meta_features = oof_meta.copy()
        
        self._base_correlations = np.corrcoef(oof_meta.T)
        
        if self.use_meta_scaler:
            self._meta_scaler = StandardScaler()
            oof_meta_scaled = self._meta_scaler.fit_transform(oof_meta)
        else:
            oof_meta_scaled = oof_meta
        
        meta_fitted = self.meta_model.fit(oof_meta_scaled, y_train, sample_weights)
        
        self._fitted_base_models = []
        for base_module in self.base_models:
            model = base_module.fit(X_train, y_train, sample_weights)
            self._fitted_base_models.append(model)
        
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
        'svr': SVRModel,
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

