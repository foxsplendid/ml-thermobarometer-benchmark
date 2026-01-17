# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 模型模块
Models Module: 包含 BaseThermoModel、CatBoostWrapper、GroupAwareStacker

核心约束：
1. 所有模型实现统一的 fit/predict 接口
2. GroupAwareStacker 必须使用 inner GroupKFold 生成 OOF 元特征
3. 支持元特征缓存，基于数据哈希实现版本控制
"""

import os
import hashlib
import numpy as np
import joblib
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone, BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor


# ============================================================
# 抽象基类
# ============================================================

class BaseThermoModel(ABC, BaseEstimator, RegressorMixin):
    """
    温压计模型的抽象基类，定义统一的 fit/predict 接口
    兼容 sklearn 的 BaseEstimator 以支持 clone() 和 get_params/set_params
    """
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'BaseThermoModel':
        """训练模型（groups 仅 Stacking 需要）"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数（sklearn 兼容）"""
        return {}
    
    def set_params(self, **params) -> 'BaseThermoModel':
        """设置模型参数（sklearn 兼容）"""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# ============================================================
# CatBoost 封装
# ============================================================

class CatBoostWrapper(BaseThermoModel):
    """
    CatBoost 回归器封装，统一接口
    
    Parameters
    ----------
    iterations : int, default=1000
        迭代次数
    depth : int, default=6
        树深度
    learning_rate : float, default=0.03
        学习率
    loss_function : str, default='RMSE'
        损失函数
    random_seed : int, default=42
        随机种子
    silent : bool, default=True
        是否静默训练
    **kwargs : dict
        其他 CatBoostRegressor 参数
    """
    
    def __init__(self, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.03,
                 loss_function: str = 'RMSE', random_seed: int = 42, silent: bool = True, **kwargs):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.random_seed = random_seed
        self.silent = silent
        self.kwargs = kwargs
        self._model: Optional[CatBoostRegressor] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'CatBoostWrapper':
        """训练 CatBoost 模型（groups 参数不使用，仅为接口一致）"""
        self._model = CatBoostRegressor(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function=self.loss_function,
            random_seed=self.random_seed,
            verbose=not self.silent,
            allow_writing_files=False,  # 禁止生成 catboost_info 文件夹
            **self.kwargs
        )
        self._model.fit(X, y)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._model.predict(X)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'loss_function': self.loss_function,
            'random_seed': self.random_seed,
            'silent': self.silent,
            **self.kwargs
        }
    
    def set_params(self, **params) -> 'CatBoostWrapper':
        """设置模型参数"""
        for key, value in params.items():
            if key in ['iterations', 'depth', 'learning_rate', 'loss_function', 'random_seed', 'silent']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


# ============================================================
# ExtraTrees 封装
# ============================================================

class ExtraTreesWrapper(BaseThermoModel):
    """
    ExtraTrees 回归器封装，统一接口

    Parameters
    ----------
    n_estimators : int, default=200
        树的数量
    max_depth : int, default=10
        树的最大深度
    min_samples_split : int, default=2
        分裂内部节点所需的最小样本数
    random_state : int, default=42
        随机种子
    **kwargs : dict
        其他 ExtraTreesRegressor 参数
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 min_samples_split: int = 2, random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'ExtraTreesWrapper':
        """训练 ExtraTrees 模型（groups 参数不使用，仅为接口一致）"""
        from sklearn.ensemble import ExtraTreesRegressor

        self._model = ExtraTreesRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            n_jobs=-1,  # 使用所有CPU核心
            **self.kwargs
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'random_state': self.random_state,
            **self.kwargs
        }

    def set_params(self, **params) -> 'ExtraTreesWrapper':
        """设置模型参数"""
        for key, value in params.items():
            if key in ['n_estimators', 'max_depth', 'min_samples_split', 'random_state']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


# ============================================================
# XGBoost 封装
# ============================================================

class XGBoostWrapper(BaseThermoModel):
    """
    XGBoost 回归器封装，统一接口

    Parameters
    ----------
    n_estimators : int, default=200
        提升轮数
    max_depth : int, default=6
        树的最大深度
    learning_rate : float, default=0.05
        学习率
    random_state : int, default=42
        随机种子
    **kwargs : dict
        其他 XGBRegressor 参数
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 learning_rate: float = 0.05, random_state: int = 42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'XGBoostWrapper':
        """训练 XGBoost 模型（groups 参数不使用，仅为接口一致）"""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("请先安装 xgboost: pip install xgboost>=1.5")

        self._model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            n_jobs=-1,  # 使用所有CPU核心
            verbosity=0,  # 静默模式
            **self.kwargs
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self._model is None:
            raise RuntimeError("模型未训练，请先调用 fit()")
        return self._model.predict(X)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            **self.kwargs
        }

    def set_params(self, **params) -> 'XGBoostWrapper':
        """设置模型参数"""
        for key, value in params.items():
            if key in ['n_estimators', 'max_depth', 'learning_rate', 'random_state']:
                setattr(self, key, value)
            else:
                self.kwargs[key] = value
        return self


# ============================================================
# Group-aware OOF Stacking
# ============================================================

class GroupAwareStacker(BaseThermoModel):
    """
    Group-aware OOF Stacking 模型
    
    核心逻辑（fold-safe）：
    1. fit() 时：用 inner GroupKFold 在 outer-train 上生成 OOF 元特征 Z_train
    2. fit() 时：用 Z_train 训练 meta_model
    3. fit() 时：用全量 outer-train 重新训练所有 base_models（用于 predict）
    4. predict() 时：用已训练的 base_models 对 X_val 预测，拼成 Z_val
    5. predict() 时：用 meta_model.predict(Z_val) 得到最终预测
    
    Parameters
    ----------
    base_models : List[BaseThermoModel]
        基学习器列表
    meta_model : BaseThermoModel, optional
        元学习器（默认使用 CatBoostWrapper）
    inner_cv : int, default=5
        inner GroupKFold 的折数
    cache_dir : str, optional
        元特征缓存目录（为 None 则不缓存）
    use_scaler : bool, default=True
        是否对元特征进行标准化
    random_seed : int, default=42
        随机种子
    """
    
    def __init__(self, base_models: List[BaseThermoModel], meta_model: Optional[BaseThermoModel] = None,
                 inner_cv: int = 5, cache_dir: Optional[str] = None, use_scaler: bool = True, random_seed: int = 42):
        self.base_models = base_models
        self.meta_model = meta_model if meta_model is not None else CatBoostWrapper(iterations=500, depth=4)
        self.inner_cv = inner_cv
        self.cache_dir = cache_dir
        self.use_scaler = use_scaler
        self.random_seed = random_seed
        
        # 训练后保存的状态
        self._fitted_base_models: List[BaseThermoModel] = []  # 在全量 outer-train 上训练的基模型
        self._meta_scaler: Optional[StandardScaler] = None     # 元特征标准化器
        self._is_fitted = False
    
    def _compute_data_hash(self, X: np.ndarray, y: np.ndarray) -> str:
        """计算数据指纹，用于缓存版本控制"""
        data_bytes = np.concatenate([X.flatten(), y.flatten()]).tobytes()
        return hashlib.md5(data_bytes).hexdigest()[:12]
    
    def _get_cache_path(self, data_hash: str, model_idx: int) -> str:
        """获取缓存文件路径"""
        if self.cache_dir is None:
            return ""
        os.makedirs(self.cache_dir, exist_ok=True)
        return os.path.join(self.cache_dir, f"oof_model{model_idx}_{data_hash}.npy")
    
    def _generate_oof_predictions(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                                   data_hash: str) -> np.ndarray:
        """
        使用 inner GroupKFold 生成 OOF 元特征
        
        返回：Z_train, shape = (n_samples, n_base_models)
        """
        n_samples = X.shape[0]
        n_models = len(self.base_models)
        Z_train = np.zeros((n_samples, n_models))
        
        gkf = GroupKFold(n_splits=self.inner_cv)
        
        for model_idx, base_model in enumerate(self.base_models):
            cache_path = self._get_cache_path(data_hash, model_idx)
            
            # 尝试从缓存加载
            if cache_path and os.path.exists(cache_path):
                Z_train[:, model_idx] = np.load(cache_path)
                continue
            
            # 生成 OOF 预测
            oof_preds = np.zeros(n_samples)
            for train_idx, val_idx in gkf.split(X, y, groups):
                X_tr, X_val = X[train_idx], X[val_idx]
                y_tr = y[train_idx]
                
                # 克隆模型以避免状态污染
                fold_model = clone(base_model)
                fold_model.fit(X_tr, y_tr)
                oof_preds[val_idx] = fold_model.predict(X_val)
            
            Z_train[:, model_idx] = oof_preds
            
            # 保存缓存
            if cache_path:
                np.save(cache_path, oof_preds)
        
        return Z_train
    
    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray] = None) -> 'GroupAwareStacker':
        """
        训练 Stacking 模型
        
        步骤：
        1. 用 inner GroupKFold 生成 Z_train（OOF 元特征）
        2. 对 Z_train 进行标准化
        3. 用标准化后的 Z_train 训练 meta_model
        4. 用全量数据重新训练所有 base_models（用于 predict）
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵（outer-train）
        y : np.ndarray, shape (n_samples,)
            目标值
        groups : np.ndarray, shape (n_samples,)
            分组标签（必须提供！）
        """
        if groups is None:
            raise ValueError("GroupAwareStacker 必须提供 groups 参数！")
        
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        groups = np.asarray(groups)
        
        # 1. 生成 OOF 元特征
        data_hash = self._compute_data_hash(X, y)
        Z_train = self._generate_oof_predictions(X, y, groups, data_hash)
        
        # 2. 标准化元特征
        if self.use_scaler:
            self._meta_scaler = StandardScaler()
            Z_train_scaled = self._meta_scaler.fit_transform(Z_train)
        else:
            Z_train_scaled = Z_train
        
        # 3. 训练元模型
        self.meta_model.fit(Z_train_scaled, y)
        
        # 4. 用全量数据重新训练基模型（用于 predict）
        self._fitted_base_models = []
        for base_model in self.base_models:
            fitted_model = clone(base_model)
            fitted_model.fit(X, y)
            self._fitted_base_models.append(fitted_model)
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        步骤：
        1. 用已训练的 base_models 预测，得到 Z_val
        2. 对 Z_val 进行标准化（使用 fit 时的 scaler）
        3. 用 meta_model 预测最终结果
        """
        if not self._is_fitted:
            raise RuntimeError("模型未训练，请先调用 fit()")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_models = len(self._fitted_base_models)
        
        # 1. 基模型预测
        Z_val = np.zeros((n_samples, n_models))
        for model_idx, fitted_model in enumerate(self._fitted_base_models):
            Z_val[:, model_idx] = fitted_model.predict(X)
        
        # 2. 标准化
        if self.use_scaler and self._meta_scaler is not None:
            Z_val_scaled = self._meta_scaler.transform(Z_val)
        else:
            Z_val_scaled = Z_val
        
        # 3. 元模型预测
        return self.meta_model.predict(Z_val_scaled)
    
    def get_oof_predictions(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        """
        获取 OOF 预测（用于偏差校正器拟合）
        
        注意：此方法生成的是 inner OOF 预测，可安全用于偏差校正器拟合
        """
        X = np.asarray(X)
        y = np.asarray(y).ravel()
        groups = np.asarray(groups)
        
        data_hash = self._compute_data_hash(X, y)
        Z_train = self._generate_oof_predictions(X, y, groups, data_hash)
        
        # 标准化后用 meta_model 预测
        if self.use_scaler and self._meta_scaler is not None:
            Z_train_scaled = self._meta_scaler.transform(Z_train)
        else:
            Z_train_scaled = Z_train
        
        return self.meta_model.predict(Z_train_scaled)
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """获取模型参数"""
        return {
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'inner_cv': self.inner_cv,
            'cache_dir': self.cache_dir,
            'use_scaler': self.use_scaler,
            'random_seed': self.random_seed
        }


# ============================================================
# 模型工厂函数
# ============================================================

def get_model(name: str, **kwargs) -> BaseThermoModel:
    """
    模型工厂函数
    
    Parameters
    ----------
    name : str
        模型名称，支持 'catboost', 'stacking'
    **kwargs : dict
        模型参数
    
    Returns
    -------
    BaseThermoModel
        模型实例
    
    Examples
    --------
    >>> model = get_model('catboost', iterations=1000, depth=6)
    >>> stacker = get_model('stacking', base_models=[...], inner_cv=5)
    """
    name = name.lower().strip()

    if name == 'catboost':
        return CatBoostWrapper(**kwargs)
    elif name == 'extratrees':
        return ExtraTreesWrapper(**kwargs)
    elif name == 'xgboost':
        return XGBoostWrapper(**kwargs)
    elif name == 'stacking':
        return GroupAwareStacker(**kwargs)
    else:
        raise ValueError(f"未知模型类型: {name}，支持 'catboost', 'extratrees', 'xgboost', 'stacking'")


def create_default_stacker(cache_dir: Optional[str] = None, inner_cv: int = 5, use_heterogeneous: bool = True) -> GroupAwareStacker:
    """
    创建默认配置的 Stacking 模型

    Parameters
    ----------
    cache_dir : str, optional
        元特征缓存目录
    inner_cv : int, default=5
        内层 CV 折数
    use_heterogeneous : bool, default=True
        是否使用异构基学习器（True: ExtraTrees+XGBoost+CatBoost，False: 3个CatBoost）

    Returns
    -------
    GroupAwareStacker
        配置好的 Stacking 模型
    """
    if use_heterogeneous:
        # 异构基学习器：ExtraTrees + XGBoost + CatBoost
        base_models = [
            ExtraTreesWrapper(n_estimators=200, max_depth=10, random_state=42),
            XGBoostWrapper(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42),
            CatBoostWrapper(iterations=200, depth=6, learning_rate=0.05, random_seed=42),
        ]
    else:
        # 同构基学习器：3个不同参数的 CatBoost
        base_models = [
            CatBoostWrapper(iterations=500, depth=4, learning_rate=0.05),
            CatBoostWrapper(iterations=800, depth=6, learning_rate=0.03),
            CatBoostWrapper(iterations=1000, depth=8, learning_rate=0.02),
        ]

    meta_model = CatBoostWrapper(iterations=100, depth=4, learning_rate=0.05)

    return GroupAwareStacker(
        base_models=base_models,
        meta_model=meta_model,
        inner_cv=inner_cv,
        cache_dir=cache_dir,
        use_scaler=True
    )


# ============================================================
# 使用示例（仅供参考，实际运行请在 Jupyter 中执行）
# ============================================================

if __name__ == "__main__":
    # 示例：单模型
    print("=== 单模型示例 ===")
    model = get_model('catboost', iterations=100, depth=4, silent=True)
    print(f"创建模型: {type(model).__name__}")
    print(f"参数: {model.get_params()}")
    
    # 示例：Stacking
    print("\n=== Stacking 示例 ===")
    stacker = create_default_stacker(cache_dir=None, inner_cv=3)
    print(f"创建模型: {type(stacker).__name__}")
    print(f"基模型数量: {len(stacker.base_models)}")
    print(f"元模型: {type(stacker.meta_model).__name__}")
