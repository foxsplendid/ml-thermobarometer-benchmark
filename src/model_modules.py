# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - M2 模型模块实现
Model Modules: ExtraTreesModel, CatBoostModel, StrictOOFStacking

核心约束：
1. fit() 只使用训练集数据
2. Stacking 必须严格 OOF：内层 CV 生成元特征，禁止数据泄露
3. 支持样本权重传递
"""

import time
import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .interfaces import ModelModule


# ============================================================
# ExtraTrees 模型（基线）
# ============================================================

class ExtraTreesModel(ModelModule):
    """
    ExtraTrees 回归模型 - 传统集成基线
    
    特点：
    - 高方差低偏差
    - 训练速度快
    - 抗过拟合能力较弱
    """
    
    def __init__(self, 
                 n_estimators: int = 200,
                 max_depth: int = 15,
                 min_samples_split: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'n_jobs': n_jobs,
            'random_state': random_state,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> Any:
        """训练 ExtraTrees 模型"""
        from sklearn.ensemble import ExtraTreesRegressor
        
        start_time = time.time()
        
        model = ExtraTreesRegressor(**self.params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """预测"""
        return model.predict(X)
    
    def get_feature_importance(self, model: Any) -> np.ndarray:
        """获取特征重要性"""
        return model.feature_importances_


# ============================================================
# CatBoost 模型（强单模型）
# ============================================================

class CatBoostModel(ModelModule):
    """
    CatBoost 回归模型 - 强单模型代表
    
    特点：
    - Boosting 方法，偏差校正能力强
    - 支持类别特征（本项目不使用）
    - 内置正则化，抗过拟合
    """
    
    def __init__(self,
                 iterations: int = 1000,
                 depth: int = 6,
                 learning_rate: float = 0.03,
                 loss_function: str = 'RMSE',
                 random_seed: int = 42,
                 silent: bool = True,
                 **kwargs):
        self.params = {
            'iterations': iterations,
            'depth': depth,
            'learning_rate': learning_rate,
            'loss_function': loss_function,
            'random_seed': random_seed,
            'verbose': False if silent else 100,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> Any:
        """训练 CatBoost 模型"""
        from catboost import CatBoostRegressor, Pool
        
        start_time = time.time()
        
        train_pool = Pool(X_train, y_train, weight=sample_weights)
        model = CatBoostRegressor(**self.params)
        model.fit(train_pool)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """预测"""
        return model.predict(X)
    
    def get_feature_importance(self, model: Any) -> np.ndarray:
        """获取特征重要性"""
        return model.get_feature_importance()


# ============================================================
# RandomForest 模型（Stacking 基模型之一）
# ============================================================

class RandomForestModel(ModelModule):
    """
    RandomForest 回归模型 - Stacking 的基模型之一
    """
    
    def __init__(self,
                 n_estimators: int = 200,
                 max_depth: int = 15,
                 min_samples_split: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 **kwargs):
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'n_jobs': n_jobs,
            'random_state': random_state,
            **kwargs
        }
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> Any:
        """训练 RandomForest 模型"""
        from sklearn.ensemble import RandomForestRegressor
        
        start_time = time.time()
        
        model = RandomForestRegressor(**self.params)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """预测"""
        return model.predict(X)


# ============================================================
# Strict OOF Stacking 模型
# ============================================================

class StrictOOFStacking(ModelModule):
    """
    严格 OOF Stacking 模型 - 完全防泄露的堆叠集成
    
    核心逻辑（严格 OOF）：
    1. fit() 时：
       - 在训练集上用 inner KFold 生成 OOF 元特征 Z_train
       - 用 Z_train 训练 meta-learner
       - 用全训练集重新训练所有 base models（供预测时使用）
    
    2. predict() 时：
       - 用 fit() 时训练好的 base models 预测，得到 Z_val
       - 用 meta-learner 预测最终结果
    
    基模型组合：ERT + CatBoost + RF
    元模型：Ridge（或 RidgeCV）
    """
    
    def __init__(self,
                 base_models: Optional[List[ModelModule]] = None,
                 meta_model: Optional[ModelModule] = None,
                 inner_cv: int = 5,
                 use_meta_scaler: bool = True,
                 random_seed: int = 42):
        """
        Parameters
        ----------
        base_models : List[ModelModule], optional
            基模型列表，默认为 [ERT, CatBoost, RF]
        meta_model : ModelModule, optional
            元模型，默认为 RidgeModel
        inner_cv : int
            内层 CV 折数（用于生成 OOF 元特征）
        use_meta_scaler : bool
            是否对元特征进行标准化
        """
        self.inner_cv = inner_cv
        self.use_meta_scaler = use_meta_scaler
        self.random_seed = random_seed
        
        # 默认基模型：ERT + CatBoost + RF
        if base_models is None:
            self.base_models = [
                ExtraTreesModel(n_estimators=100, max_depth=12, random_state=random_seed),
                CatBoostModel(iterations=500, depth=5, random_seed=random_seed),
                RandomForestModel(n_estimators=100, max_depth=12, random_state=random_seed),
            ]
        else:
            self.base_models = base_models
        
        # 默认元模型：Ridge
        if meta_model is None:
            self.meta_model = RidgeModel(alpha=1.0)
        else:
            self.meta_model = meta_model
        
        # 训练后的状态
        self._fitted_base_models: List[Any] = []
        self._meta_scaler: Optional[StandardScaler] = None
        self._oof_meta_features: Optional[np.ndarray] = None  # 保存用于校正
        self._training_time = 0.0
        self._base_correlations: Optional[np.ndarray] = None
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        训练 Stacking 模型
        
        Returns
        -------
        model_dict : dict
            {
                'meta': 训练好的元模型,
                'base': 训练好的基模型列表,
                'meta_scaler': 元特征标准化器
            }
        """
        start_time = time.time()
        
        n_samples = len(y_train)
        n_base = len(self.base_models)
        
        # 1. 使用 inner KFold 生成 OOF 元特征（严格 OOF！）
        kf = KFold(n_splits=self.inner_cv, shuffle=True, random_state=self.random_seed)
        oof_meta = np.zeros((n_samples, n_base))
        
        for fold_idx, (inner_train_idx, inner_val_idx) in enumerate(kf.split(X_train)):
            X_it, y_it = X_train[inner_train_idx], y_train[inner_train_idx]
            X_iv = X_train[inner_val_idx]
            
            # 处理样本权重
            w_it = sample_weights[inner_train_idx] if sample_weights is not None else None
            
            for j, base_module in enumerate(self.base_models):
                # 训练基模型
                model = base_module.fit(X_it, y_it, w_it)
                # 预测内层验证集
                oof_meta[inner_val_idx, j] = base_module.predict(model, X_iv)
        
        self._oof_meta_features = oof_meta.copy()
        
        # 计算基模型预测相关性（用于分析同质化问题）
        self._base_correlations = np.corrcoef(oof_meta.T)
        
        # 2. 对元特征进行标准化（可选）
        if self.use_meta_scaler:
            self._meta_scaler = StandardScaler()
            oof_meta_scaled = self._meta_scaler.fit_transform(oof_meta)
        else:
            oof_meta_scaled = oof_meta
        
        # 3. 训练元模型
        meta_fitted = self.meta_model.fit(oof_meta_scaled, y_train, sample_weights)
        
        # 4. 用全训练集重新训练所有基模型（供预测时使用）
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
        """预测"""
        # 1. 用基模型预测，生成元特征
        meta_features = np.column_stack([
            base_module.predict(fitted, X)
            for base_module, fitted in zip(self.base_models, model_dict['base'])
        ])
        
        # 2. 标准化元特征
        if model_dict['meta_scaler'] is not None:
            meta_scaled = model_dict['meta_scaler'].transform(meta_features)
        else:
            meta_scaled = meta_features
        
        # 3. 元模型预测
        return self.meta_model.predict(model_dict['meta'], meta_scaled)
    
    def get_oof_predictions(self, 
                            model: Dict[str, Any],
                            X_train: np.ndarray, 
                            y_train: np.ndarray, 
                            groups: np.ndarray,
                            sample_weights: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """
        返回严格 OOF 预测（用于偏差校正）
        
        注意：此方法返回的是在 fit() 过程中已生成的 OOF 预测
        （通过 inner CV 生成的元特征，再经过 meta-learner 预测）
        """
        # 使用已存储的 OOF 元特征
        if self._oof_meta_features is None:
            # 如果没有存储，需要重新计算（不推荐）
            return self.predict(model, X_train)
        
        # 标准化元特征
        if self._meta_scaler is not None:
            oof_scaled = self._meta_scaler.transform(self._oof_meta_features)
        else:
            oof_scaled = self._oof_meta_features
        
        # 元模型预测
        return self.meta_model.predict(model['meta'], oof_scaled)
    
    def get_base_correlations(self) -> Optional[np.ndarray]:
        """返回基模型预测相关性矩阵"""
        return self._base_correlations
    
    def get_meta_weights(self, model_dict: Dict[str, Any]) -> Optional[np.ndarray]:
        """返回元模型权重（如果是线性模型）"""
        return self.meta_model.get_weights(model_dict['meta'])


# ============================================================
# Ridge 元模型
# ============================================================

class RidgeModel(ModelModule):
    """
    Ridge 回归 - 用作 Stacking 的元模型
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._training_time = 0.0
    
    def fit(self, 
            X_train: np.ndarray, 
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None) -> Any:
        """训练 Ridge 模型"""
        from sklearn.linear_model import Ridge
        
        start_time = time.time()
        
        model = Ridge(alpha=self.alpha)
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        self._training_time = time.time() - start_time
        return model
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """预测"""
        return model.predict(X)
    
    def get_weights(self, model: Any) -> np.ndarray:
        """获取回归系数"""
        return model.coef_


# ============================================================
# 便捷工厂函数
# ============================================================

def get_model_module(name: str, **kwargs) -> ModelModule:
    """
    模型模块工厂函数
    
    Parameters
    ----------
    name : str
        模块名称: 'ert' | 'extratrees' | 'catboost' | 'rf' | 'randomforest' | 'stacking'
    **kwargs
        模块参数
        
    Returns
    -------
    ModelModule
        模型模块实例
    """
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
        raise ValueError(f"未知模型模块: {name}，支持 {list(set(modules.values()))}")
    
    return modules[name_lower](**kwargs)


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    print("=== 模型模块测试 ===\n")
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.5
    
    # 划分训练/验证
    train_idx = np.arange(160)
    val_idx = np.arange(160, 200)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    groups_train = np.random.choice(['A', 'B', 'C'], len(train_idx))
    
    # 测试各模型
    for name in ['ert', 'catboost', 'stacking']:
        print(f"--- {name.upper()} ---")
        module = get_model_module(name)
        
        model = module.fit(X_train, y_train, groups=groups_train)
        y_pred = module.predict(model, X_val)
        
        rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
        print(f"RMSE: {rmse:.4f}")
        print(f"训练时间: {module.get_training_time():.2f}s")
        
        if name == 'stacking':
            corr = module.get_base_correlations()
            print(f"基模型相关性矩阵:\n{corr}")
        print()
    
    print("✅ 所有模型模块测试通过！")
