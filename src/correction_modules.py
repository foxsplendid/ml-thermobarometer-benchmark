# -*- coding: utf-8 -*-
"""
M3 校正模块 - NoCorrection, ResidualRegressionCorrector, SegmentedLinearCorrector

【地质学ML传统实践说明】
在地质学领域的机器学习实践中，简单模型（ERT、CatBoost、RandomForest）的
偏差校正器通常使用in-sample预测进行训练。这种做法风险较低但理论上可能
导致轻微过拟合。StrictOOFStacking使用严格的内层CV生成OOF预测，避免此问题。

详见 interfaces.py 中 ModelModule.get_oof_predictions() 的说明。
"""

import numpy as np
from typing import Any, Dict, Optional

from .interfaces import CorrectionModule

# ============================================================
# 无校正模块
# ============================================================

class NoCorrection(CorrectionModule):
    """无校正 - 直接返回原始预测"""

    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Any:
        """不做任何操作"""
        return None
    
    def apply(self, corr_model: Any, y_pred: np.ndarray) -> np.ndarray:
        """直接返回原始预测"""
        return y_pred.copy()
    
    def get_correction_params(self, corr_model: Any) -> Dict[str, float]:
        """无参数"""
        return {'method': 'none'}

# ============================================================
# 残差回归校正模块
# ============================================================

class ResidualRegressionCorrector(CorrectionModule):
    """
    残差回归校正器
    
    策略（两步法）：
    1. 拟合残差模型：residual = g(y_pred)
       - residual = y_true - y_pred
       - 用 Ridge/GBDT 学习 y_pred 到 residual 的映射
    
    2. 应用校正：y_corr = y_pred + g(y_pred)
    
    这种方法可以校正预测的系统性偏差，尤其是在极值区域
    
    注意：必须使用 OOF 预测进行拟合，否则会过拟合！
    """
    
    def __init__(self, 
                 method: str = 'ridge',
                 alpha: float = 1.0,
                 use_polynomial: bool = False,
                 poly_degree: int = 2):
        """
        Parameters
        ----------
        method : str
            残差模型类型: 'ridge' | 'linear'
        alpha : float
            Ridge 正则化参数
        use_polynomial : bool
            是否使用多项式特征
        poly_degree : int
            多项式次数
        """
        self.method = method
        self.alpha = alpha
        self.use_polynomial = use_polynomial
        self.poly_degree = poly_degree
    
    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Dict[str, Any]:
        """
        拟合残差回归模型
        
        Returns
        -------
        corr_model : dict
            {
                'residual_model': 残差回归模型,
                'poly_features': 多项式特征转换器（如果使用）,
                'slope': 诊断用斜率,
                'intercept': 诊断用截距
            }
        """
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        
        # 计算残差：正值表示模型低估
        residuals = y_true_train - y_pred_train
        
        # 准备特征
        X_pred = y_pred_train.reshape(-1, 1)
        
        # 多项式特征（可选）
        poly_features = None
        if self.use_polynomial:
            poly_features = PolynomialFeatures(degree=self.poly_degree, include_bias=False)
            X_features = poly_features.fit_transform(X_pred)
        else:
            X_features = X_pred
        
        # 拟合残差模型
        if self.method == 'ridge':
            residual_model = Ridge(alpha=self.alpha)
        else:
            residual_model = LinearRegression()
        
        residual_model.fit(X_features, residuals)
        
        # 计算诊断指标（原始预测 vs 真值的回归）
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
        """应用校正"""
        X_pred = y_pred.reshape(-1, 1)
        
        # 多项式特征
        if corr_model['poly_features'] is not None:
            X_features = corr_model['poly_features'].transform(X_pred)
        else:
            X_features = X_pred
        
        # 预测残差
        predicted_residual = corr_model['residual_model'].predict(X_features)
        
        # 校正：y_corr = y_pred + g(y_pred)
        return y_pred + predicted_residual
    
    def get_correction_params(self, corr_model: Dict[str, Any]) -> Dict[str, float]:
        """返回校正参数"""
        if corr_model is None:
            return {}
        
        result = {
            'method': corr_model.get('method', 'unknown'),
            'slope_before': corr_model.get('slope', np.nan),
            'intercept_before': corr_model.get('intercept', np.nan),
        }
        
        # 线性模型系数
        model = corr_model.get('residual_model')
        if model is not None and hasattr(model, 'coef_'):
            result['residual_coef'] = float(model.coef_[0]) if len(model.coef_) == 1 else 0.0
            result['residual_intercept'] = float(model.intercept_)
        
        return result

# ============================================================
# 端部分段线性校正（可选扩展）
# ============================================================

class SegmentedLinearCorrector(CorrectionModule):
    """
    分段线性校正器

    策略：
    1. 根据预测值分位数将数据划分为多个区段
    2. 每个区段独立拟合线性回归（y_true ~ y_pred）
    3. 应用时根据预测值所在区段选择对应模型
    4. 可选：将校正结果裁剪到训练集目标值范围内
    """

    def __init__(self,
                 n_segments: int = 3,
                 quantiles: Optional[list] = None,
                 clip_to_train_range: bool = True):
        """
        Parameters
        ----------
        n_segments : int
            分段数量
        quantiles : list, optional
            分段分位数边界，默认 [1/3, 2/3]（三段）
        clip_to_train_range : bool
            是否将校正结果裁剪到训练集目标值范围
        """
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
            'clip_min': corr_model.get('y_min'),
            'clip_max': corr_model.get('y_max'),
        }

        for i, model in enumerate(corr_model.get('segment_models', [])):
            if model is not None and hasattr(model, 'coef_'):
                params[f'segment_{i}_slope'] = float(model.coef_[0])
                params[f'segment_{i}_intercept'] = float(model.intercept_)

        return params

# ============================================================
# 便捷工厂函数
# ============================================================

def get_correction_module(name: str, **kwargs) -> CorrectionModule:
    """
    校正模块工厂函数
    
    Parameters
    ----------
    name : str
        模块名称: 'none' | 'residual' | 'segmented'
    **kwargs
        模块参数
        
    Returns
    -------
    CorrectionModule
        校正模块实例
    """
    modules = {
        'none': NoCorrection,
        'residual': ResidualRegressionCorrector,
        'segmented': SegmentedLinearCorrector,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"未知校正模块: {name}，支持 {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

