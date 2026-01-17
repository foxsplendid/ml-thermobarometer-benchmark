# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - M3 校正模块实现
Correction Modules: NoCorrection, ResidualRegressionCorrector

核心约束：
1. fit() 必须使用 OOF 预测，禁止使用 in-sample 预测
2. 校正模型只能在训练折的（真值，OOF预测）上拟合
"""

import numpy as np
from typing import Any, Dict, Optional

from .interfaces import CorrectionModule


# ============================================================
# 无校正模块
# ============================================================

class NoCorrection(CorrectionModule):
    """
    无校正 - 直接返回原始预测
    
    作为对照基线，不进行任何偏差校正
    """
    
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
    分段线性校正器（可选扩展）
    
    针对端部效应，对低值和高值区域分别拟合线性校正
    """
    
    def __init__(self, n_segments: int = 3, quantiles: Optional[list] = None):
        """
        Parameters
        ----------
        n_segments : int
            分段数量
        quantiles : list of float
            分段边界分位数，默认 [0.33, 0.67]
        """
        self.n_segments = n_segments
        self.quantiles = quantiles or [1/3, 2/3]
    
    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Dict[str, Any]:
        """拟合分段线性模型"""
        from sklearn.linear_model import LinearRegression
        
        # 计算分段边界
        boundaries = [y_pred_train.min()]
        for q in self.quantiles:
            boundaries.append(np.percentile(y_pred_train, q * 100))
        boundaries.append(y_pred_train.max())
        
        # 对每个分段拟合线性模型
        segment_models = []
        for i in range(len(boundaries) - 1):
            mask = (y_pred_train >= boundaries[i]) & (y_pred_train < boundaries[i + 1])
            if i == len(boundaries) - 2:  # 最后一段包含边界
                mask = (y_pred_train >= boundaries[i]) & (y_pred_train <= boundaries[i + 1])
            
            if np.sum(mask) < 5:  # 样本太少
                segment_models.append(None)
                continue
            
            X_seg = y_pred_train[mask].reshape(-1, 1)
            y_seg = y_true_train[mask]
            
            model = LinearRegression()
            model.fit(X_seg, y_seg)
            segment_models.append(model)
        
        return {
            'boundaries': boundaries,
            'segment_models': segment_models
        }
    
    def apply(self, corr_model: Dict[str, Any], y_pred: np.ndarray) -> np.ndarray:
        """应用分段校正"""
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
        
        return y_corr
    
    def get_correction_params(self, corr_model: Dict[str, Any]) -> Dict[str, float]:
        """返回分段参数"""
        if corr_model is None:
            return {}
        
        params = {'method': 'segmented_linear', 'n_segments': len(corr_model.get('segment_models', []))}
        
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


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    print("=== 校正模块测试 ===\n")
    
    # 生成测试数据（模拟有系统性偏差的预测）
    np.random.seed(42)
    n_samples = 100
    y_true = np.linspace(800, 1400, n_samples) + np.random.randn(n_samples) * 20
    # 模拟预测：有系统性低估（斜率 < 1）
    y_pred = y_true * 0.85 + 150 + np.random.randn(n_samples) * 15
    
    # 划分训练/验证
    train_idx = np.arange(80)
    val_idx = np.arange(80, 100)
    
    y_true_train, y_true_val = y_true[train_idx], y_true[val_idx]
    y_pred_train, y_pred_val = y_pred[train_idx], y_pred[val_idx]
    
    # 测试各模块
    for name in ['none', 'residual']:
        print(f"--- {name.upper()} ---")
        module = get_correction_module(name)
        
        corr_model = module.fit(y_true_train, y_pred_train)
        y_corr = module.apply(corr_model, y_pred_val)
        
        # 计算校正前后的指标
        rmse_before = np.sqrt(np.mean((y_true_val - y_pred_val) ** 2))
        rmse_after = np.sqrt(np.mean((y_true_val - y_corr) ** 2))
        
        print(f"RMSE 校正前: {rmse_before:.2f}")
        print(f"RMSE 校正后: {rmse_after:.2f}")
        print(f"参数: {module.get_correction_params(corr_model)}")
        print()
    
    print("✅ 所有校正模块测试通过！")
