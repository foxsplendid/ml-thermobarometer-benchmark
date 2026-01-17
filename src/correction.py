# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 偏差校正模块
Correction Module: BiasCorrector, LinearBiasCorrector, IdentityCorrector

核心约束（fold-safe）：
1. 校正器只能用训练折的 OOF 预测拟合
2. 禁止使用包含验证折的全局 OOF 来拟合后再校正验证折
3. 每个 outer fold 需要创建新的校正器实例
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression


# ============================================================
# 抽象基类
# ============================================================

class BiasCorrector(ABC):
    """
    偏差校正器抽象基类
    
    接口：
    - fit_on_oof(y_true, y_pred_oof): 基于 OOF 预测拟合校正器
    - transform(y_pred): 应用校正
    """
    
    @abstractmethod
    def fit_on_oof(self, y_true: np.ndarray, y_pred_oof: np.ndarray) -> 'BiasCorrector':
        """
        基于 OOF 预测拟合校正器
        
        Parameters
        ----------
        y_true : np.ndarray
            训练折的真实值
        y_pred_oof : np.ndarray
            训练折的 OOF 预测值（由 inner CV 生成）
        
        注意：必须确保 y_pred_oof 是通过 inner CV 生成的 OOF 预测，
              禁止使用 in-sample 训练预测！
        """
        pass
    
    @abstractmethod
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """
        应用校正
        
        Parameters
        ----------
        y_pred : np.ndarray
            待校正的预测值（通常是验证折的预测）
        
        Returns
        -------
        y_corrected : np.ndarray
            校正后的预测值
        """
        pass
    
    def fit_transform(self, y_true: np.ndarray, y_pred_oof: np.ndarray) -> np.ndarray:
        """拟合并返回校正后的 OOF 预测"""
        self.fit_on_oof(y_true, y_pred_oof)
        return self.transform(y_pred_oof)


# ============================================================
# 线性偏差校正器
# ============================================================

class LinearBiasCorrector(BiasCorrector):
    """
    线性偏差校正器
    
    模型: y_corrected = a * y_pred + b
    
    通过线性回归拟合 y_true ~ y_pred_oof，得到系数 a 和截距 b
    """
    
    def __init__(self):
        self._coef: Optional[float] = None      # 斜率 a
        self._intercept: Optional[float] = None  # 截距 b
        self._is_fitted = False
    
    def fit_on_oof(self, y_true: np.ndarray, y_pred_oof: np.ndarray) -> 'LinearBiasCorrector':
        """
        基于 OOF 预测拟合线性校正器
        
        使用线性回归: y_true = a * y_pred_oof + b
        """
        y_true = np.asarray(y_true).ravel()
        y_pred_oof = np.asarray(y_pred_oof).ravel()
        
        if len(y_true) != len(y_pred_oof):
            raise ValueError(f"y_true 和 y_pred_oof 长度不匹配: {len(y_true)} vs {len(y_pred_oof)}")
        
        # 使用 sklearn 的 LinearRegression
        lr = LinearRegression()
        lr.fit(y_pred_oof.reshape(-1, 1), y_true)
        
        self._coef = lr.coef_[0]
        self._intercept = lr.intercept_
        self._is_fitted = True
        
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """应用线性校正"""
        if not self._is_fitted:
            raise RuntimeError("校正器未拟合，请先调用 fit_on_oof()")
        
        y_pred = np.asarray(y_pred).ravel()
        return self._coef * y_pred + self._intercept
    
    @property
    def coef(self) -> float:
        """获取斜率系数"""
        if not self._is_fitted:
            raise RuntimeError("校正器未拟合")
        return self._coef
    
    @property
    def intercept(self) -> float:
        """获取截距"""
        if not self._is_fitted:
            raise RuntimeError("校正器未拟合")
        return self._intercept
    
    def get_params(self) -> dict:
        """获取拟合后的参数"""
        if not self._is_fitted:
            return {'coef': None, 'intercept': None, 'is_fitted': False}
        return {
            'coef': self._coef,
            'intercept': self._intercept,
            'is_fitted': True
        }


# ============================================================
# 恒等校正器（无校正）
# ============================================================

class IdentityCorrector(BiasCorrector):
    """
    恒等校正器（不进行任何校正）
    
    用于 correct=False 的实验配置
    """
    
    def fit_on_oof(self, y_true: np.ndarray, y_pred_oof: np.ndarray) -> 'IdentityCorrector':
        """不进行拟合"""
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """直接返回原值"""
        return np.asarray(y_pred).ravel().copy()


# ============================================================
# 多项式偏差校正器（可选扩展）
# ============================================================

class PolynomialBiasCorrector(BiasCorrector):
    """
    多项式偏差校正器
    
    模型: y_corrected = a_n * y_pred^n + ... + a_1 * y_pred + a_0
    
    Parameters
    ----------
    degree : int, default=2
        多项式阶数
    """
    
    def __init__(self, degree: int = 2):
        self.degree = degree
        self._coeffs: Optional[np.ndarray] = None  # 多项式系数
        self._is_fitted = False
    
    def fit_on_oof(self, y_true: np.ndarray, y_pred_oof: np.ndarray) -> 'PolynomialBiasCorrector':
        """拟合多项式校正器"""
        y_true = np.asarray(y_true).ravel()
        y_pred_oof = np.asarray(y_pred_oof).ravel()
        
        # 使用 numpy 的 polyfit
        self._coeffs = np.polyfit(y_pred_oof, y_true, self.degree)
        self._is_fitted = True
        
        return self
    
    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """应用多项式校正"""
        if not self._is_fitted:
            raise RuntimeError("校正器未拟合，请先调用 fit_on_oof()")
        
        y_pred = np.asarray(y_pred).ravel()
        return np.polyval(self._coeffs, y_pred)


# ============================================================
# 工厂函数
# ============================================================

def get_corrector(name: str = 'linear', **kwargs) -> BiasCorrector:
    """
    偏差校正器工厂函数
    
    Parameters
    ----------
    name : str, default='linear'
        校正器类型：'linear', 'identity', 'polynomial'
    **kwargs : dict
        校正器参数
    
    Returns
    -------
    BiasCorrector
        校正器实例
    """
    name = name.lower().strip()
    
    if name == 'linear':
        return LinearBiasCorrector()
    elif name == 'identity' or name == 'none':
        return IdentityCorrector()
    elif name == 'polynomial' or name == 'poly':
        degree = kwargs.get('degree', 2)
        return PolynomialBiasCorrector(degree=degree)
    else:
        raise ValueError(f"未知校正器类型: {name}，支持 'linear', 'identity', 'polynomial'")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    # 示例：线性校正
    print("=== 线性偏差校正示例 ===")
    
    # 模拟数据：预测值存在系统性偏差
    np.random.seed(42)
    y_true = np.random.uniform(800, 1200, 100)
    y_pred_oof = y_true * 0.95 + 20 + np.random.normal(0, 10, 100)  # 有偏预测
    
    # 拟合校正器
    corrector = LinearBiasCorrector()
    corrector.fit_on_oof(y_true, y_pred_oof)
    
    print(f"拟合参数: coef={corrector.coef:.4f}, intercept={corrector.intercept:.4f}")
    
    # 应用校正
    y_corrected = corrector.transform(y_pred_oof)
    
    # 计算校正效果
    rmse_before = np.sqrt(np.mean((y_true - y_pred_oof) ** 2))
    rmse_after = np.sqrt(np.mean((y_true - y_corrected) ** 2))
    
    print(f"校正前 RMSE: {rmse_before:.2f}")
    print(f"校正后 RMSE: {rmse_after:.2f}")
