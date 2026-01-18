# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - 统一接口定义
Interfaces: DataModule, ModelModule, CorrectionModule, UncertaintyModule

所有模块必须实现这些抽象接口，确保可插拔与防泄露
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================
# 数据模块状态（用于在验证折应用相同变换）
# ============================================================

@dataclass
class DataModuleState:
    """
    数据模块拟合状态
    
    存储在训练折上拟合的参数，用于在验证折上应用相同变换
    禁止在验证折上重新拟合！
    """
    scaler: Any = None                     # StandardScaler 实例
    bin_edges: Optional[np.ndarray] = None # 分箱边界
    feature_std: Optional[np.ndarray] = None # 原始特征标准差（用于MC增强）
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# M1 数据模块接口
# ============================================================

class DataModule(ABC):
    """
    M1 数据模块抽象基类
    
    职责：
    - 数据标准化
    - 分布处理（分箱重加权、增强等）
    - 输出样本权重
    
    防泄露约束：
    - fit_transform() 只在训练折调用
    - transform() 只在验证折调用，禁止任何拟合操作
    """
    
    @abstractmethod
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray, 
                      groups_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """
        在训练折上拟合并转换
        
        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
            训练集特征
        y_train : np.ndarray, shape (n_train,)
            训练集目标
        groups_train : np.ndarray, shape (n_train,)
            训练集分组标签
            
        Returns
        -------
        X_train2 : np.ndarray
            转换后的特征（可能样本数变化，如增强后）
        y_train2 : np.ndarray
            转换后的目标（与 X_train2 对应）
        sample_weights : np.ndarray
            样本权重（无权重时为全1向量）
        state : DataModuleState
            拟合状态（传递给 transform 使用）
        """
        pass
    
    @abstractmethod
    def transform(self, 
                  X_val: np.ndarray, 
                  state: DataModuleState
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        在验证折上应用变换（禁止任何拟合操作！）
        
        Parameters
        ----------
        X_val : np.ndarray, shape (n_val, n_features)
            验证集特征
        state : DataModuleState
            由 fit_transform 返回的拟合状态
            
        Returns
        -------
        X_val2 : np.ndarray
            转换后的验证集特征
        sample_weights_val : Optional[np.ndarray]
            验证集权重（通常为 None，评估时不需要权重）
        """
        pass
    
    def get_name(self) -> str:
        """返回模块名称（用于日志和结果记录）"""
        return self.__class__.__name__


# ============================================================
# M2 模型模块接口
# ============================================================

class ModelModule(ABC):
    """
    M2 模型模块抽象基类
    
    职责：
    - 模型训练
    - 预测
    - OOF 预测生成（用于偏差校正）
    
    防泄露约束：
    - fit() 只使用训练集数据
    - get_oof_predictions() 必须通过 inner CV 生成，禁止使用 in-sample 预测
    """
    
    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
            groups: Optional[np.ndarray] = None,
            stratify_labels: Optional[np.ndarray] = None) -> Any:
        """
        训练模型
        
        Parameters
        ----------
        X_train : np.ndarray
            训练集特征（已标准化）
        y_train : np.ndarray
            训练集目标
        sample_weights : np.ndarray, optional
            样本权重
        groups : np.ndarray, optional
            分组标签（Stacking 内层 CV 可能需要）
            
        Returns
        -------
        model : Any
            已训练的模型对象（或模型字典，如 Stacking）
        """
        pass
    
    @abstractmethod
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Parameters
        ----------
        model : Any
            由 fit() 返回的模型
        X : np.ndarray
            特征矩阵（已标准化）
            
        Returns
        -------
        y_pred : np.ndarray
            预测值
        """
        pass
    
    def get_oof_predictions(self,
                            model: Any,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            groups: Optional[np.ndarray] = None,
                            sample_weights: Optional[np.ndarray] = None,
                            stratify_labels: Optional[np.ndarray] = None
                            ) -> np.ndarray:
        """
        获取训练集 OOF 预测（用于偏差校正器拟合）

        默认实现：返回 in-sample 预测（非严格 OOF）

        【地质学ML传统做法说明】
        在地质学机器学习领域，传统做法是在外层划分基础上使用内层CV进行模型训练，
        但偏差校正器通常直接使用in-sample预测而非严格OOF预测。这种做法在实践中
        风险较低（对于简单模型如ERT、CatBoost），但理论上可能导致校正器轻微过拟合。

        StrictOOFStacking模型会重写此方法，返回真正的inner OOF预测。

        Parameters
        ----------
        model : Any
            训练好的模型
        X_train : np.ndarray
            训练集特征
        y_train : np.ndarray
            训练集目标
        groups : np.ndarray, optional
            分组标签（已废弃，保留向后兼容）
        sample_weights : np.ndarray, optional
            样本权重
        stratify_labels : np.ndarray, optional
            分层标签（用于内层CV，如有）

        Returns
        -------
        y_oof : np.ndarray
            OOF预测（默认实现返回in-sample预测）
        """
        return self.predict(model, X_train)
    
    def get_name(self) -> str:
        """返回模块名称"""
        return self.__class__.__name__
    
    def get_training_time(self) -> float:
        """返回最近一次训练时间（秒）"""
        return getattr(self, '_training_time', 0.0)


# ============================================================
# M3 校正模块接口
# ============================================================

class CorrectionModule(ABC):
    """
    M3 校正模块抽象基类
    
    职责：
    - 拟合残差回归模型
    - 应用校正
    
    防泄露约束：
    - fit() 必须使用 OOF 预测，禁止使用 in-sample 预测
    - 校正模型只能在训练折的（真值，OOF预测）上拟合
    """
    
    @abstractmethod
    def fit(self, 
            y_true_train: np.ndarray, 
            y_pred_train: np.ndarray) -> Any:
        """
        在训练折上拟合校正模型
        
        Parameters
        ----------
        y_true_train : np.ndarray
            训练折真值
        y_pred_train : np.ndarray
            训练折 OOF 预测（必须是 OOF，不能是 in-sample）
            
        Returns
        -------
        corr_model : Any
            校正模型
        """
        pass
    
    @abstractmethod
    def apply(self, corr_model: Any, y_pred: np.ndarray) -> np.ndarray:
        """
        应用校正
        
        Parameters
        ----------
        corr_model : Any
            由 fit() 返回的校正模型
        y_pred : np.ndarray
            待校正的预测值
            
        Returns
        -------
        y_corrected : np.ndarray
            校正后的预测值
        """
        pass
    
    def get_name(self) -> str:
        """返回模块名称"""
        return self.__class__.__name__
    
    def get_correction_params(self, corr_model: Any) -> Dict[str, float]:
        """返回校正参数（如斜率、截距），用于分析和可视化"""
        return {}


# ============================================================
# M4 不确定性模块接口
# ============================================================

class UncertaintyModule(ABC):
    """
    M4 不确定性模块抽象基类
    
    职责：
    - 执行蒙特卡洛输入扰动
    - 计算预测分布（均值、标准差、置信区间）
    - 计算校准指标（PICP、区间宽度等）
    """
    
    @abstractmethod
    def predict_distribution(self, 
                             pipeline: 'Pipeline',
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, np.ndarray]:
        """
        执行蒙特卡洛不确定性估计
        
        Parameters
        ----------
        pipeline : Pipeline
            完整的预测管道（包含 data_module, model_module, corr_module）
        X : np.ndarray
            原始特征（未标准化）
        mc_params : dict, optional
            MC 参数覆盖
            
        Returns
        -------
        dist : dict
            {
                'samples': np.ndarray,    # (n_mc, n_samples) 所有迭代预测
                'mean': np.ndarray,       # 预测均值
                'std': np.ndarray,        # 预测标准差
                'ci_lower': np.ndarray,   # 2.5% 分位数
                'ci_upper': np.ndarray,   # 97.5% 分位数
                'median': np.ndarray,     # 50% 分位数
            }
        """
        pass
    
    @abstractmethod
    def compute_calibration_metrics(self, 
                                    y_true: np.ndarray, 
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        """
        计算校准指标
        
        Returns
        -------
        metrics : dict
            {
                'picp': float,                  # Prediction Interval Coverage Probability
                'mean_interval_width': float,   # 平均区间宽度
                'error_uncertainty_corr': float # |error| 与 width 的相关性
            }
        """
        pass
    
    def get_name(self) -> str:
        """返回模块名称"""
        return self.__class__.__name__


# ============================================================
# Pipeline 类型定义（前向声明）
# ============================================================

class Pipeline:
    """
    完整预测管道（封装 DataModule + ModelModule + CorrectionModule）
    
    此处仅作类型提示，完整实现在 protocol.py 中
    """
    pass
