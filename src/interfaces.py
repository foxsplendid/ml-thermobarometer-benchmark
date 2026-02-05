# -*- coding: utf-8 -*-
"""
统一接口定义 - DataModule, ModelModule, CorrectionModule, UncertaintyModule

本模块定义了 M1-M4 各模块的抽象基类，确保所有实现遵循统一契约。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import numpy as np

# 避免循环导入
if TYPE_CHECKING:
    from .protocol import Pipeline


# ============================================================
# 数据模块状态
# ============================================================

@dataclass
class DataModuleState:
    """数据模块拟合状态，用于验证折应用相同变换"""
    scaler: Any = None                     # StandardScaler 实例
    bin_edges: Optional[np.ndarray] = None # 分箱边界
    feature_std: Optional[np.ndarray] = None # 原始特征标准差（用于MC增强）
    feature_names: Optional[List[str]] = None  # 特征列名（用于绘图/MC）
    extra: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# M1 数据模块接口
# ============================================================

class DataModule(ABC):
    """
    M1 数据模块抽象基类
    
    职责：
    - 数据标准化（Z-score normalization）
    - 分布处理（平衡/增强）
    - 输出样本权重

    约束：
    - fit_transform() 仅在训练折调用
    - transform() 仅在验证/测试折调用
    - 所有拟合参数通过 DataModuleState 传递

    数据契约：
    - 输入 X: shape (n_samples, n_features), dtype=float64
      - 单位: wt%（氧化物质量百分比）
      - 范围: 通常 0-100，但可能有负值（测量误差）
    - 输入 y: shape (n_samples,)
      - 温度 T: 单位 °C，范围约 700-1500
      - 压力 P: 单位 kbar，范围约 0-25
    - 输出 X: 标准化后，均值≈0，标准差≈1
    - 输出 sample_weights: 非负，总和≈n_samples
    """

    def _infer_feature_names(self, n_features: int, feature_names: Optional[List[str]] = None) -> List[str]:
        """
        根据特征数推断特征名列表（基类通用方法）

        Parameters
        ----------
        n_features : int
            特征数量
        feature_names : List[str], optional
            已指定的特征名列表，优先使用

        Returns
        -------
        List[str]
            特征名列表
        """
        if feature_names is not None:
            return feature_names
        # 根据特征数推断特征集类型
        if n_features == 18:
            from config import DataConfig
            return DataConfig().feature_sets['Liquid']
        elif n_features == 9:
            from config import DataConfig
            return DataConfig().feature_sets['NoLiquid']
        raise ValueError(f"无法根据特征数推断特征名，n_features={n_features}，必须显式传入 feature_names")

    @abstractmethod
    def fit_transform(self, 
                      X_train: np.ndarray, 
                      y_train: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, DataModuleState]:
        """
        在训练折上拟合并转换
        
        Parameters
        ----------
        X_train : np.ndarray, shape (n_train, n_features)
            训练集特征
        y_train : np.ndarray, shape (n_train,)
            训练集目标
            
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

    防泄露约束：
    - fit() 只使用训练集数据
    - 校正器拟合在协议层完成，使用全局 OOF 预测
    """
    
    @abstractmethod
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weights: Optional[np.ndarray] = None,
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
        stratify_labels : np.ndarray, optional
            分层标签（Stacking 内层 CV 使用）
            
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


