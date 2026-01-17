# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - M4 不确定性模块实现
Uncertainty Modules: MCUncertaintyEstimator

核心功能：
1. 蒙特卡洛输入扰动
2. 预测分布计算（均值、标准差、置信区间）
3. 校准指标计算（PICP、区间宽度、误差-不确定性相关性）
"""

import numpy as np
from typing import Any, Dict, Optional

from .interfaces import UncertaintyModule


# ============================================================
# 蒙特卡洛不确定性估计器
# ============================================================

class MCUncertaintyEstimator(UncertaintyModule):
    """
    蒙特卡洛输入扰动不确定性估计器
    
    策略：
    1. 对每个验证样本的输入特征添加高斯噪声（模拟分析误差）
    2. 重复执行 N 次预测
    3. 计算预测分布的统计量（均值、标准差、分位数）
    
    应用场景：
    - 评估预测对输入不确定性的敏感性
    - 生成预测置信区间
    - 校准模型的不确定性估计
    """
    
    def __init__(self,
                 n_mc: int = 50,
                 noise_level: float = 0.02,
                 percentiles: tuple = (2.5, 50, 97.5),
                 random_seed: int = 42):
        """
        Parameters
        ----------
        n_mc : int
            蒙特卡洛迭代次数
        noise_level : float
            噪声水平（相对于特征标准差的比例）
        percentiles : tuple
            要计算的分位数
        random_seed : int
            随机种子
        """
        self.n_mc = n_mc
        self.noise_level = noise_level
        self.percentiles = percentiles
        self.random_seed = random_seed
    
    def predict_distribution(self,
                             pipeline: Any,  # Pipeline 对象
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, np.ndarray]:
        """
        执行蒙特卡洛不确定性估计
        
        Parameters
        ----------
        pipeline : Pipeline
            完整的预测管道，必须实现 predict_raw(X) 方法
            - predict_raw 接受原始（未标准化）特征，内部处理标准化和校正
        X : np.ndarray
            原始特征（未标准化）
        mc_params : dict, optional
            MC 参数覆盖
            
        Returns
        -------
        dist : dict
            预测分布统计量
        """
        np.random.seed(self.random_seed)
        
        # 参数覆盖
        n_mc = mc_params.get('n_mc', self.n_mc) if mc_params else self.n_mc
        noise_level = mc_params.get('noise_level', self.noise_level) if mc_params else self.noise_level
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_mc, n_samples))
        
        # 计算特征标准差（用于生成相对噪声）
        feature_std = np.std(X, axis=0)
        feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)  # 避免除零
        
        for i in range(n_mc):
            # 添加高斯噪声
            noise = np.random.normal(0, noise_level, X.shape) * feature_std
            X_perturbed = X + noise
            
            # 通过 pipeline 预测
            y_pred = pipeline.predict_raw(X_perturbed)
            predictions[i] = y_pred
        
        # 计算统计量
        return {
            'samples': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0),
            'median': np.percentile(predictions, 50, axis=0),
            'p5': np.percentile(predictions, 5, axis=0),
            'p95': np.percentile(predictions, 95, axis=0),
        }
    
    def compute_calibration_metrics(self,
                                    y_true: np.ndarray,
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        """
        计算校准指标
        
        Parameters
        ----------
        y_true : np.ndarray
            真实值
        dist : dict
            由 predict_distribution 返回的分布字典
            
        Returns
        -------
        metrics : dict
            校准指标：
            - picp_95: 95% 预测区间覆盖概率
            - picp_90: 90% 预测区间覆盖概率
            - mean_interval_width: 平均区间宽度
            - median_interval_width: 中位区间宽度
            - error_uncertainty_corr: |error| 与 width 的相关性
            - sharpness: 区间宽度的标准差（越小越好）
        """
        # 95% 预测区间覆盖概率
        in_95_interval = (y_true >= dist['ci_lower']) & (y_true <= dist['ci_upper'])
        picp_95 = np.mean(in_95_interval)
        
        # 90% 预测区间覆盖概率
        in_90_interval = (y_true >= dist['p5']) & (y_true <= dist['p95'])
        picp_90 = np.mean(in_90_interval)
        
        # 区间宽度
        widths_95 = dist['ci_upper'] - dist['ci_lower']
        mean_width = np.mean(widths_95)
        median_width = np.median(widths_95)
        
        # 误差与不确定性的相关性
        # 理想情况：误差大的地方，不确定性也应该大
        abs_errors = np.abs(y_true - dist['mean'])
        if np.std(abs_errors) > 1e-10 and np.std(widths_95) > 1e-10:
            corr = np.corrcoef(abs_errors, widths_95)[0, 1]
        else:
            corr = np.nan
        
        # Sharpness：区间宽度的变异性
        sharpness = np.std(widths_95)
        
        return {
            'picp_95': picp_95,
            'picp_90': picp_90,
            'mean_interval_width': mean_width,
            'median_interval_width': median_width,
            'error_uncertainty_corr': corr,
            'sharpness': sharpness,
        }
    
    def compute_reliability_diagram_data(self,
                                         y_true: np.ndarray,
                                         dist: Dict[str, np.ndarray],
                                         n_bins: int = 10
                                         ) -> Dict[str, np.ndarray]:
        """
        计算可靠性图数据（用于校准可视化）
        
        Returns
        -------
        data : dict
            - expected_coverage: 期望覆盖率（分位数级别）
            - observed_coverage: 观测覆盖率
        """
        quantiles = np.linspace(5, 95, n_bins)
        observed_coverages = []
        
        for q in quantiles:
            lower = np.percentile(dist['samples'], (100 - q) / 2, axis=0)
            upper = np.percentile(dist['samples'], 100 - (100 - q) / 2, axis=0)
            in_interval = (y_true >= lower) & (y_true <= upper)
            observed_coverages.append(np.mean(in_interval))
        
        return {
            'expected_coverage': quantiles / 100,
            'observed_coverage': np.array(observed_coverages),
        }


# ============================================================
# MC-CV 重复切分（可选扩展）
# ============================================================

class MCCVUncertaintyEstimator(UncertaintyModule):
    """
    MC-CV 不确定性估计器（可选扩展）
    
    通过多次随机重新划分训练/验证集来评估模型不确定性
    这种方法评估的是模型本身的稳定性，而非输入不确定性
    """
    
    def __init__(self, n_repeats: int = 10, random_seed: int = 42):
        self.n_repeats = n_repeats
        self.random_seed = random_seed
    
    def predict_distribution(self,
                             pipeline: Any,
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, np.ndarray]:
        """需要完整的 CV 重新训练，通常在 Protocol 层面实现"""
        raise NotImplementedError(
            "MC-CV 需要在 Protocol 层面实现，因为需要重新训练模型"
        )
    
    def compute_calibration_metrics(self,
                                    y_true: np.ndarray,
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        """与 MCUncertaintyEstimator 相同"""
        # 复用 MCUncertaintyEstimator 的实现
        return MCUncertaintyEstimator().compute_calibration_metrics(y_true, dist)


# ============================================================
# 便捷工厂函数
# ============================================================

def get_uncertainty_module(name: str, **kwargs) -> UncertaintyModule:
    """
    不确定性模块工厂函数
    
    Parameters
    ----------
    name : str
        模块名称: 'mc' | 'mccv'
    **kwargs
        模块参数
        
    Returns
    -------
    UncertaintyModule
        不确定性模块实例
    """
    modules = {
        'mc': MCUncertaintyEstimator,
        'mccv': MCCVUncertaintyEstimator,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"未知不确定性模块: {name}，支持 {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)


# ============================================================
# 模块测试
# ============================================================

if __name__ == "__main__":
    print("=== 不确定性模块测试 ===\n")
    
    # 创建一个简单的 mock pipeline
    class MockPipeline:
        def __init__(self, noise_scale=0.1):
            self.noise_scale = noise_scale
        
        def predict_raw(self, X):
            # 简单的线性预测 + 噪声
            return np.sum(X, axis=1) + np.random.randn(len(X)) * self.noise_scale
    
    # 生成测试数据
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 10)
    y_true = np.sum(X, axis=1) + np.random.randn(n_samples) * 0.5
    
    # 测试 MC 估计器
    pipeline = MockPipeline(noise_scale=0.1)
    mc_estimator = MCUncertaintyEstimator(n_mc=30, noise_level=0.05)
    
    print("--- MCUncertaintyEstimator ---")
    dist = mc_estimator.predict_distribution(pipeline, X)
    
    print(f"样本数: {dist['samples'].shape}")
    print(f"均值形状: {dist['mean'].shape}")
    print(f"95%CI 宽度均值: {np.mean(dist['ci_upper'] - dist['ci_lower']):.4f}")
    
    # 计算校准指标
    metrics = mc_estimator.compute_calibration_metrics(y_true, dist)
    print(f"\n校准指标:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✅ 不确定性模块测试通过！")
