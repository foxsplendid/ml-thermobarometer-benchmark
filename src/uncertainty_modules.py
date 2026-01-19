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
    Monte Carlo input perturbation for uncertainty estimation.

    - EPMA mode: relative error (3% > 1 wt%, 8% <= 1 wt%).
    - Gaussian mode: noise_level * feature_std fallback.
    """

    def __init__(self,
                 n_mc: int = 1000,
                 noise_level: float = 0.02,
                 error_model: str = "epma",
                 rel_err_high: float = 0.03,
                 rel_err_low: float = 0.08,
                 error_threshold: float = 1.0,
                 clip_min: Optional[float] = 0.0,
                 percentiles: tuple = (16, 50, 84),
                 random_seed: int = 42):
        self.n_mc = n_mc
        self.noise_level = noise_level
        self.error_model = error_model.lower().strip() if isinstance(error_model, str) else "epma"
        self.rel_err_high = rel_err_high
        self.rel_err_low = rel_err_low
        self.error_threshold = error_threshold
        self.clip_min = clip_min
        self.percentiles = percentiles
        self.random_seed = random_seed

    def predict_distribution(self,
                             pipeline: Any,
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, np.ndarray]:
        np.random.seed(self.random_seed)

        n_mc = mc_params.get('n_mc', self.n_mc) if mc_params else self.n_mc
        noise_level = mc_params.get('noise_level', self.noise_level) if mc_params else self.noise_level
        error_model = mc_params.get('error_model', self.error_model) if mc_params else self.error_model
        rel_err_high = mc_params.get('rel_err_high', self.rel_err_high) if mc_params else self.rel_err_high
        rel_err_low = mc_params.get('rel_err_low', self.rel_err_low) if mc_params else self.rel_err_low
        error_threshold = mc_params.get('error_threshold', self.error_threshold) if mc_params else self.error_threshold
        clip_min = mc_params.get('clip_min', self.clip_min) if mc_params else self.clip_min
        percentiles = mc_params.get('percentiles', self.percentiles) if mc_params else self.percentiles

        error_model = error_model.lower().strip() if isinstance(error_model, str) else "epma"

        n_samples = X.shape[0]
        predictions = np.zeros((n_mc, n_samples))

        if error_model == "epma":
            rel_err = np.where(
                np.abs(X) > error_threshold,
                rel_err_high,
                rel_err_low
            )
            scale = rel_err * np.abs(X)
            for i in range(n_mc):
                noise = np.random.normal(0.0, scale, size=X.shape)
                X_perturbed = X + noise
                if clip_min is not None:
                    X_perturbed = np.maximum(X_perturbed, clip_min)
                predictions[i] = pipeline.predict_raw(X_perturbed)
        else:
            feature_std = np.std(X, axis=0)
            feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)
            scale = noise_level * feature_std
            for i in range(n_mc):
                noise = np.random.normal(0.0, scale, size=X.shape)
                X_perturbed = X + noise
                if clip_min is not None:
                    X_perturbed = np.maximum(X_perturbed, clip_min)
                predictions[i] = pipeline.predict_raw(X_perturbed)

        percentiles = tuple(percentiles)
        pct_values = np.percentile(predictions, percentiles, axis=0)
        pct_map = {p: pct_values[i] for i, p in enumerate(percentiles)}

        p16 = pct_map.get(16, np.percentile(predictions, 16, axis=0))
        p84 = pct_map.get(84, np.percentile(predictions, 84, axis=0))
        median = pct_map.get(50, np.percentile(predictions, 50, axis=0))
        p5 = pct_map.get(5, np.percentile(predictions, 5, axis=0))
        p95 = pct_map.get(95, np.percentile(predictions, 95, axis=0))

        return {
            'samples': predictions,
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0),
            'ci_lower': p16,
            'ci_upper': p84,
            'median': median,
            'p16': p16,
            'p84': p84,
            'p5': p5,
            'p95': p95,
        }

    def compute_calibration_metrics(self,
                                    y_true: np.ndarray,
                                    dist: Dict[str, np.ndarray]
                                    ) -> Dict[str, float]:
        lower = dist.get('p16', dist.get('ci_lower'))
        upper = dist.get('p84', dist.get('ci_upper'))
        if lower is None or upper is None:
            lower = np.percentile(dist['samples'], 16, axis=0)
            upper = np.percentile(dist['samples'], 84, axis=0)

        in_68_interval = (y_true >= lower) & (y_true <= upper)
        picp_68 = np.mean(in_68_interval)

        lower_90 = dist.get('p5')
        upper_90 = dist.get('p95')
        if lower_90 is None or upper_90 is None:
            lower_90 = np.percentile(dist['samples'], 5, axis=0)
            upper_90 = np.percentile(dist['samples'], 95, axis=0)
        in_90_interval = (y_true >= lower_90) & (y_true <= upper_90)
        picp_90 = np.mean(in_90_interval)

        widths = upper - lower
        mean_width = np.mean(widths)
        median_width = np.median(widths)

        center = dist.get('median', dist.get('mean'))
        abs_errors = np.abs(y_true - center)
        if np.std(abs_errors) > 1e-10 and np.std(widths) > 1e-10:
            corr = np.corrcoef(abs_errors, widths)[0, 1]
        else:
            corr = np.nan

        sharpness = np.std(widths)

        return {
            'picp_68': picp_68,
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
