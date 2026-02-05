# -*- coding: utf-8 -*-
"""
M4 不确定性模块 - MCUncertaintyEstimator

功能：蒙特卡洛输入扰动、预测分布、校准指标

设计说明：
    EPMA 误差模型采用按氧化物列名映射的相对误差：
    - 主量元素（SiO2, Al2O3, FeO, MgO, CaO）：3% 相对误差
    - 低含量元素（TiO2, MnO, Na2O, Cr2O3, K2O）：8% 相对误差

    设计依据：Ágreda-López et al. (2024) ML_PT_Pyworkflow

    扰动策略：
    - 按列名固定映射相对误差（而非按数值阈值）
    - 不做负值截断（clip），保留完整的正态分布扰动
    - 不做闭合约束（closure），与训练数据预处理保持一致
    - 与数据增强模块共用扰动函数，确保一致性
"""
import numpy as np
from typing import Any, Dict, List, Optional
from .interfaces import UncertaintyModule

# ============================================================
# 蒙特卡洛不确定性估计器
# ============================================================
class MCUncertaintyEstimator(UncertaintyModule):
    """
    MC输入扰动不确定性估计

    EPMA模式：按氧化物列名映射相对误差
    - 主量元素：3% (SiO2, Al2O3, FeO, MgO, CaO)
    - 低含量元素：8% (TiO2, MnO, Na2O, Cr2O3, K2O)

    注意：
    - 不做负值截断，允许扰动后出现负值以保持分布完整性
    - 与 AugmentedDataModule 共用扰动函数
    """

    def __init__(self,
                 n_mc: int = 1000,
                 feature_names: Optional[List[str]] = None,
                 percentiles: tuple = (5, 16, 50, 84, 95),
                 random_seed: int = 42):
        """
        Parameters
        ----------
        n_mc : int
            MC 采样次数
        feature_names : List[str], optional
            特征列名列表，用于按列名映射相对误差
        percentiles : tuple
            要计算的分位数
        random_seed : int
            随机种子
        """
        self.n_mc = n_mc
        self.feature_names = feature_names
        self.percentiles = percentiles
        self.random_seed = random_seed

    def predict_distribution(self,
                             pipeline: Any,
                             X: np.ndarray,
                             mc_params: Optional[Dict[str, Any]] = None,
                             fold_idx: int = 0
                             ) -> Dict[str, np.ndarray]:
        """
        对输入 X 进行 MC 扰动，返回预测分布统计量

        Parameters
        ----------
        pipeline : Pipeline
            已拟合的模型管道
        X : np.ndarray
            输入特征矩阵 (n_samples, n_features)
        mc_params : dict, optional
            可选参数覆盖，支持:
            - n_mc: MC 采样次数
            - feature_names: 特征列名列表
            - percentiles: 分位数列表
            - seed_offset: 额外的种子偏移量
        fold_idx : int
            折索引，用于派生不同的随机种子，确保各折/重复使用不同随机序列

        Returns
        -------
        dict
            包含 samples, mean, std, median, p5/p16/p84/p95 等
        """
        from .perturbation import get_rel_err_vector, epma_perturb

        # 使用 base_seed + fold_idx 派生种子，确保各折使用不同随机序列
        seed_offset = mc_params.get('seed_offset', 0) if mc_params else 0
        effective_seed = self.random_seed + fold_idx + seed_offset
        rng = np.random.RandomState(effective_seed)

        n_mc = mc_params.get('n_mc', self.n_mc) if mc_params else self.n_mc
        percentiles = mc_params.get('percentiles', self.percentiles) if mc_params else self.percentiles
        feature_names = mc_params.get('feature_names', self.feature_names) if mc_params else self.feature_names

        # 未显式传入时按特征数推断（9/18），未知则报错
        if feature_names is None:
            if X.shape[1] == 18:
                from config import DataConfig
                feature_names = DataConfig().feature_sets['Liquid']
            elif X.shape[1] == 9:
                from config import DataConfig
                feature_names = DataConfig().feature_sets['NoLiquid']
            else:
                raise ValueError(f"无法根据特征数推断 feature_names，n_features={X.shape[1]}")

        if len(feature_names) != X.shape[1]:
            raise ValueError("feature_names 长度必须与 X 的特征维度一致")
        rel_err_vec = get_rel_err_vector(feature_names, strict=True)

        n_samples = X.shape[0]
        predictions = np.zeros((n_mc, n_samples))

        # MC 扰动预测
        for i in range(n_mc):
            X_perturbed = epma_perturb(X, rel_err_vec, rng)
            predictions[i] = pipeline.predict_raw(X_perturbed)

        # 计算分位数
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


# ============================================================
# 便捷工厂函数
# ============================================================
def get_uncertainty_module(name: str, **kwargs) -> UncertaintyModule:
    """
    不确定性模块工厂函数
    
    Parameters
    ----------
    name : str
        模块名称: 'mc'
    **kwargs
        模块参数
        
    Returns
    -------
    UncertaintyModule
        不确定性模块实例
    """
    modules = {
        'mc': MCUncertaintyEstimator,
    }
    
    name_lower = name.lower().strip()
    if name_lower not in modules:
        raise ValueError(f"未知不确定性模块: {name}，支持 {list(modules.keys())}")
    
    return modules[name_lower](**kwargs)

