# -*- coding: utf-8 -*-
"""
Chapter 3 Benchmark Protocol - 协议执行器
Protocol: Pipeline, GroupCVProtocol, RandomSplitProtocol, ExperimentMatrix

核心功能：
1. Pipeline：封装 DataModule + ModelModule + CorrectionModule
2. GroupCVProtocol：主协议（GroupKFold，10折）
3. RandomSplitProtocol：对照协议（Random Split）
4. ExperimentMatrix：实验矩阵执行器
"""

import os
import time
import yaml
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .interfaces import DataModule, ModelModule, CorrectionModule, UncertaintyModule, DataModuleState


# ============================================================
# Pipeline 类
# ============================================================

class Pipeline:
    """
    完整预测管道（封装 DataModule + ModelModule + CorrectionModule）
    
    职责：
    1. 统一管理数据处理、模型训练、偏差校正的流程
    2. 提供 predict() 和 predict_raw() 接口
    3. 存储训练状态（用于 MC 不确定性估计）
    """
    
    def __init__(self,
                 data_module: DataModule,
                 model_module: ModelModule,
                 corr_module: CorrectionModule):
        self.data_module = data_module
        self.model_module = model_module
        self.corr_module = corr_module
        
        # 训练后的状态
        self._state: Optional[DataModuleState] = None
        self._model: Optional[Any] = None
        self._corr_model: Optional[Any] = None
        self._is_fitted = False
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            groups_train: np.ndarray,
            stratify_labels: Optional[np.ndarray] = None) -> 'Pipeline':
        """
        训练完整管道
        
        流程：
        1. 数据处理（标准化、增强等）
        2. 模型训练
        3. 获取 OOF 预测
        4. 拟合偏差校正器
        """
        # 1. 数据处理
        X2, y2, weights, self._state = self.data_module.fit_transform(
            X_train, y_train, groups_train
        )
        
        # 对于增强后的数据，groups 也需要扩展
        if len(y2) > len(groups_train):
            # 增强模块会扩展样本数
            n_aug = len(y2) // len(groups_train)
            groups2 = np.tile(groups_train, n_aug)
        else:
            groups2 = groups_train

        if stratify_labels is not None:
            if len(y2) > len(stratify_labels):
                n_aug = len(y2) // len(stratify_labels)
                stratify2 = np.tile(stratify_labels, n_aug)
            else:
                stratify2 = stratify_labels
        else:
            stratify2 = None
        
        # 2. 模型训练
        self._model = self.model_module.fit(X2, y2, weights, groups2, stratify_labels=stratify2)
        
        # 3. 获取 OOF 预测（用于偏差校正）
        # 注意：简单模型（ERT/CatBoost）使用in-sample预测（地质学ML传统做法）
        # StrictOOFStacking使用真正的inner CV OOF预测
        y_oof = self.model_module.get_oof_predictions(
            self._model, X2, y2, groups2, weights, stratify_labels=stratify2
        )

        # 4. 拟合偏差校正器
        # 警告：如果y_oof是in-sample预测，校正器可能轻微过拟合
        self._corr_model = self.corr_module.fit(y2, y_oof)
        
        self._is_fitted = True
        return self
    
    def predict(self,
                X: np.ndarray,
                apply_correction: bool = True) -> np.ndarray:
        """
        预测（输入为已标准化的特征）
        
        Parameters
        ----------
        X : np.ndarray
            已标准化的特征
        apply_correction : bool
            是否应用偏差校正
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline 未训练，请先调用 fit()")
        
        y_pred = self.model_module.predict(self._model, X)
        
        if apply_correction:
            y_pred = self.corr_module.apply(self._corr_model, y_pred)
        
        return y_pred
    
    def predict_raw(self,
                    X_raw: np.ndarray,
                    apply_correction: bool = True) -> np.ndarray:
        """
        预测（输入为原始特征，内部处理标准化）
        
        用于 MC 不确定性估计
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline 未训练，请先调用 fit()")
        
        # 应用数据变换（使用训练时的状态）
        X_scaled, _ = self.data_module.transform(X_raw, self._state)
        
        return self.predict(X_scaled, apply_correction)
    
    def get_model(self) -> Any:
        """返回训练好的模型"""
        return self._model
    
    def get_correction_params(self) -> Dict[str, float]:
        """返回校正参数"""
        return self.corr_module.get_correction_params(self._corr_model)
    
    def get_name(self) -> str:
        """返回管道名称"""
        return f"{self.data_module.get_name()}_{self.model_module.get_name()}_{self.corr_module.get_name()}"


# ============================================================
# 指标计算函数
# ============================================================

def compute_all_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        y_pred_raw: Optional[np.ndarray] = None,
                        n_bins: int = 5) -> Dict[str, float]:
    """
    计算完整指标集
    
    Parameters
    ----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        校正后预测值（或唯一预测值）
    y_pred_raw : np.ndarray, optional
        校正前预测值
    n_bins : int
        分箱数量（用于端元诊断）
        
    Returns
    -------
    metrics : dict
        完整指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from scipy.stats import linregress
    
    metrics = {}
    
    # 基础指标
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mbe'] = np.mean(y_pred - y_true)  # Mean Bias Error
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    metrics['r2'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    
    # 回归诊断（y_pred 预测 y_true）
    if len(y_pred) > 2 and np.std(y_pred) > 1e-10:
        reg = linregress(y_pred, y_true)
        metrics['slope'] = reg.slope
        metrics['intercept'] = reg.intercept
    else:
        metrics['slope'] = np.nan
        metrics['intercept'] = np.nan
    
    # 残差标准差
    metrics['resid_std'] = np.std(y_true - y_pred)
    
    # 校正前指标（如果提供）
    if y_pred_raw is not None:
        metrics['rmse_raw'] = np.sqrt(mean_squared_error(y_true, y_pred_raw))
        metrics['mae_raw'] = mean_absolute_error(y_true, y_pred_raw)
        metrics['mbe_raw'] = np.mean(y_pred_raw - y_true)
        
        if len(y_pred_raw) > 2 and np.std(y_pred_raw) > 1e-10:
            reg_raw = linregress(y_pred_raw, y_true)
            metrics['slope_raw'] = reg_raw.slope
            metrics['intercept_raw'] = reg_raw.intercept
    
    # 分箱误差
    if n_bins > 0 and len(y_true) >= n_bins:
        try:
            percentiles = np.linspace(0, 100, n_bins + 1)
            bin_edges = np.percentile(y_true, percentiles)
            
            for i in range(n_bins):
                if i == n_bins - 1:
                    mask = (y_true >= bin_edges[i]) & (y_true <= bin_edges[i + 1])
                else:
                    mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
                
                if np.sum(mask) >= 3:
                    metrics[f'mae_bin{i}'] = mean_absolute_error(y_true[mask], y_pred[mask])
                    metrics[f'mbe_bin{i}'] = np.mean(y_pred[mask] - y_true[mask])
        except Exception:
            pass
    
    return metrics


def summarize_folds(fold_metrics: List[Dict[str, float]],
                    compute_ci: bool = True,
                    ci_level: float = 0.95) -> Dict[str, float]:
    """
    汇总各折指标，计算均值和置信区间
    
    Parameters
    ----------
    fold_metrics : List[Dict]
        各折指标列表
    compute_ci : bool
        是否计算置信区间
    ci_level : float
        置信水平
        
    Returns
    -------
    summary : dict
        汇总指标（包含 _mean, _std, _ci_lower, _ci_upper）
    """
    from scipy import stats
    
    df = pd.DataFrame(fold_metrics)
    summary = {}
    
    for col in df.columns:
        if col in ['fold_id']:
            continue
        
        values = df[col].dropna().values
        if len(values) == 0:
            continue
        
        summary[f'{col}_mean'] = np.mean(values)
        summary[f'{col}_std'] = np.std(values, ddof=1) if len(values) > 1 else 0.0
        
        if compute_ci and len(values) > 2:
            # t 分布置信区间
            n = len(values)
            se = summary[f'{col}_std'] / np.sqrt(n)
            t_val = stats.t.ppf((1 + ci_level) / 2, n - 1)
            summary[f'{col}_ci_lower'] = summary[f'{col}_mean'] - t_val * se
            summary[f'{col}_ci_upper'] = summary[f'{col}_mean'] + t_val * se
    
    return summary


# ============================================================
# Group CV Protocol（主协议）
# ============================================================

class GroupCVProtocol:
    """
    主协议：Group K-Fold 交叉验证
    
    核心约束：
    - 按文献来源（group_id）分组
    - 同一组的样本不能同时出现在训练集和验证集
    - 所有拟合操作只在训练折进行
    """
    
    def __init__(self,
                 n_splits: int = 10,
                 random_seed: int = 42):
        self.n_splits = n_splits
        self.random_seed = random_seed
    
    def run(self,
            X: np.ndarray,
            y: np.ndarray,
            groups: np.ndarray,
            pipeline_factory: Callable[..., Pipeline],
            uncertainty_module: Optional[UncertaintyModule] = None,
            corr_module: Optional[CorrectionModule] = None,
            stratify_labels: Optional[np.ndarray] = None,
            verbose: bool = True) -> Dict[str, Any]:
        """
        执行 Group K-Fold CV
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵（原始，未标准化）
        y : np.ndarray
            目标值
        groups : np.ndarray
            分组标签
        pipeline_factory : Callable
            返回新 Pipeline 实例的工厂函数
        uncertainty_module : UncertaintyModule, optional
            不确定性估计模块
        verbose : bool
            是否打印进度
            
        Returns
        -------
        results : dict
            {
                'fold_metrics': pd.DataFrame,  # 每折指标
                'predictions': pd.DataFrame,   # 逐样本预测
                'summary': dict,               # 汇总指标
                'uncertainty': dict or None,   # 不确定性结果
            }
        """
        # 使用StratifiedKFold（不再使用GroupKFold）
        # 注意：移除Ref分组约束，优先保证P-T分布平衡
        if stratify_labels is None:
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X)
        else:
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_seed
            )
            split_iter = splitter.split(X, stratify_labels)

        oof_pred_raw = np.full(len(y), np.nan, dtype=np.float64)
        fold_records = []
        training_times = []

        for fold_idx, (train_idx, val_idx) in enumerate(split_iter):
            if verbose:
                print(f"  Fold {fold_idx + 1}/{self.n_splits}: ", end="")

            start_time = time.time()

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_train = groups[train_idx]
            stratify_train = stratify_labels[train_idx] if stratify_labels is not None else None

            try:
                pipeline = pipeline_factory(self.random_seed)
            except TypeError:
                pipeline = pipeline_factory()

            pipeline.fit(X_train, y_train, groups_train, stratify_labels=stratify_train)

            X_val_scaled, _ = pipeline.data_module.transform(X_val, pipeline._state)
            y_pred_raw = pipeline.predict(X_val_scaled, apply_correction=False)

            oof_pred_raw[val_idx] = y_pred_raw

            fold_time = time.time() - start_time
            training_times.append(fold_time)

            fold_records.append({
                'fold_id': fold_idx,
                'val_idx': val_idx,
                'y_val': y_val,
                'y_pred_raw': y_pred_raw,
                'pipeline': pipeline,
                'X_val': X_val,
                'training_time': fold_time,
            })

        if np.any(np.isnan(oof_pred_raw)):
            raise RuntimeError("OOF prediction contains NaN values.")

        if corr_module is None:
            from .correction_modules import NoCorrection
            corr_module = NoCorrection()

        corr_model = corr_module.fit(y, oof_pred_raw)

        fold_metrics = []
        all_predictions = []

        for record in fold_records:
            y_pred_corr = corr_module.apply(corr_model, record['y_pred_raw'])
            metrics = compute_all_metrics(record['y_val'], y_pred_corr, record['y_pred_raw'])
            metrics['fold_id'] = record['fold_id']
            metrics['training_time'] = record['training_time']
            fold_metrics.append(metrics)

            if verbose:
                print(f"RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.4f}")

            preds_df = pd.DataFrame({
                'fold_id': record['fold_id'],
                'sample_idx': record['val_idx'],
                'y_true': record['y_val'],
                'y_pred_raw': record['y_pred_raw'],
                'y_pred_corr': y_pred_corr,
                'residual': record['y_val'] - y_pred_corr,
            })
            all_predictions.append(preds_df)

        predictions_df = pd.concat(all_predictions, ignore_index=True)
        summary = summarize_folds(fold_metrics)
        summary['total_training_time'] = sum(training_times)

        uncertainty_results = None
        if uncertainty_module is not None:
            if verbose:
                print("  Running MC uncertainty across folds...")

            unc_fold_metrics = []
            for record in fold_records:
                pipeline = record['pipeline']
                pipeline.corr_module = corr_module
                pipeline._corr_model = corr_model

                dist = uncertainty_module.predict_distribution(pipeline, record['X_val'])
                calib_metrics = uncertainty_module.compute_calibration_metrics(record['y_val'], dist)
                calib_metrics['fold_id'] = record['fold_id']
                unc_fold_metrics.append(calib_metrics)

            unc_summary = summarize_folds(unc_fold_metrics)
            for k, v in unc_summary.items():
                summary[f"unc_{k}"] = v

            uncertainty_results = {
                'fold_metrics': pd.DataFrame(unc_fold_metrics),
                'summary': unc_summary,
            }

        return {
            'fold_metrics': pd.DataFrame(fold_metrics),
            'predictions': predictions_df,
            'summary': summary,
            'uncertainty': uncertainty_results,
            'corr_module': corr_module,
            'corr_model': corr_model,
        }

# ============================================================
# Random Split Protocol（对照协议）
# ============================================================

class RandomSplitProtocol:
    """
    对照协议：随机划分
    
    用于评估随机划分带来的乐观偏差
    """
    
    def __init__(self,
                 test_size: float = 0.2,
                 n_repeats: int = 5,
                 random_seed: int = 42):
        self.test_size = test_size
        self.n_repeats = n_repeats
        self.random_seed = random_seed
    
    def run(self,
            X: np.ndarray,
            y: np.ndarray,
            groups: np.ndarray,
            pipeline_factory: Callable[[], Pipeline],
            verbose: bool = True) -> Dict[str, Any]:
        """
        执行随机划分验证
        """
        repeat_metrics = []
        
        for i in range(self.n_repeats):
            if verbose:
                print(f"  Repeat {i + 1}/{self.n_repeats}: ", end="")
            
            # 随机划分
            X_train, X_val, y_train, y_val, g_train, _ = train_test_split(
                X, y, groups,
                test_size=self.test_size,
                random_state=self.random_seed + i
            )
            
            # 训练并预测
            pipeline = pipeline_factory()
            pipeline.fit(X_train, y_train, g_train)
            
            X_val_scaled, _ = pipeline.data_module.transform(X_val, pipeline._state)
            y_pred = pipeline.predict(X_val_scaled)
            
            # 计算指标
            metrics = compute_all_metrics(y_val, y_pred)
            metrics['repeat_id'] = i
            repeat_metrics.append(metrics)
            
            if verbose:
                print(f"RMSE={metrics['rmse']:.3f}")
        
        return {
            'repeat_metrics': pd.DataFrame(repeat_metrics),
            'summary': summarize_folds(repeat_metrics),
        }


# ============================================================
# 实验矩阵执行器
# ============================================================

@dataclass
class ExperimentConfig:
    """实验配置"""
    exp_id: str                    # 实验ID
    data_module_name: str          # M1 数据模块名称
    model_module_name: str         # M2 模型模块名称
    corr_module_name: str          # M3 校正模块名称
    feature_set: str = 'Liquid'    # 特征集选择：'NoLiquid' 或 'Liquid'
    data_params: Dict = field(default_factory=dict)
    model_params: Dict = field(default_factory=dict)
    corr_params: Dict = field(default_factory=dict)
    run_uncertainty: bool = False  # 是否运行 M4
    run_random_split: bool = False # 是否运行对照


class ExperimentMatrix:
    """
    实验矩阵执行器
    
    批量运行多个实验配置，保存结果
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y_T: np.ndarray,
                 y_P: np.ndarray,
                 groups: np.ndarray,
                 output_dir: str = 'results',
                 target_names: Tuple[str, str] = ('T', 'P')):
        """
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y_T : np.ndarray
            温度目标
        y_P : np.ndarray
            压力目标
        groups : np.ndarray
            分组标签
        output_dir : str
            输出目录
        target_names : tuple
            目标名称
        """
        self.X = X
        self.y_T = y_T
        self.y_P = y_P
        self.groups = groups
        self.output_dir = output_dir
        self.target_names = target_names
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    def run_experiments(self,
                        configs: List[ExperimentConfig],
                        n_splits: int = 10,
                        stratify_labels: Optional[np.ndarray] = None,
                        random_seed: int = 42,
                        verbose: bool = True) -> pd.DataFrame:
        """
        运行实验矩阵
        
        Returns
        -------
        summary_df : pd.DataFrame
            所有实验的汇总结果
        """
        from .data_modules import get_data_module
        from .model_modules import get_model_module
        from .correction_modules import get_correction_module
        from .uncertainty_modules import MCUncertaintyEstimator
        
        all_results = []
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"实验: {config.exp_id}")
            print(f"配置: {config.data_module_name} + {config.model_module_name} + {config.corr_module_name}")
            print(f"{'='*60}")
            
            # 创建 pipeline 工厂
            def apply_seed(params: Dict[str, Any], keys: List[str], seed: int) -> Dict[str, Any]:
                updated = dict(params)
                for key in keys:
                    if key not in updated:
                        updated[key] = seed
                return updated

            def make_pipeline_factory(cfg):
                def factory(seed: Optional[int] = None):
                    seed_value = random_seed if seed is None else seed

                    data_params = apply_seed(cfg.data_params, ['random_seed'], seed_value)
                    model_params = dict(cfg.model_params)
                    if cfg.model_module_name.lower() in ['ert', 'extratrees', 'rf', 'randomforest']:
                        model_params = apply_seed(model_params, ['random_state'], seed_value)
                    else:
                        model_params = apply_seed(model_params, ['random_seed'], seed_value)

                    data_mod = get_data_module(cfg.data_module_name, **data_params)
                    model_mod = get_model_module(cfg.model_module_name, **model_params)
                    corr_mod = get_correction_module(cfg.corr_module_name, **cfg.corr_params)
                    return Pipeline(data_mod, model_mod, corr_mod)
                return factory

            pipeline_factory = make_pipeline_factory(config)
            
            # 不确定性模块
            unc_module = MCUncertaintyEstimator(random_seed=random_seed) if config.run_uncertainty else None
            
            # 运行两个目标（T 和 P）
            exp_result = {'exp_id': config.exp_id}
            
            for target_name, y in [('T', self.y_T), ('P', self.y_P)]:
                print(f"\n--- 目标: {target_name} ---")
                
                # 主协议
                protocol = GroupCVProtocol(n_splits=n_splits, random_seed=random_seed)
                corr_module = get_correction_module(config.corr_module_name, **config.corr_params)
                results = protocol.run(
                    self.X, y, self.groups,
                    pipeline_factory,
                    uncertainty_module=unc_module,
                    corr_module=corr_module,
                    stratify_labels=stratify_labels,
                    verbose=verbose
                )
                
                # 保存每折指标
                results['fold_metrics'].to_csv(
                    os.path.join(self.output_dir, f'{config.exp_id}_{target_name}_fold_metrics.csv'),
                    index=False
                )
                
                # 保存预测
                results['predictions'].to_parquet(
                    os.path.join(self.output_dir, f'{config.exp_id}_{target_name}_predictions.parquet'),
                    index=False
                )
                
                # 汇总到结果
                for k, v in results['summary'].items():
                    exp_result[f'{target_name}_{k}'] = v
                
                # 对照协议（如果需要）
                if config.run_random_split:
                    print(f"\n  [对照] Random Split:")
                    random_protocol = RandomSplitProtocol()
                    random_results = random_protocol.run(
                        self.X, y, self.groups,
                        pipeline_factory,
                        verbose=verbose
                    )
                    
                    for k, v in random_results['summary'].items():
                        exp_result[f'{target_name}_random_{k}'] = v
            
            all_results.append(exp_result)
        
        # 汇总 DataFrame
        summary_df = pd.DataFrame(all_results)
        summary_df.to_csv(
            os.path.join(self.output_dir, 'metrics_summary.csv'),
            index=False
        )
        
        return summary_df
    
    def run_stability_repeats(self,
                              configs: List[ExperimentConfig],
                              X_test: np.ndarray,
                              y_T_test: np.ndarray,
                              y_P_test: np.ndarray,
                              groups_test: np.ndarray,
                              stratify_labels: Optional[np.ndarray] = None,
                              n_splits: int = 10,
                              test_size: float = 0.3,
                              n_repeats: int = 1000,
                              random_seed: int = 0,
                              verbose: bool = True) -> pd.DataFrame:
        from .data_modules import get_data_module
        from .model_modules import get_model_module
        from .correction_modules import get_correction_module

        stability_dir = os.path.join(self.output_dir, 'stability')
        os.makedirs(stability_dir, exist_ok=True)

        summary_rows = []

        for config in configs:
            print(f"\n{'='*60}")
            print(f"Stability: {config.exp_id}")
            print(f"Config: {config.data_module_name} + {config.model_module_name} + {config.corr_module_name}")
            print(f"{'='*60}")

            def apply_seed(params: Dict[str, Any], keys: List[str], seed: int) -> Dict[str, Any]:
                updated = dict(params)
                for key in keys:
                    if key not in updated:
                        updated[key] = seed
                return updated

            def make_pipeline_factory(cfg):
                def factory(seed: Optional[int] = None):
                    seed_value = random_seed if seed is None else seed

                    data_params = apply_seed(cfg.data_params, ['random_seed'], seed_value)
                    model_params = dict(cfg.model_params)
                    if cfg.model_module_name.lower() in ['ert', 'extratrees', 'rf', 'randomforest']:
                        model_params = apply_seed(model_params, ['random_state'], seed_value)
                    else:
                        model_params = apply_seed(model_params, ['random_seed'], seed_value)

                    data_mod = get_data_module(cfg.data_module_name, **data_params)
                    model_mod = get_model_module(cfg.model_module_name, **model_params)
                    corr_mod = get_correction_module(cfg.corr_module_name, **cfg.corr_params)
                    return Pipeline(data_mod, model_mod, corr_mod)
                return factory

            pipeline_factory = make_pipeline_factory(config)
            corr_module = get_correction_module(config.corr_module_name, **config.corr_params)

            for target_name, y_full, y_test in [('T', self.y_T, y_T_test), ('P', self.y_P, y_P_test)]:
                repeat_metrics = []
                idx_all = np.arange(len(self.X))

                for i in range(n_repeats):
                    seed = random_seed + i
                    train_idx, _ = train_test_split(
                        idx_all,
                        test_size=test_size,
                        random_state=seed
                    )

                    X_train = self.X[train_idx]
                    y_train = y_full[train_idx]
                    groups_train = self.groups[train_idx]
                    stratify_train = stratify_labels[train_idx] if stratify_labels is not None else None

                    protocol = GroupCVProtocol(n_splits=n_splits, random_seed=seed)
                    cv_results = protocol.run(
                        X_train,
                        y_train,
                        groups_train,
                        pipeline_factory,
                        uncertainty_module=None,
                        corr_module=corr_module,
                        stratify_labels=stratify_train,
                        verbose=False
                    )
                    corr_model = cv_results['corr_model']

                    pipeline = pipeline_factory(seed)
                    pipeline.fit(X_train, y_train, groups_train, stratify_labels=stratify_train)
                    pipeline.corr_module = corr_module
                    pipeline._corr_model = corr_model

                    X_test_scaled, _ = pipeline.data_module.transform(X_test, pipeline._state)
                    y_pred_raw = pipeline.predict(X_test_scaled, apply_correction=False)
                    y_pred_corr = corr_module.apply(corr_model, y_pred_raw)

                    metrics = compute_all_metrics(y_test, y_pred_corr, y_pred_raw)
                    metrics['repeat_id'] = i
                    repeat_metrics.append(metrics)

                    if verbose and (i + 1) % 50 == 0:
                        print(f"  Repeat {i + 1}/{n_repeats}")

                repeat_df = pd.DataFrame(repeat_metrics)
                repeat_df.to_csv(
                    os.path.join(stability_dir, f'{config.exp_id}_{target_name}_test_metrics.csv'),
                    index=False
                )

                summary = summarize_folds(repeat_metrics, compute_ci=True)
                summary_row = {'exp_id': config.exp_id, 'target': target_name}
                summary_row.update(summary)
                summary_rows.append(summary_row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(
            os.path.join(stability_dir, 'stability_summary.csv'),
            index=False
        )

        return summary_df
    def compute_effect_table(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算模块效应表
        
        比较不同配置相对于基线的改进
        """
        # 找到基线（通常是第一个实验）
        baseline = summary_df.iloc[0]
        
        effects = []
        for _, row in summary_df.iterrows():
            effect = {'exp_id': row['exp_id']}
            
            for target in ['T', 'P']:
                # RMSE 改进（负值表示改进）
                if f'{target}_rmse_mean' in row and f'{target}_rmse_mean' in baseline:
                    effect[f'{target}_delta_rmse'] = row[f'{target}_rmse_mean'] - baseline[f'{target}_rmse_mean']
                    effect[f'{target}_pct_rmse'] = (effect[f'{target}_delta_rmse'] / baseline[f'{target}_rmse_mean']) * 100
                
                # MBE 改进
                if f'{target}_mbe_mean' in row:
                    effect[f'{target}_delta_mbe'] = abs(row[f'{target}_mbe_mean']) - abs(baseline.get(f'{target}_mbe_mean', 0))
            
            effects.append(effect)
        
        effect_df = pd.DataFrame(effects)
        effect_df.to_csv(
            os.path.join(self.output_dir, 'effect_table.csv'),
            index=False
        )
        
        return effect_df
    
    def save_config(self, configs: List[ExperimentConfig], extra_info: Dict = None):
        """保存实验配置"""
        config_data = {
            'experiments': [
                {
                    'exp_id': c.exp_id,
                    'data_module': c.data_module_name,
                    'model_module': c.model_module_name,
                    'corr_module': c.corr_module_name,
                    'data_params': c.data_params,
                    'model_params': c.model_params,
                    'corr_params': c.corr_params,
                }
                for c in configs
            ],
            'data_shape': {
                'n_samples': len(self.X),
                'n_features': self.X.shape[1],
                'n_groups': len(np.unique(self.groups)),
            }
        }
        
        if extra_info:
            config_data.update(extra_info)
        
        with open(os.path.join(self.output_dir, 'config_used.yaml'), 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
