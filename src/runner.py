# -*- coding: utf-8 -*-
"""
机器学习温压计评估框架 - 实验运行器模块
Runner Module: ExperimentRunner, ExperimentConfig

核心约束：
1. 外层评估必须使用 GroupKFold（按 Ref 分组）
2. T 与 P 采用两条独立建模链路分别训练、预测、校正与评估
3. 所有拟合操作只能在训练折内完成
4. 每折结束立即落盘（fail-safe）
"""

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .models import BaseThermoModel, get_model, GroupAwareStacker


# ============================================================
# 实验配置
# ============================================================

@dataclass
class ExperimentConfig:
    """
    实验配置类
    
    Parameters
    ----------
    exp_name : str
        实验名称（如 'exp1_catboost_base'）
    model_type : str
        模型类型 ('catboost' 或 'stacking')
    model_params : dict
        模型参数字典
    augment : bool
        是否进行数据增强
    correct : bool
        是否进行偏差校正
    n_splits : int
        外层 GroupKFold 折数
    random_seed : int
        随机种子
    output_dir : str
        输出目录
    """
    exp_name: str
    model_type: str = 'catboost'
    model_params: Dict[str, Any] = field(default_factory=lambda: {'iterations': 1000, 'depth': 6})
    augment: bool = False
    correct: bool = False
    n_splits: int = 5
    random_seed: int = 42
    output_dir: str = 'outputs'


# ============================================================
# 单目标运行器（T 或 P）
# ============================================================

class SingleTargetRunner:
    """
    单目标（T 或 P）的实验运行器
    
    实现 fold-safe 的训练、预测、校正流程
    """
    
    def __init__(self, target_name: str, config: ExperimentConfig):
        """
        Parameters
        ----------
        target_name : str
            目标名称 ('T' 或 'P')
        config : ExperimentConfig
            实验配置
        """
        self.target_name = target_name
        self.config = config
        
        # 存储各折结果
        self.fold_metrics: List[Dict[str, float]] = []
        self.fold_predictions: List[pd.DataFrame] = []
    
    def _create_model(self) -> BaseThermoModel:
        """创建模型实例"""
        return get_model(self.config.model_type, **self.config.model_params)
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                      aug_factor: int = 2, noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        数据增强（仅在训练集使用）
        
        方法：添加高斯噪声
        """
        n_samples = X.shape[0]
        X_aug_list = [X]
        y_aug_list = [y]
        groups_aug_list = [groups]
        
        for _ in range(aug_factor - 1):
            noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0)
            X_aug_list.append(X + noise)
            y_aug_list.append(y)
            groups_aug_list.append(groups)
        
        return (
            np.vstack(X_aug_list),
            np.concatenate(y_aug_list),
            np.concatenate(groups_aug_list)
        )
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算评估指标（完整版，包含 slope/intercept/bias_mean/resid_std）"""
        from .metrics import compute_metrics

        # 使用更新后的 compute_metrics 函数，返回完整指标
        return compute_metrics(y_true, y_pred, prefix=f'{self.target_name}_')
    
    def run_fold(self, fold_idx: int, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray,
                 X_val: np.ndarray, y_val: np.ndarray, row_ids_val: np.ndarray, refs_val: np.ndarray,
                 corrector: Optional[Any] = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        """
        运行单折实验
        
        Parameters
        ----------
        fold_idx : int
            折索引
        X_train, y_train, groups_train : np.ndarray
            训练集数据
        X_val, y_val : np.ndarray
            验证集数据
        row_ids_val, refs_val : np.ndarray
            验证集的行ID和文献来源
        corrector : BiasCorrector, optional
            偏差校正器（如果 config.correct=True 则必须提供）
        
        Returns
        -------
        metrics : dict
            当前折的指标
        preds_df : pd.DataFrame
            逐样本预测表
        """
        np.random.seed(self.config.random_seed + fold_idx)
        
        # 1. 标准化（只在训练集 fit）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 2. 数据增强（仅训练集）
        if self.config.augment:
            X_train_aug, y_train_aug, groups_train_aug = self._augment_data(
                X_train_scaled, y_train, groups_train
            )
        else:
            X_train_aug, y_train_aug, groups_train_aug = X_train_scaled, y_train, groups_train
        
        # 3. 训练模型
        model = self._create_model()
        
        if isinstance(model, GroupAwareStacker):
            # Stacking 需要 groups 参数
            model.fit(X_train_aug, y_train_aug, groups_train_aug)
            # 获取训练集 OOF 预测（用于偏差校正）
            y_train_oof = model.get_oof_predictions(X_train_aug, y_train_aug, groups_train_aug)
        else:
            model.fit(X_train_aug, y_train_aug)
            # 非 Stacking 模型：使用训练集预测作为 OOF（简化处理）
            # 注意：理想情况应该用 inner CV 生成 OOF
            y_train_oof = model.predict(X_train_aug)
        
        # 4. 预测验证集
        y_pred_raw = model.predict(X_val_scaled)
        
        # 5. 偏差校正
        if self.config.correct and corrector is not None:
            # 用训练集 OOF 拟合校正器
            corrector.fit_on_oof(y_train_aug, y_train_oof)
            y_pred_corr = corrector.transform(y_pred_raw)
        else:
            y_pred_corr = y_pred_raw.copy()
        
        # 6. 计算指标
        metrics = self._compute_metrics(y_val, y_pred_corr)
        metrics['fold_id'] = fold_idx
        
        # 7. 构建逐样本预测表（包含原始预测、校正预测和残差）
        preds_df = pd.DataFrame({
            'row_id': row_ids_val,
            'Ref': refs_val,
            f'{self.target_name}_true': y_val,
            f'{self.target_name}_pred_raw': y_pred_raw,
            f'{self.target_name}_pred_corr': y_pred_corr,
            f'{self.target_name}_residual': y_val - y_pred_corr,  # 残差 = 真值 - 校正预测
            'fold_id': fold_idx,
            'exp_name': self.config.exp_name
        })
        
        return metrics, preds_df


# ============================================================
# 完整实验运行器（T + P 双链路）
# ============================================================

class ExperimentRunner:
    """
    完整实验运行器
    
    T 与 P 采用两条独立建模链路分别训练、预测、校正与评估
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Parameters
        ----------
        config : ExperimentConfig
            实验配置
        """
        self.config = config
        self.output_dir = os.path.join(config.output_dir, config.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 结果存储
        self.all_metrics: List[Dict[str, Any]] = []
        self.all_predictions: List[pd.DataFrame] = []
    
    def run_experiment(self, X: np.ndarray, y_T: np.ndarray, y_P: np.ndarray,
                       groups: np.ndarray, row_ids: np.ndarray, refs: np.ndarray,
                       corrector_T: Optional[Any] = None, corrector_P: Optional[Any] = None) -> Dict[str, Any]:
        """
        运行完整实验（T + P 双链路）
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            特征矩阵
        y_T : np.ndarray, shape (n_samples,)
            温度目标
        y_P : np.ndarray, shape (n_samples,)
            压力目标
        groups : np.ndarray, shape (n_samples,)
            分组标签
        row_ids : np.ndarray, shape (n_samples,)
            行索引
        refs : np.ndarray, shape (n_samples,)
            文献来源
        corrector_T, corrector_P : BiasCorrector, optional
            偏差校正器
        
        Returns
        -------
        results : dict
            包含汇总指标和各折详情
        """
        gkf = GroupKFold(n_splits=self.config.n_splits)
        
        # 初始化单目标运行器
        runner_T = SingleTargetRunner('T', self.config)
        runner_P = SingleTargetRunner('P', self.config)
        
        print(f"\n{'='*60}")
        print(f"实验: {self.config.exp_name}")
        print(f"模型: {self.config.model_type}, 增强: {self.config.augment}, 校正: {self.config.correct}")
        print(f"{'='*60}")
        
        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y_T, groups)):
            print(f"\n[Fold {fold_idx + 1}/{self.config.n_splits}] ", end="")
            print(f"训练: {len(train_idx)} 样本, 验证: {len(val_idx)} 样本")
            
            # 切分数据
            X_train, X_val = X[train_idx], X[val_idx]
            y_T_train, y_T_val = y_T[train_idx], y_T[val_idx]
            y_P_train, y_P_val = y_P[train_idx], y_P[val_idx]
            groups_train = groups[train_idx]
            row_ids_val = row_ids[val_idx]
            refs_val = refs[val_idx]
            
            # 为每个折创建新的校正器实例（避免状态污染）
            from .correction import LinearBiasCorrector, IdentityCorrector
            if self.config.correct:
                fold_corrector_T = LinearBiasCorrector()
                fold_corrector_P = LinearBiasCorrector()
            else:
                fold_corrector_T = IdentityCorrector()
                fold_corrector_P = IdentityCorrector()
            
            # 运行 T 链路
            metrics_T, preds_T = runner_T.run_fold(
                fold_idx, X_train, y_T_train, groups_train,
                X_val, y_T_val, row_ids_val, refs_val, fold_corrector_T
            )
            
            # 运行 P 链路
            metrics_P, preds_P = runner_P.run_fold(
                fold_idx, X_train, y_P_train, groups_train,
                X_val, y_P_val, row_ids_val, refs_val, fold_corrector_P
            )
            
            # 合并指标
            fold_metrics = {**metrics_T, **metrics_P}
            self.all_metrics.append(fold_metrics)
            
            # 合并预测表
            preds_merged = preds_T.merge(
                preds_P[['row_id', 'P_true', 'P_pred_raw', 'P_pred_corr']],
                on='row_id'
            )
            self.all_predictions.append(preds_merged)
            
            # 打印当前折指标
            print(f"  T: RMSE={metrics_T['rmse_T']:.2f}, R²={metrics_T['r2_T']:.4f}")
            print(f"  P: RMSE={metrics_P['rmse_P']:.3f}, R²={metrics_P['r2_P']:.4f}")
            
            # 立即落盘（fail-safe）
            self._save_fold_results(fold_idx)
        
        # 汇总结果
        results = self._summarize_results()
        self._save_final_results(results)
        
        return results
    
    def _save_fold_results(self, fold_idx: int) -> None:
        """保存单折结果（增量保存）"""
        # 保存指标
        metrics_df = pd.DataFrame(self.all_metrics)
        metrics_path = os.path.join(self.output_dir, 'metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        
        # 保存预测
        preds_df = pd.concat(self.all_predictions, ignore_index=True)
        preds_path = os.path.join(self.output_dir, 'preds.parquet')
        preds_df.to_parquet(preds_path, index=False)
    
    def _summarize_results(self) -> Dict[str, Any]:
        """汇总所有折的结果"""
        metrics_df = pd.DataFrame(self.all_metrics)
        
        summary = {
            'exp_name': self.config.exp_name,
            'n_folds': self.config.n_splits,
        }
        
        # 计算均值和标准差
        for col in metrics_df.columns:
            if col != 'fold_id' and col != 'exp_name':
                summary[f'{col}_mean'] = metrics_df[col].mean()
                summary[f'{col}_std'] = metrics_df[col].std()
        
        return summary
    
    def _save_final_results(self, results: Dict[str, Any]) -> None:
        """保存最终汇总结果"""
        summary_path = os.path.join(self.output_dir, 'summary.csv')
        pd.DataFrame([results]).to_csv(summary_path, index=False)
        
        print(f"\n{'='*60}")
        print("实验完成！汇总结果：")
        print(f"  T: RMSE={results['rmse_T_mean']:.2f}±{results['rmse_T_std']:.2f}, R²={results['r2_T_mean']:.4f}±{results['r2_T_std']:.4f}")
        print(f"  P: RMSE={results['rmse_P_mean']:.3f}±{results['rmse_P_std']:.3f}, R²={results['r2_P_mean']:.4f}±{results['r2_P_std']:.4f}")
        print(f"\n输出目录: {self.output_dir}")
        print(f"{'='*60}")


# ============================================================
# 便捷函数
# ============================================================

def run_single_experiment(X: np.ndarray, y_T: np.ndarray, y_P: np.ndarray,
                          groups: np.ndarray, row_ids: np.ndarray, refs: np.ndarray,
                          config: ExperimentConfig) -> Dict[str, Any]:
    """
    运行单个实验的便捷函数
    
    Examples
    --------
    >>> config = ExperimentConfig(exp_name='exp1_catboost_base', model_type='catboost')
    >>> results = run_single_experiment(X, y_T, y_P, groups, row_ids, refs, config)
    """
    runner = ExperimentRunner(config)
    return runner.run_experiment(X, y_T, y_P, groups, row_ids, refs)


def run_experiment_matrix(X: np.ndarray, y_T: np.ndarray, y_P: np.ndarray,
                          groups: np.ndarray, row_ids: np.ndarray, refs: np.ndarray,
                          configs: List[ExperimentConfig]) -> pd.DataFrame:
    """
    运行实验矩阵（多个实验配置）
    
    Examples
    --------
    >>> configs = [
    ...     ExperimentConfig('exp1', model_type='catboost', augment=False, correct=False),
    ...     ExperimentConfig('exp2', model_type='catboost', augment=True, correct=False),
    ... ]
    >>> results_df = run_experiment_matrix(X, y_T, y_P, groups, row_ids, refs, configs)
    """
    all_results = []
    
    for config in configs:
        runner = ExperimentRunner(config)
        results = runner.run_experiment(X, y_T, y_P, groups, row_ids, refs)
        all_results.append(results)
    
    return pd.DataFrame(all_results)
