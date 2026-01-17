# 机器学习温压计模块化评估协议

面向 Jupyter 编排的可复用 Python 工程骨架，用于辉石地质温压计的机器学习模型标准化评估。

## 🎯 核心目标

通过**模块化消融实验**系统性评估各模块的独立贡献与性能边界：
- **M1 - 数据模块**：增强/重加权对模型泛化能力的贡献
- **M2 - 算法模块**：CatBoost vs 异构 Stacking 的复杂度收益分析
- **M3 - 校正模块**：OOF 偏差校正对预测精度的提升

**核心理念**：不是刷榜，而是建立可审计、可复现、可扩展的实验矩阵。

---

## ✨ 核心特性

- **GroupKFold 外层评估**：按文献来源 (`Ref`) 分组，严禁随机划分
- **T/P 独立双链路**：温度与压力采用独立建模链路
- **Fold-safe 设计**：标准化、增强、校正均在训练折内完成
- **完整评估指标**：RMSE、MAE、R²、Slope、Intercept、Bias、Resid_Std
- **可审计输出**：每折输出 `metrics.csv` + `preds.parquet`（含 raw/corr 预测）
- **异构 Stacking**：ExtraTrees + XGBoost + CatBoost（Group-aware OOF）
- **论文级可视化**：4张核心图表（阶梯误差、校正对比、特征重要性、残差分布）

---

## 📁 项目结构

```
ml-thermobarometer-benchmark/
├── config.py                    # 全局配置（5组实验矩阵）
├── requirements.txt             # Python 依赖
├── input.csv                    # 校准数据集（2079行×45列）
├── src/
│   ├── __init__.py              # 模块导出
│   ├── models.py                # CatBoost、ExtraTrees、XGBoost、Stacking
│   ├── runner.py                # 实验运行器（T/P双链路）
│   ├── correction.py            # 偏差校正器（Fold-safe）
│   ├── preprocessing.py         # 数据预处理
│   ├── metrics.py               # 指标计算（含 slope/intercept）
│   └── viz.py                   # 可视化（4个核心图表函数）
├── notebooks/
│   └── run_experiments.ipynb    # Jupyter 一键运行入口
├── outputs/                     # 实验输出
│   ├── exp1_baseline/           # 基线实验
│   ├── exp2_aug_only/           # 仅数据增强
│   ├── exp3_corr_only/          # 仅偏差校正（新增）
│   ├── exp4_aug_corr/           # 增强+校正组合
│   ├── exp5_stacking/           # 异构Stacking
│   └── figures/                 # 论文图表输出
└── reference_files/             # 参考文件（R脚本、论文PDF）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

**主要依赖**：
- `catboost>=1.0` - CatBoost 模型
- `xgboost>=1.5` - XGBoost 模型（异构Stacking）
- `scipy>=1.7` - 统计分析（KDE、置信区间）
- `scikit-learn>=1.0` - ExtraTrees、指标计算
- `pandas`, `numpy`, `matplotlib`, `seaborn` - 数据处理与可视化

### 2. 一键运行（推荐）

打开 `notebooks/run_experiments.ipynb`，按顺序执行所有单元格：

```python
# Cell 1: 加载数据
from src import load_data, prepare_data
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')  # 36列特征

# Cell 2: 运行5组实验
from config import EXPERIMENT_CONFIGS
from src import ExperimentConfig, ExperimentRunner

for exp_name in ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only',
                  'exp4_aug_corr', 'exp5_stacking']:
    config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
    runner = ExperimentRunner(config)
    results = runner.run_experiment(
        X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
        groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
    )

# Cell 3: 生成可视化
from src.viz import (plot_stepwise_rmse_comparison, plot_correction_effect,
                      plot_feature_importance, plot_residual_distribution_comparison)

# 阶梯误差对比图
fig1 = plot_stepwise_rmse_comparison(metrics_dict, target='T',
                                     save_path='outputs/figures/fig1_stepwise_rmse_T.png')
```

### 3. 命令行运行（可选）

```python
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner

# 加载数据
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

# 配置实验
config = ExperimentConfig(
    exp_name='exp4_aug_corr',
    model_type='catboost',
    model_params={'iterations': 1000, 'depth': 6},
    augment=True,
    correct=True
)

# 运行
runner = ExperimentRunner(config)
results = runner.run_experiment(
    X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
    groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
)
```

---

## 🧪 实验矩阵（5组消融实验）

| 实验编号 | 数据增强 | 偏差校正 | 模型 | 目的 |
|---------|---------|---------|------|------|
| **exp1_baseline** | ❌ | ❌ | CatBoost | 基线（无任何优化） |
| **exp2_aug_only** | ✅ | ❌ | CatBoost | M1评估（仅数据增强） |
| **exp3_corr_only** | ❌ | ✅ | CatBoost | M3评估（仅偏差校正）⭐新增 |
| **exp4_aug_corr** | ✅ | ✅ | CatBoost | M1+M3组合（标准流程） |
| **exp5_stacking** | ✅ | ✅ | Stacking | M2评估（复杂集成边界） |

**Stacking 配置**：异构基学习器
- ExtraTreesRegressor (n_estimators=200, max_depth=10)
- XGBRegressor (n_estimators=200, max_depth=6, lr=0.05)
- CatBoostRegressor (iterations=200, depth=6, lr=0.05)
- 元学习器：CatBoostRegressor (iterations=100, depth=4)

---

## 📊 评估指标（完整版）

### 基础指标
- **RMSE** - 均方根误差
- **MAE** - 平均绝对误差
- **R²** - 决定系数

### 校正诊断指标（新增）
- **Slope** - 预测-真值回归斜率（理想值≈1）
- **Intercept** - 预测-真值回归截距（理想值≈0）
- **Bias_mean** - 偏差均值（残差均值）
- **Resid_std** - 残差标准差

**用途**：量化偏差校正的有效性（校正后 slope→1, intercept→0）

---

## 📁 输出格式

### metrics.csv（完整指标）

| 列名 | 说明 | 示例值 |
|------|------|--------|
| fold_id | 折索引 | 0, 1, 2, 3, 4 |
| T_rmse | 温度RMSE | 28.5 |
| T_mae | 温度MAE | 21.3 |
| T_r2 | 温度R² | 0.89 |
| **T_slope** | 温度预测斜率 ⭐新增 | 0.98 |
| **T_intercept** | 温度预测截距 ⭐新增 | 8.2 |
| **T_bias_mean** | 温度偏差均值 ⭐新增 | -1.5 |
| **T_resid_std** | 温度残差标准差 ⭐新增 | 27.8 |
| P_rmse, P_mae, ... | 压力指标（同上） | ... |

### preds.parquet（逐样本预测）

| 列名 | 说明 |
|------|------|
| row_id | 样本索引 |
| Ref | 文献来源（分组） |
| fold_id | 折索引 |
| exp_name | 实验名称 |
| T_true | 温度真值 |
| **T_pred_raw** | 温度原始预测（校正前）⭐ |
| **T_pred_corr** | 温度校正预测（校正后）⭐ |
| **T_residual** | 温度残差（true - corr）⭐新增 |
| P_true, P_pred_raw, ... | 压力字段（同上） |

---

## 📈 可视化函数（论文级图表）

### 1. 阶梯误差对比图
```python
from src.viz import plot_stepwise_rmse_comparison

fig = plot_stepwise_rmse_comparison(
    results_dict=metrics_dict,  # {exp_name: metrics_df}
    target='T',  # 'T' 或 'P'
    save_path='outputs/figures/fig1_stepwise_rmse_T.png'
)
```
**用途**：展示 Exp1→Exp5 的 RMSE 递减趋势，自动标注相对提升百分比

### 2. 校正前后散点图
```python
from src.viz import plot_correction_effect

fig = plot_correction_effect(
    preds_df=preds_dict['exp4_aug_corr'],
    exp_name='exp4_aug_corr',
    target='T',
    save_path='outputs/figures/fig2_correction_effect_T.png'
)
```
**用途**：双子图对比校正效果，显示斜率从偏离1回归到~1

### 3. 特征重要性图
```python
from src.viz import plot_feature_importance

fig = plot_feature_importance(
    model=trained_model,
    feature_names=data['feature_cols'],
    target='T',
    top_n=20,
    save_path='outputs/figures/fig3_feature_importance_T.png'
)
```
**用途**：识别关键氧化物/阳离子特征（Top 20）

### 4. 残差分布对比图
```python
from src.viz import plot_residual_distribution_comparison

fig = plot_residual_distribution_comparison(
    results_dict=preds_dict,
    exp_names=['exp4_aug_corr', 'exp5_stacking'],
    target='T',
    save_path='outputs/figures/fig4_residual_comparison_T.png'
)
```
**用途**：对比 Exp4 vs Exp5 的残差分布，说明 Stacking 收益递减

---

## ⚙️ 核心协议约束（不可违反）

1. **外层评估**：必须使用 `GroupKFold`，groups 来自 `Ref` 列
2. **训练折隔离**：所有拟合操作（标准化、增强、超参、stacking inner CV、偏差校正）只在训练折内完成
3. **偏差校正**：校正器只能用训练折 OOF 拟合（fold-safe），严禁使用包含验证折的全局 OOF
4. **Stacking**：inner CV 必须使用 `GroupKFold` 生成 OOF 元特征
5. **T/P 独立**：温度和压力采用两条独立建模链路

---

## 📖 预期论文产出

### 表3-1：温度预测性能对比（示例）

| 实验 | RMSE (℃) | MAE (℃) | R² | Slope | Intercept | 相对Exp1提升 |
|------|----------|---------|-----|-------|-----------|------------|
| Exp1 | 35.2±2.1 | 27.3 | 0.82 | 0.95 | 18.5 | - |
| Exp2 | 32.8±1.9 | 25.1 | 0.85 | 0.96 | 15.2 | ↓6.8% |
| Exp3 | 30.5±1.7 | 23.4 | 0.87 | 0.98 | 8.1 | ↓13.4% |
| Exp4 | 28.1±1.6 | 21.8 | 0.90 | 0.99 | 3.2 | ↓20.2% |
| Exp5 | 27.9±1.5 | 21.5 | 0.90 | 0.99 | 2.8 | ↓20.7% |

**关键发现**：
- M1（数据增强）单独贡献：6.8% RMSE 降低
- M3（偏差校正）单独贡献：13.4% RMSE 降低（**主要贡献**）
- M1+M3 协同效应：20.2% RMSE 降低
- M2（Stacking）边际收益：仅 0.5%（**复杂度不值得**）

---

## 🔍 验证与调试

### 快速验证（1折测试）
```python
# 临时修改 config.py
N_SPLITS = 1  # 仅1折快速验证

# 运行 exp1 和 exp4 验证输出格式
```

### 输出检查
```python
import os
import pandas as pd

# 检查目录存在
assert os.path.exists('outputs/exp3_corr_only/metrics.csv')

# 检查新增指标列
df_metrics = pd.read_csv('outputs/exp4_aug_corr/metrics.csv')
assert 'T_slope' in df_metrics.columns
assert 'T_intercept' in df_metrics.columns

# 检查预测值列
df_preds = pd.read_parquet('outputs/exp4_aug_corr/preds.parquet')
assert 'T_pred_raw' in df_preds.columns
assert 'T_pred_corr' in df_preds.columns
assert 'T_residual' in df_preds.columns
```

---

## 📚 参考文档

- **MANUAL.md** - 详细使用手册（模块API、自定义配置）
- **reference_files/** - 参考实现（R脚本、论文PDF）
- **notebooks/run_experiments.ipynb** - 完整实验流程示例

---

## 📄 License

MIT

---

## 🙏 致谢

本项目参考了 Jorgenson et al. (2022) 的机器学习温压计工作，并在此基础上构建了模块化评估协议。
