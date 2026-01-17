# 机器学习温压计模块化评估协议

面向 Jupyter 编排的可复用 Python 工程骨架，用于辉石（Clinopyroxene）地质温压计的机器学习模型标准化评估。

## 核心目标

通过**模块化消融实验**（Ablation Study）系统性评估各模块的独立贡献与性能边界：
- **M1 - 数据模块**：量化数据增强对模型泛化能力的贡献
- **M2 - 算法模块**：对比 CatBoost vs 异构 Stacking 的复杂度收益
- **M3 - 校正模块**：评估 OOF 偏差校正对预测精度的提升

**核心理念**：建立可审计、可复现、可扩展的实验矩阵，而非单纯追求最优指标。

---

## 核心特性

- ✅ **严格的交叉验证协议**：外层使用 GroupKFold 按文献来源（Ref）分组
- ✅ **T/P 独立双链路**：温度（T）与压力（P）采用完全独立的建模流程
- ✅ **Fold-safe 设计**：所有拟合操作（标准化、增强、校正）仅在训练折内完成
- ✅ **完整评估指标**：RMSE、MAE、R²、Slope、Intercept、Bias_mean、Resid_std（7个指标）
- ✅ **可审计输出**：每折输出完整指标表 + 逐样本预测表（含 raw/corr 预测）
- ✅ **异构 Stacking**：ExtraTrees + XGBoost + CatBoost（Group-aware OOF）
- ✅ **论文级可视化**：4个核心图表函数

---

## 项目结构

```
ml-thermobarometer-benchmark/
│
├── input.csv                    # 校准数据集（2079行×45列，latin-1编码）
├── config.py                    # 全局配置（5组实验矩阵、特征集合定义）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
├── .md                    #  Code 指导文件
│
├── src/                         # 核心代码模块
│   ├── __init__.py              # 模块导出
│   ├── models.py                # 模型（CatBoost、ExtraTrees、XGBoost、Stacking）
│   ├── runner.py                # 实验运行器（T/P双链路、Fold-safe）
│   ├── correction.py            # 偏差校正器（Linear、Identity）
│   ├── preprocessing.py         # 数据预处理（加载、增强、特征选择）
│   ├── metrics.py               # 指标计算（含 slope/intercept 等扩展指标）
│   └── viz.py                   # 可视化（4个核心图表函数）
│
├── notebooks/                   # Jupyter 入口
│   └── run_experiments.ipynb    # 一键运行5组实验
│
├── outputs/                     # 实验输出目录
│   ├── exp1_baseline/           # 基线实验
│   ├── exp2_aug_only/           # 仅数据增强
│   ├── exp3_corr_only/          # 仅偏差校正
│   ├── exp4_aug_corr/           # 增强+校正组合
│   ├── exp5_stacking/           # 异构Stacking
│   ├── cache/                   # 模型缓存（Stacking元特征）
│   └── figures/                 # 论文图表输出
│
└── reference_files/             # 参考文件（R脚本、论文PDF等）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "from src import models, runner, metrics, viz; print('OK')"
```

### 3. 快速验证（1折测试）

```python
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS
import config

config.N_SPLITS = 1  # 临时改为1折快速验证
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

config_obj = ExperimentConfig(exp_name='test', **EXPERIMENT_CONFIGS['exp1_baseline'])
runner = ExperimentRunner(config_obj)
results = runner.run_experiment(
    X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
    groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
)
```

### 4. 运行完整实验

```python
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

for exp_name in exp_names:
    config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
    runner = ExperimentRunner(config)
    results = runner.run_experiment(
        X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
        groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
    )
```

---

## 实验矩阵（5组消融实验）

| 实验编号 | 数据增强 | 偏差校正 | 模型 | 目的 |
|---------|---------|---------|------|------|
| exp1_baseline | ❌ | ❌ | CatBoost | 基线（无任何优化） |
| exp2_aug_only | ✅ | ❌ | CatBoost | M1评估（数据增强贡献） |
| exp3_corr_only | ❌ | ✅ | CatBoost | M3评估（偏差校正贡献） |
| exp4_aug_corr | ✅ | ✅ | CatBoost | M1+M3组合 |
| exp5_stacking | ✅ | ✅ | Stacking | M2评估（Stacking边际收益） |

### 基准测试结果

| 实验 | T_rmse | T_mae | T_r2 | P_rmse | P_mae | P_r2 |
|------|--------|-------|------|--------|-------|------|
| exp1_baseline | 45.69 | 33.42 | 0.881 | 2.85 | 2.02 | 0.840 |
| exp2_aug_only | 45.85 | 33.64 | 0.880 | 2.83 | 2.00 | 0.842 |
| exp3_corr_only | 45.56 | 33.35 | 0.882 | 2.83 | 1.99 | 0.841 |
| exp4_aug_corr | 45.76 | 33.60 | 0.880 | 2.82 | 1.98 | 0.843 |
| exp5_stacking | 46.25 | 34.24 | 0.878 | 2.87 | 1.95 | 0.836 |

---

## 评估指标（7个完整指标）

### 基础指标
- **RMSE** - 均方根误差
- **MAE** - 平均绝对误差
- **R²** - 决定系数

### 校正诊断指标
- **Slope** - 预测-真值回归斜率（理想值≈1）
- **Intercept** - 预测-真值回归截距（理想值≈0）
- **Bias_mean** - 偏差均值
- **Resid_std** - 残差标准差

---

## 输出格式

### metrics.csv

每折一行，包含所有指标：

| 列名 | 说明 |
|------|------|
| fold_id | 折索引 |
| T_rmse, T_mae, T_r2 | 温度基础指标 |
| T_slope, T_intercept | 温度校正诊断指标 |
| T_bias_mean, T_resid_std | 温度偏差统计 |
| P_* | 压力对应指标 |

### preds.parquet

逐样本预测表：

| 列名 | 说明 |
|------|------|
| row_id | 样本索引 |
| Ref | 文献来源 |
| fold_id | 所属验证折 |
| T_true, T_pred_raw, T_pred_corr | 温度真值、原始预测、校正预测 |
| T_residual | 温度残差 |
| P_* | 压力对应字段 |

---

## 核心模块

### models.py

| 类 | 说明 |
|---|------|
| `CatBoostWrapper` | CatBoost 回归器封装 |
| `ExtraTreesWrapper` | ExtraTrees 回归器封装 |
| `XGBoostWrapper` | XGBoost 回归器封装 |
| `GroupAwareStacker` | Group-aware OOF Stacking |

### runner.py

| 类 | 说明 |
|---|------|
| `ExperimentConfig` | 实验配置数据类 |
| `SingleTargetRunner` | 单目标运行器（T或P） |
| `ExperimentRunner` | 完整实验运行器（T+P双链路） |

### correction.py

| 类 | 说明 |
|---|------|
| `LinearBiasCorrector` | 线性偏差校正（y_corr = a×y_pred + b） |
| `IdentityCorrector` | 恒等校正（无校正） |
| `PolynomialBiasCorrector` | 多项式校正 |

### preprocessing.py

| 函数 | 说明 |
|------|------|
| `load_data()` | 加载CSV数据 |
| `prepare_data()` | 提取特征、目标、分组 |
| `augment_data()` | 数据增强（高斯噪声） |

### viz.py

| 函数 | 说明 |
|------|------|
| `plot_stepwise_rmse_comparison()` | 阶梯误差对比图 |
| `plot_correction_effect()` | 校正前后对比图 |
| `plot_feature_importance()` | 特征重要性图 |
| `plot_residual_distribution_comparison()` | 残差分布对比图 |

---

## 核心协议约束

1. **外层CV必须用GroupKFold**：`groups=df["Ref"]`，禁止随机划分
2. **所有拟合只在训练折内**：标准化、增强、stacking inner CV、偏差校正器
3. **偏差校正必须fold-safe**：校正器只能用训练折OOF拟合
4. **Stacking必须group-aware OOF**：inner CV用GroupKFold生成元特征
5. **T/P独立双链路**：温度和压力分别建模、预测、校正

### Fold-safe 校正流程

```python
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model.fit(X_train, y_train, groups=groups[train_idx])
    y_train_oof = inner_cv_predict(...)  # 训练折OOF
    corrector.fit_on_oof(y_train, y_train_oof)  # 仅用训练折拟合
    y_val_pred_corr = corrector.transform(model.predict(X_val))
```

---

## 特征集合

| 特征集合 | 列数 | 包含内容 |
|---------|------|---------|
| `cpx_oxide` | 12 | CPX氧化物（SiO2.cpx, Al2O3.cpx, ...） |
| `liq_oxide` | 12 | 液相氧化物（SiO2.liq, Al2O3.liq, ...） |
| `cpx_cation` | 12 | CPX阳离子（Si.cpx, Al.cpx, ...） |
| `cpx_only` | 24 | CPX氧化物 + CPX阳离子 |
| `cpx_liq` | 36 | 全部特征（**推荐**） |

---

## 依赖

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
catboost>=1.0
xgboost>=1.5
scipy>=1.7
matplotlib>=3.4
seaborn>=0.11
pyarrow>=6.0
joblib>=1.0
```

---

## 参考文献

Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

---

## License

MIT
