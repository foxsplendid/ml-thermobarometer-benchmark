# 机器学习温压计标准化评估协议 - 项目说明书

## 一、项目概述

本项目为辉石（Clinopyroxene）地质温压计的机器学习模型标准化评估提供可复用的 Python 工程骨架。采用 GroupKFold 交叉验证确保评估结果的可靠性，支持温度（T）与压力（P）的独立建模链路。

### 核心特性

- **严格的交叉验证协议**：外层使用 GroupKFold 按文献来源分组
- **T/P 独立双链路**：温度与压力采用完全独立的建模流程
- **Fold-safe 设计**：所有拟合操作仅在训练折内完成
- **可审计输出**：每折输出指标表和逐样本预测表
- **Group-aware Stacking**：内层 CV 同样支持分组

---

## 二、项目结构

```
ml-thermobarometer-benchmark/
│
├── input.csv                    # 校准数据集（主数据）
├── config.py                    # 全局配置文件
├── requirements.txt             # Python 依赖清单
├── README.md                    # 快速入门指南
├── MANUAL.md                    # 本说明书
│
├── src/                         # 核心代码模块
│   ├── __init__.py              # 模块导出
│   ├── models.py                # 模型定义（CatBoost、Stacking）
│   ├── runner.py                # 实验运行器
│   ├── correction.py            # 偏差校正器
│   ├── preprocessing.py         # 数据预处理
│   ├── metrics.py               # 指标计算
│   └── viz.py                   # 可视化函数
│
├── notebooks/                   # Jupyter 入口
│   └── run_experiments.ipynb    # 一键运行实验
│
├── outputs/                     # 实验输出目录
│   └── {exp_name}/
│       ├── metrics.csv          # 各折指标
│       ├── preds.parquet        # 逐样本预测
│       └── summary.csv          # 实验汇总
│
├── reference_files/             # 参考文件（原始脚本、论文）
│   ├── 1 Preprocessing_cpx thermobaro.R
│   ├── 2 Filtering_cpx thermobaro.R
│   ├── 3 Grid Search_cpx thermobaro.R
│   ├── Paper_model_stacking_regression_Bayes.ipynb
│   ├── Jorgenson 等 - 2022 - ....pdf
│   ├── cpx_dat.csv
│   └── Supplementary Table 1.xlsx
│
└── temp/                        # 临时文件
    ├── tmp_inspect.py
    ├── tmp_result.txt
    └── columns_info.txt
```

---

## 三、模块说明

### 3.1 models.py - 模型定义

| 类名 | 说明 |
|------|------|
| `BaseThermoModel` | 抽象基类，定义 fit/predict 接口 |
| `CatBoostWrapper` | CatBoost 回归器封装 |
| `GroupAwareStacker` | Group-aware OOF Stacking 模型 |
| `get_model(name)` | 模型工厂函数 |

**使用示例**：
```python
from src import get_model

# 创建 CatBoost 模型
model = get_model('catboost', iterations=1000, depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 3.2 runner.py - 实验运行器

| 类名 | 说明 |
|------|------|
| `ExperimentConfig` | 实验配置数据类 |
| `SingleTargetRunner` | 单目标（T或P）运行器 |
| `ExperimentRunner` | 完整实验运行器（T+P双链路） |

**使用示例**：
```python
from src import ExperimentConfig, ExperimentRunner

config = ExperimentConfig(
    exp_name='exp1_catboost_base',
    model_type='catboost',
    augment=False,
    correct=False
)
runner = ExperimentRunner(config)
results = runner.run_experiment(X, y_T, y_P, groups, row_ids, refs)
```

### 3.3 correction.py - 偏差校正

| 类名 | 说明 |
|------|------|
| `LinearBiasCorrector` | 线性偏差校正：y_corr = a×y_pred + b |
| `IdentityCorrector` | 恒等校正（不校正） |

**Fold-safe 原则**：校正器仅使用训练折的 OOF 预测拟合。

### 3.4 preprocessing.py - 预处理

| 函数 | 说明 |
|------|------|
| `load_data(path)` | 加载 CSV 数据 |
| `prepare_data(df)` | 提取特征、目标、分组 |
| `augment_data(X, y, groups)` | 数据增强 |
| `get_feature_cols(mode)` | 获取特征列名 |

**特征集合**：
- `cpx_oxide`: 12 列 CPX 氧化物
- `liq_oxide`: 12 列液相氧化物
- `cpx_cation`: 12 列 CPX 阳离子
- `cpx_liq`: 36 列（推荐，全部特征）

### 3.5 metrics.py - 指标计算

| 函数 | 说明 |
|------|------|
| `rmse(y_true, y_pred)` | 均方根误差 |
| `mae(y_true, y_pred)` | 平均绝对误差 |
| `r2(y_true, y_pred)` | 决定系数 |
| `summarize_folds(metrics_list)` | 汇总各折指标 |

### 3.6 viz.py - 可视化

| 函数 | 说明 |
|------|------|
| `plot_pred_vs_true()` | 预测-真实散点图 |
| `plot_residuals()` | 残差分布图 |
| `plot_full_report()` | 完整报告图（T+P） |

---

## 四、实验矩阵

| 实验名 | 模型 | 数据增强 | 偏差校正 |
|--------|------|----------|----------|
| exp1_catboost_base | CatBoost | ❌ | ❌ |
| exp2_catboost_aug | CatBoost | ✅ | ❌ |
| exp3_catboost_aug_corr | CatBoost | ✅ | ✅ |
| exp4_stacking_aug_corr | Stacking | ✅ | ✅ |

---

## 五、协议约束（必须遵守）

### 5.1 外层交叉验证
- **必须**使用 `GroupKFold`
- **分组列**：`Ref`（文献来源）
- **禁止**随机划分作为主结果

### 5.2 训练折隔离
以下操作**只能**在训练折内完成：
- 标准化器拟合
- 数据增强
- 模型训练
- 偏差校正器拟合

### 5.3 偏差校正（Fold-safe）
- 校正器仅用**训练折 OOF 预测**拟合
- **禁止**使用包含验证折的全局 OOF

### 5.4 Stacking
- 内层 CV 必须使用 `GroupKFold`
- 元特征由 inner OOF 生成
- Z_val 由全量训练数据训练的基模型预测

---

## 六、输出文件格式

### 6.1 metrics.csv

| 列名 | 说明 |
|------|------|
| fold_id | 折索引 (0-4) |
| rmse_T / mae_T / r2_T | 温度指标 |
| rmse_P / mae_P / r2_P | 压力指标 |

### 6.2 preds.parquet

| 列名 | 说明 |
|------|------|
| row_id | 样本索引 |
| Ref | 文献来源 |
| T_true | 温度真值 (℃) |
| T_pred_raw | 温度原始预测 |
| T_pred_corr | 温度校正预测 |
| P_true | 压力真值 (kbar) |
| P_pred_raw | 压力原始预测 |
| P_pred_corr | 压力校正预测 |
| fold_id | 折索引 |
| exp_name | 实验名称 |

---

## 七、快速入门

### 7.1 安装依赖
```bash
pip install -r requirements.txt
```

### 7.2 一键运行（推荐）
1. 打开 `notebooks/run_experiments.ipynb`
2. 依次执行所有单元格

### 7.3 命令行运行
```python
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner

# 加载数据
df = load_data('input.csv')
data = prepare_data(df)

# 运行实验
config = ExperimentConfig(exp_name='test', model_type='catboost')
runner = ExperimentRunner(config)
results = runner.run_experiment(
    data['X'], data['y_T'], data['y_P'],
    data['groups'], data['row_ids'], data['refs']
)
```

---

## 八、依赖清单

```
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
catboost>=1.0
matplotlib>=3.4
seaborn>=0.11
pyarrow>=6.0
joblib>=1.0
```

---

## 九、参考文献

- Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry...

---

*文档版本：v1.0 | 更新日期：2026-01-17*
