# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作提供指导。

## 项目概述

机器学习温压计（Clinopyroxene Thermobarometer）模块化评估框架。通过消融实验评估数据增强、算法选择、偏差校正各模块的独立贡献。面向 Jupyter 编排，核心目标是可审计、可复现的实验矩阵，而非刷榜。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 验证模块导入
python -c "from src import models, runner, metrics, viz; print('OK')"

# 运行完整实验（推荐在 Jupyter 中）
jupyter notebook notebooks/run_experiments.ipynb
```

### 快速验证（1折测试）

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

### 完整测试运行

```bash
python - <<'PY'
import pandas as pd
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

results_list = []
for exp_name in exp_names:
    config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
    runner = ExperimentRunner(config)
    results = runner.run_experiment(
        X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
        groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
    )
    results_list.append(results)

summary_df = pd.DataFrame(results_list)
summary_df.to_csv('outputs/summary_all.csv', index=False)
PY
```

## 架构说明

### 核心模块 (`src/`)

| 模块 | 功能 |
|------|------|
| `models.py` | 模型封装器（CatBoostWrapper、ExtraTreesWrapper、XGBoostWrapper、GroupAwareStacker） |
| `runner.py` | ExperimentRunner 编排 T/P 双链路实验，使用 GroupKFold |
| `correction.py` | 偏差校正器（LinearBiasCorrector、IdentityCorrector）- 必须 fold-safe |
| `preprocessing.py` | 数据加载（latin-1 编码）、特征提取、数据增强 |
| `metrics.py` | 7个评估指标：rmse、mae、r2、slope、intercept、bias_mean、resid_std |
| `viz.py` | 4个论文级可视化函数 |

### 数据流

```
input.csv (latin-1编码, 2079×45)
  → load_data() → prepare_data(feature_mode='cpx_liq')
  → ExperimentRunner.run_experiment()  [T/P独立双链路]
  → outputs/{exp_name}/metrics.csv + preds.parquet
```

### 实验矩阵（5组实验，定义在 `config.py`）

| 实验 | 数据增强 | 偏差校正 | 模型 | 目的 |
|------|---------|---------|------|------|
| exp1_baseline | ❌ | ❌ | CatBoost | 基线 |
| exp2_aug_only | ✅ | ❌ | CatBoost | M1评估（数据增强贡献） |
| exp3_corr_only | ❌ | ✅ | CatBoost | M3评估（偏差校正贡献） |
| exp4_aug_corr | ✅ | ✅ | CatBoost | M1+M3组合 |
| exp5_stacking | ✅ | ✅ | Stacking | M2评估（Stacking边际收益） |

### 基准测试结果（完整数据集）

| 实验 | T_rmse_mean | T_mae_mean | T_r2_mean | P_rmse_mean | P_mae_mean | P_r2_mean |
|------|-------------|------------|-----------|-------------|------------|-----------|
| exp1_baseline | 45.69 | 33.42 | 0.881 | 2.85 | 2.02 | 0.840 |
| exp2_aug_only | 45.85 | 33.64 | 0.880 | 2.83 | 2.00 | 0.842 |
| exp3_corr_only | 45.56 | 33.35 | 0.882 | 2.83 | 1.99 | 0.841 |
| exp4_aug_corr | 45.76 | 33.60 | 0.880 | 2.82 | 1.98 | 0.843 |
| exp5_stacking | 46.25 | 34.24 | 0.878 | 2.87 | 1.95 | 0.836 |

## 核心协议约束（不可违反）

1. **外层CV必须用GroupKFold**：`groups=df["Ref"]`，禁止随机划分
2. **所有拟合只在训练折内**：标准化、增强、stacking inner CV、偏差校正器
3. **偏差校正必须fold-safe**：校正器只能用训练折OOF拟合，严禁用包含验证折的全局OOF
4. **Stacking必须group-aware OOF**：inner CV 用 GroupKFold 生成元特征
5. **T/P独立双链路**：温度和压力分别建模、预测、校正（避免 MultiOutputRegressor）

### Fold-safe 校正流程示例

```python
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model.fit(X_train, y_train, groups=groups[train_idx])
    y_train_oof = inner_cv_predict(...)  # 训练折OOF
    corrector.fit_on_oof(y_train, y_train_oof)  # 仅用训练折拟合
    y_val_pred_corr = corrector.transform(model.predict(X_val))
```

## 代码风格要求

- 所有新增/修改注释用**中文**
- 函数定义尽量单行
- 不要过度抽象
- 面向 Jupyter 一键运行设计
- Python 4空格缩进；函数保持小而专注
- 函数/变量用 `snake_case`，类用 `PascalCase`
- 保持与现有 `src/` 模块风格一致

## 已实现的项目要求

- 指标键统一为 `T_rmse` / `P_rmse` 格式（runner输出、汇总、图表）
- Stacking 缓存哈希包含 `groups` 和模型签名，避免跨配置缓存复用
- Stacking 可在没有显式 `base_models` 时运行，回退到默认 stacker
- 保持模块在清理损坏的注释/文档字符串后可导入
- Stacking 默认 `n_jobs=1` 以避免 Windows 多进程权限错误

## 特征列（固定列名）

- **CPX氧化物**(12列): SiO2.cpx, Al2O3.cpx, TiO2.cpx, CaO.cpx, Na2O.cpx, K2O.cpx, FeO.cpx, MgO.cpx, MnO.cpx, Cr2O3.cpx, NiO.cpx, P2O5.cpx
- **LIQ氧化物**(12列): SiO2.liq, Al2O3.liq, ... (同上，.liq 后缀)
- **CPX阳离子**(12列): Si.cpx, Al.cpx, Ti.cpx, Ca.cpx, Na.cpx, K.cpx, Fe.cpx, Mg.cpx, Mn.cpx, Cr.cpx, Ni.cpx, P.cpx
- **推荐特征集**: `cpx_liq` (36列全部) 或 `cpx_only` (24列)

## 输出格式

- `outputs/{exp_name}/metrics.csv`: 每折一行，含 T_rmse, T_mae, T_r2, T_slope, T_intercept, T_bias_mean, T_resid_std + P同理
- `outputs/{exp_name}/preds.parquet`: 逐样本预测，含 row_id, Ref, fold_id, T_true, T_pred_raw, T_pred_corr, T_residual + P同理

## 测试指南

- 目前没有定义自动化测试套件
- 通过运行 notebook 工作流并检查 `outputs/*/metrics.csv` 和 `outputs/*/preds.parquet` 来验证更改
- 尽可能使用小范围检查（例如，`config.py` 中的单个实验配置）

## 提交与PR指南

- 提交主题简短、描述性，可用中文或英文
- PR 应包含：简要摘要、如何运行/验证（notebook 单元格或命令）、生成的新输出文件
- 避免提交大型生成输出，除非需要用于可复现性或审查

## 参考资料

- `reference_files/`: 原始R脚本、论文PDF、参考Notebook
- `input.csv`: 规范数据集；记录对其模式或编码的任何更改
