# 机器学习温压计模块化评估协议 - 详细使用手册

## 一、项目概述

本项目为辉石（Clinopyroxene）地质温压计的机器学习模型提供**模块化标准化评估框架**。通过消融实验（Ablation Study）系统性评估数据增强、算法选择、偏差校正各模块的独立贡献与性能边界。

### 1.1 核心目标

- **M1 - 数据模块**：量化数据增强/重加权对模型泛化能力的贡献
- **M2 - 算法模块**：对比 CatBoost vs 异构 Stacking 的复杂度收益
- **M3 - 校正模块**：评估 OOF 偏差校正对预测精度的提升

**核心理念**：建立可审计、可复现、可扩展的实验矩阵，而非单纯追求最优指标。

### 1.2 核心特性

- ✅ **严格的交叉验证协议**：外层使用 GroupKFold 按文献来源（Ref）分组
- ✅ **T/P 独立双链路**：温度（T）与压力（P）采用完全独立的建模流程
- ✅ **Fold-safe 设计**：所有拟合操作（标准化、增强、校正）仅在训练折内完成
- ✅ **完整评估指标**：RMSE、MAE、R²、Slope、Intercept、Bias_mean、Resid_std
- ✅ **可审计输出**：每折输出完整指标表 + 逐样本预测表（含 raw/corr 预测）
- ✅ **异构 Stacking**：ExtraTrees + XGBoost + CatBoost（Group-aware OOF）
- ✅ **论文级可视化**：4张核心图表函数

---

## 二、项目结构

```
ml-thermobarometer-benchmark/
│
├── input.csv                    # 校准数据集（2079行×45列，latin-1编码）
├── config.py                    # 全局配置（5组实验矩阵、特征集合定义）
├── requirements.txt             # Python 依赖（含 xgboost、scipy）
├── README.md                    # 快速入门指南
├── MANUAL.md                    # 本详细手册
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
│   ├── exp1_baseline/           # 基线实验（无增强、无校正）
│   ├── exp2_aug_only/           # 仅数据增强
│   ├── exp3_corr_only/          # 仅偏差校正（新增）
│   ├── exp4_aug_corr/           # 增强+校正组合
│   ├── exp5_stacking/           # 异构Stacking
│   ├── cache/                   # 模型缓存（Stacking元特征）
│   └── figures/                 # 论文图表输出
│
└── reference_files/             # 参考文件
    ├── 1 Preprocessing_cpx thermobaro.R
    ├── 2 Filtering_cpx thermobaro.R
    ├── 3 Grid Search_cpx thermobaro.R
    ├── Paper_model_stacking_regression_Bayes.ipynb
    ├── Jorgenson 等 - 2022 - ....pdf
    ├── cpx_dat.csv
    └── Supplementary Table 1.xlsx
```

---

## 三、核心模块详解

### 3.1 models.py - 模型定义

#### 类层次结构

```
BaseThermoModel (抽象基类)
├── CatBoostWrapper          # CatBoost 回归器封装
├── ExtraTreesWrapper        # ExtraTrees 回归器封装（新增）
├── XGBoostWrapper           # XGBoost 回归器封装（新增）
└── GroupAwareStacker        # Group-aware OOF Stacking
```

#### 主要类说明

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| `BaseThermoModel` | 抽象基类，定义 fit/predict 接口 | - |
| `CatBoostWrapper` | CatBoost 封装 | iterations, depth, learning_rate |
| `ExtraTreesWrapper` | ExtraTrees 封装 | n_estimators, max_depth |
| `XGBoostWrapper` | XGBoost 封装 | n_estimators, max_depth, learning_rate |
| `GroupAwareStacker` | Group-aware Stacking | base_models, meta_model, inner_cv |

#### 使用示例

```python
from src import get_model

# 方式1：使用工厂函数创建模型
catboost_model = get_model('catboost', iterations=1000, depth=6)
extratrees_model = get_model('extratrees', n_estimators=200, max_depth=10)
xgboost_model = get_model('xgboost', n_estimators=200, max_depth=6)

# 方式2：直接实例化
from src.models import CatBoostWrapper, ExtraTreesWrapper, XGBoostWrapper

model = CatBoostWrapper(iterations=1000, depth=6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 方式3：创建异构Stacking模型
from src.models import create_default_stacker

stacker = create_default_stacker(
    use_heterogeneous=True,  # 异构基学习器（ExtraTrees+XGBoost+CatBoost）
    inner_cv=5,
    cache_dir='outputs/cache'
)
```

#### Stacking 配置详解

**异构模式**（默认，推荐）：
```python
base_models = [
    ExtraTreesWrapper(n_estimators=200, max_depth=10),
    XGBoostWrapper(n_estimators=200, max_depth=6, learning_rate=0.05),
    CatBoostWrapper(iterations=200, depth=6, learning_rate=0.05)
]
meta_model = CatBoostWrapper(iterations=100, depth=4)
```

**同构模式**（可选）：
```python
base_models = [
    CatBoostWrapper(iterations=500, depth=4),
    CatBoostWrapper(iterations=800, depth=6),
    CatBoostWrapper(iterations=1000, depth=8)
]
```

---

### 3.2 runner.py - 实验运行器

#### 类结构

| 类名 | 说明 |
|------|------|
| `ExperimentConfig` | 实验配置数据类（exp_name、model_type、augment、correct） |
| `SingleTargetRunner` | 单目标（T或P）运行器（独立链路） |
| `ExperimentRunner` | 完整实验运行器（T+P双链路，协调 SingleTargetRunner） |

#### 使用示例

```python
from src import ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS

# 方式1：使用预定义配置
config = ExperimentConfig(
    exp_name='exp4_aug_corr',
    **EXPERIMENT_CONFIGS['exp4_aug_corr']
)

# 方式2：自定义配置
config = ExperimentConfig(
    exp_name='custom_exp',
    model_type='catboost',
    model_params={'iterations': 1000, 'depth': 6},
    augment=True,
    correct=True,
    n_splits=5,
    random_seed=42,
    output_dir='outputs'
)

# 运行实验
runner = ExperimentRunner(config)
results = runner.run_experiment(
    X=data['X'],
    y_T=data['y_T'],
    y_P=data['y_P'],
    groups=data['groups'],
    row_ids=data['row_ids'],
    refs=data['refs']
)

# 结果包含
# - metrics_T: T的各折指标列表
# - metrics_P: P的各折指标列表
# - preds_T: T的逐样本预测DataFrame
# - preds_P: P的逐样本预测DataFrame
```

---

### 3.3 correction.py - 偏差校正

#### 校正器类型

| 类名 | 校正方法 | 使用场景 |
|------|---------|---------|
| `LinearBiasCorrector` | y_corr = a×y_pred + b | 标准线性校正（推荐） |
| `PolynomialBiasCorrector` | 多项式校正 | 非线性偏差（可选） |
| `IdentityCorrector` | y_corr = y_pred | 无校正（基线对照） |

#### Fold-safe 原则（关键）

```python
# ✅ 正确：仅使用训练折OOF拟合
corrector = LinearBiasCorrector()
corrector.fit_on_oof(y_train, y_train_oof)  # y_train_oof由inner CV生成
y_val_corrected = corrector.transform(y_val_pred)

# ❌ 错误：使用包含验证折的全局OOF拟合
corrector.fit_on_oof(y_all, y_all_oof)  # 包含验证折数据，违反fold-safe
```

---

### 3.4 preprocessing.py - 数据预处理

#### 主要函数

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `load_data(path, encoding)` | 加载CSV数据 | pd.DataFrame |
| `prepare_data(df, feature_mode)` | 提取特征、目标、分组 | dict |
| `augment_data(X, y, groups, ...)` | 数据增强（高斯噪声） | X_aug, y_aug, groups_aug |
| `get_feature_cols(mode)` | 获取特征列名列表 | list[str] |

#### 特征集合定义

| 特征集合 | 列数 | 包含内容 |
|---------|------|---------|
| `cpx_oxide` | 12 | CPX氧化物（SiO2.cpx, Al2O3.cpx, ...） |
| `liq_oxide` | 12 | 液相氧化物（SiO2.liq, Al2O3.liq, ...） |
| `cpx_cation` | 12 | CPX阳离子（Si.cpx, Al.cpx, ...） |
| `cpx_only` | 24 | CPX氧化物 + CPX阳离子 |
| `cpx_liq` | 36 | 全部特征（推荐） |

#### 使用示例

```python
from src import load_data, prepare_data

# 加载数据
df = load_data('input.csv', encoding='latin-1')

# 准备数据（使用36列特征）
data = prepare_data(df, feature_mode='cpx_liq')

# data 包含：
# - X: 特征矩阵 (n_samples, 36)
# - y_T: 温度目标 (n_samples,)
# - y_P: 压力目标 (n_samples,)
# - groups: 分组标签（Ref列）
# - row_ids: 样本索引
# - refs: 文献来源字符串
# - feature_cols: 特征列名列表
```

---

### 3.5 metrics.py - 指标计算

#### 基础指标函数

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `rmse(y_true, y_pred)` | 均方根误差 | float |
| `mae(y_true, y_pred)` | 平均绝对误差 | float |
| `r2(y_true, y_pred)` | 决定系数 | float |

#### 扩展指标函数（新增）

| 函数 | 说明 | 返回值 |
|------|------|--------|
| `compute_slope_intercept(y_true, y_pred)` | 预测-真值回归斜率和截距 | (slope, intercept) |
| `compute_bias_stats(y_true, y_pred)` | 偏差统计量 | dict{'bias_mean', 'resid_std'} |
| `compute_metrics(y_true, y_pred, prefix)` | 完整指标集 | dict（7个指标） |

#### 完整指标集（7个指标）

```python
from src.metrics import compute_metrics

metrics = compute_metrics(y_true, y_pred, prefix='T_')

# 返回：
{
    'T_rmse': 28.5,         # 均方根误差
    'T_mae': 21.3,          # 平均绝对误差
    'T_r2': 0.89,           # 决定系数
    'T_slope': 0.98,        # 回归斜率（新增，理想值≈1）
    'T_intercept': 8.2,     # 回归截距（新增，理想值≈0）
    'T_bias_mean': -1.5,    # 偏差均值（新增）
    'T_resid_std': 27.8     # 残差标准差（新增）
}
```

**用途**：
- **slope, intercept**：量化偏差校正效果（校正后应趋近1和0）
- **bias_mean**：系统性偏差的均值
- **resid_std**：残差离散程度

---

### 3.6 viz.py - 可视化

#### 4个核心可视化函数（新增）

| 函数 | 用途 | 对应论文图表 |
|------|------|-------------|
| `plot_stepwise_rmse_comparison()` | 阶梯误差对比 | 图3-1（展示模块贡献） |
| `plot_correction_effect()` | 校正前后对比 | 图3-2（校正有效性） |
| `plot_feature_importance()` | 特征重要性 | 图3-3（可解释性） |
| `plot_residual_distribution_comparison()` | 残差分布对比 | 图3-4（Stacking收益） |

#### 使用示例

```python
from src.viz import (plot_stepwise_rmse_comparison, plot_correction_effect,
                      plot_feature_importance, plot_residual_distribution_comparison)

# 1. 阶梯误差对比图
fig1 = plot_stepwise_rmse_comparison(
    results_dict={'exp1_baseline': metrics_df1, 'exp2_aug_only': metrics_df2, ...},
    target='T',
    save_path='outputs/figures/fig1_stepwise_rmse_T.png'
)

# 2. 校正前后散点图
fig2 = plot_correction_effect(
    preds_df=preds_dict['exp4_aug_corr'],
    exp_name='exp4_aug_corr',
    target='T',
    save_path='outputs/figures/fig2_correction_effect_T.png'
)

# 3. 特征重要性图
fig3 = plot_feature_importance(
    model=trained_catboost_model,
    feature_names=data['feature_cols'],
    target='T',
    top_n=20,
    save_path='outputs/figures/fig3_feature_importance_T.png'
)

# 4. 残差分布对比图
fig4 = plot_residual_distribution_comparison(
    results_dict=preds_dict,
    exp_names=['exp4_aug_corr', 'exp5_stacking'],
    target='T',
    save_path='outputs/figures/fig4_residual_comparison_T.png'
)
```

---

## 四、实验矩阵（5组消融实验）

### 4.1 实验设计

| 实验编号 | 数据增强 | 偏差校正 | 模型 | 目的 |
|---------|---------|---------|------|------|
| **exp1_baseline** | ❌ | ❌ | CatBoost | 基线（无任何优化） |
| **exp2_aug_only** | ✅ | ❌ | CatBoost | M1评估（仅数据增强贡献） |
| **exp3_corr_only** | ❌ | ✅ | CatBoost | M3评估（仅偏差校正贡献）⭐新增 |
| **exp4_aug_corr** | ✅ | ✅ | CatBoost | M1+M3组合（标准流程主结果） |
| **exp5_stacking** | ✅ | ✅ | Stacking | M2评估（复杂集成边际收益） |

### 4.2 预期对比效果

- **Exp2 vs Exp1**：量化 M1（数据增强）的单独贡献
- **Exp3 vs Exp1**：量化 M3（偏差校正）的单独贡献
- **Exp4 vs Exp2/Exp3**：验证 M1+M3 协同效应
- **Exp5 vs Exp4**：评估 M2（Stacking）的边际收益（预期递减）

---

## 五、核心协议约束（不可违反）

### 5.1 外层交叉验证

- ✅ **必须使用** `GroupKFold`（按 `Ref` 列分组）
- ✅ **折数**：默认 N_SPLITS=5
- ❌ **禁止**：随机 KFold、StratifiedKFold 作为主结果

### 5.2 训练折隔离（Fold-safe）

以下操作**只能**在训练折内完成：
1. 标准化器（Scaler）拟合
2. 数据增强（augment_data）
3. 模型训练（model.fit）
4. Stacking 的 inner CV（生成 OOF 元特征）
5. 偏差校正器拟合（corrector.fit_on_oof）

### 5.3 偏差校正（Fold-safe）

```python
# ✅ 正确流程
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # 1. 训练模型（可能包含inner CV生成OOF）
    model.fit(X_train, y_train, groups=groups[train_idx])

    # 2. 生成训练折OOF预测（用于拟合校正器）
    y_train_oof = inner_cv_predict(model, X_train, y_train, groups[train_idx])

    # 3. 用训练折OOF拟合校正器（fold-safe！）
    corrector.fit_on_oof(y_train, y_train_oof)

    # 4. 预测验证折
    y_val_pred_raw = model.predict(X_val)
    y_val_pred_corr = corrector.transform(y_val_pred_raw)
```

❌ **禁止**：使用包含验证折的全局 OOF 拟合校正器

### 5.4 Stacking

- ✅ Inner CV 必须使用 `GroupKFold`
- ✅ 元特征 Z_train 由 inner OOF 生成
- ✅ 元特征 Z_val 由全量训练基模型预测
- ❌ 禁止使用 in-sample 预测作为元特征

### 5.5 T/P 独立链路

- 温度（T）和压力（P）采用两条完全独立的建模链路
- 分别训练、预测、校正、评估
- 避免 MultiOutputRegressor（接口复杂、审计困难）

---

## 六、输出文件格式详解

### 6.1 metrics.csv（完整指标）

**路径**：`outputs/{exp_name}/metrics.csv`

| 列名 | 说明 | 示例值 | 备注 |
|------|------|--------|------|
| fold_id | 折索引 | 0, 1, 2, 3, 4 | - |
| T_rmse | 温度RMSE | 28.5 | ℃ |
| T_mae | 温度MAE | 21.3 | ℃ |
| T_r2 | 温度R² | 0.89 | - |
| **T_slope** | 温度回归斜率 | 0.98 | ⭐新增，理想值≈1 |
| **T_intercept** | 温度回归截距 | 8.2 | ⭐新增，理想值≈0 |
| **T_bias_mean** | 温度偏差均值 | -1.5 | ⭐新增 |
| **T_resid_std** | 温度残差标准差 | 27.8 | ⭐新增 |
| P_rmse, P_mae, ... | 压力指标（同上） | ... | kbar |

### 6.2 preds.parquet（逐样本预测）

**路径**：`outputs/{exp_name}/preds.parquet`

| 列名 | 说明 | 备注 |
|------|------|------|
| row_id | 样本索引 | 原始数据行号 |
| Ref | 文献来源 | 分组标签 |
| fold_id | 折索引 | 该样本所属验证折 |
| exp_name | 实验名称 | 如 'exp4_aug_corr' |
| T_true | 温度真值 | ℃ |
| **T_pred_raw** | 温度原始预测（校正前） | ⭐ |
| **T_pred_corr** | 温度校正预测（校正后） | ⭐ |
| **T_residual** | 温度残差（true - corr） | ⭐新增 |
| P_true, P_pred_raw, ... | 压力字段（同上） | kbar |

**用途**：
- 逐样本分析偏差模式
- 绘制校正前后对比散点图
- 生成残差分布图

---

## 七、完整工作流程示例

### 7.1 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 验证安装
python -c "from src import models, runner, metrics, viz; print('✅ 模块导入成功')"
```

### 7.2 快速验证（1折测试）

```python
import sys
sys.path.insert(0, '.')

from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS

# 临时修改折数为1（快速验证）
import config
config.N_SPLITS = 1

# 加载数据
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

# 运行基线实验
config_obj = ExperimentConfig(exp_name='test_exp1', **EXPERIMENT_CONFIGS['exp1_baseline'])
runner = ExperimentRunner(config_obj)
results = runner.run_experiment(
    X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
    groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
)

# 检查输出
import os
assert os.path.exists('outputs/test_exp1/metrics.csv'), "❌ metrics.csv未生成"
assert os.path.exists('outputs/test_exp1/preds.parquet'), "❌ preds.parquet未生成"
print("✅ 快速验证通过")
```

### 7.3 完整实验运行（Jupyter推荐）

打开 `notebooks/run_experiments.ipynb`，按顺序执行：

```python
# Cell 1: 加载数据
from src import load_data, prepare_data
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

# Cell 2: 运行5组实验
from config import EXPERIMENT_CONFIGS
from src import ExperimentConfig, ExperimentRunner

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only',
             'exp4_aug_corr', 'exp5_stacking']

results_dict = {}
for exp_name in exp_names:
    print(f"\n{'='*60}\n运行 {exp_name}...\n{'='*60}")
    config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
    runner = ExperimentRunner(config)
    results_dict[exp_name] = runner.run_experiment(
        X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
        groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
    )

# Cell 3: 加载结果
import pandas as pd
metrics_dict = {}
preds_dict = {}
for exp_name in exp_names:
    metrics_dict[exp_name] = pd.read_csv(f'outputs/{exp_name}/metrics.csv')
    preds_dict[exp_name] = pd.read_parquet(f'outputs/{exp_name}/preds.parquet')

# Cell 4: 生成可视化
from src.viz import (plot_stepwise_rmse_comparison, plot_correction_effect,
                      plot_feature_importance, plot_residual_distribution_comparison)
import os
os.makedirs('outputs/figures', exist_ok=True)

fig1 = plot_stepwise_rmse_comparison(metrics_dict, target='T',
                                     save_path='outputs/figures/fig1_stepwise_rmse_T.png')
fig2 = plot_correction_effect(preds_dict['exp4_aug_corr'], exp_name='exp4_aug_corr', target='T',
                              save_path='outputs/figures/fig2_correction_effect_T.png')
# ... (fig3, fig4 同理)
```

---

## 八、自定义扩展指南

### 8.1 添加新的模型

```python
# src/models.py 中添加新模型
from sklearn.ensemble import RandomForestRegressor

class RandomForestWrapper(BaseThermoModel):
    def __init__(self, n_estimators=200, max_depth=10, random_state=42, **kwargs):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None

    def fit(self, X, y, groups=None):
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1
        )
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)

# 更新 get_model() 函数
def get_model(name, **kwargs):
    if name == 'randomforest':
        return RandomForestWrapper(**kwargs)
    # ... 其他模型
```

### 8.2 添加新的评估指标

```python
# src/metrics.py 中添加新指标
def compute_mape(y_true, y_pred, epsilon=1e-8):
    """平均绝对百分比误差"""
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

# 在 compute_metrics() 中集成
def compute_metrics(y_true, y_pred, prefix=''):
    # ... 现有指标
    return {
        # ... 现有7个指标
        f'{prefix}mape': compute_mape(y_true, y_pred)  # 新增
    }
```

---

## 九、常见问题（FAQ）

### Q1: 为什么需要5组实验而不是4组？

**A**: 新增的 `exp3_corr_only`（仅校正无增强）用于独立评估偏差校正的贡献。通过对比：
- Exp2 vs Exp1 → M1（数据增强）单独贡献
- **Exp3 vs Exp1 → M3（偏差校正）单独贡献**
- Exp4 vs Exp2/Exp3 → M1+M3 协同效应

这样可以更清晰地拆解各模块的独立贡献与边界。

### Q2: slope 和 intercept 指标的意义是什么？

**A**: 用于量化偏差校正的有效性。通过对 y_true 和 y_pred 进行线性回归：
- **slope ≈ 1**：预测值与真实值呈完美线性关系
- **intercept ≈ 0**：无系统性偏差

校正前 slope 可能偏离1（如0.95），校正后应趋近1（如0.99），证明校正有效。

### Q3: 为什么 Stacking 使用异构基学习器？

**A**: 同构基学习器（3个不同参数的CatBoost）同质性过强，可能被质疑。异构基学习器（ExtraTrees + XGBoost + CatBoost）具有不同的建模假设：
- **ExtraTrees**: 高方差、强随机化
- **XGBoost**: 经典GBDT、擅长特征交互
- **CatBoost**: 对类别特征友好、稳定

多样性更强，更符合Stacking理念。

### Q4: 如何修改实验配置？

**A**: 在 `config.py` 中修改 `EXPERIMENT_CONFIGS`：

```python
EXPERIMENT_CONFIGS = {
    'exp1_baseline': {
        'model_type': 'catboost',
        'model_params': {
            'iterations': 1500,  # 修改迭代次数
            'depth': 8,          # 修改树深度
        },
        'augment': False,
        'correct': False,
    },
    # ...
}
```

### Q5: 如何处理运行时间过长的问题？

**A**: 优化策略：
1. **减少外层折数**：`N_SPLITS = 3`（临时验证）
2. **减少模型迭代**：CatBoost `iterations=500`
3. **使用缓存**：Stacking 元特征会自动缓存到 `outputs/cache/`
4. **跳过 exp5_stacking**：仅运行 exp1~exp4（exp5 训练时间最长）

---

## 十、依赖清单

```txt
# 核心依赖
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
catboost>=1.0
xgboost>=1.5        # 新增（异构Stacking）
scipy>=1.7          # 新增（KDE、统计分析）

# 可视化
matplotlib>=3.4
seaborn>=0.11

# 数据IO
pyarrow>=6.0        # parquet格式
joblib>=1.0         # 模型缓存
```

---

## 十一、参考文献

1. Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

2. 原始 R 脚本：`reference_files/1-3 *.R`（数据预处理、筛选、网格搜索）

3. 参考 Notebook：`reference_files/Paper_model_stacking_regression_Bayes.ipynb`

---

*文档版本：v2.0 | 更新日期：2026-01-17 | 包含最新修改（5组实验、异构Stacking、扩展指标、4个可视化函数）*
