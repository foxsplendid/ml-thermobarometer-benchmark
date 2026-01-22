# ML Thermobarometer Benchmark Protocol

机器学习温压计模块化评估协议框架 - 用于 Chapter 3 标准化效能分析

## 核心特性

- ✅ **模块化消融实验**：M1(数据) × M2(算法) × M3(校正) × 特征集 全因子设计
- ✅ **严格防泄露**：所有拟合操作仅在训练折内完成
- ✅ **双特征集对比**：NoLiquid(9特征) vs Liquid(18特征)
- ✅ **P-T分层采样**：基于P-T网格的分层交叉验证，优先保证P-T分布平衡
- ✅ **T/P 独立建模**：温度与压力采用完全独立的建模链路
- ✅ **Strict OOF Stacking**：严格 OOF 元特征生成，无数据泄露
- ✅ **工具分离**：主实验与绘图/稳定性测试/不确定性量化分离，按需运行

---

## 项目结构

```
ml-thermobarometer-benchmark/
│
├── input.csv                    # 校准数据集（2079行×45列，latin-1编码）
├── main.py                      # 主入口（一键运行实验矩阵）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
│
├── src/                         # 核心代码模块
│   ├── __init__.py              # 模块导出
│   ├── interfaces.py            # M1-M4 统一接口定义
│   ├── data_modules.py          # M1: Raw/Balanced/Augmented
│   ├── model_modules.py         # M2: ERT/CatBoost/Stacking
│   ├── correction_modules.py    # M3: None/Segmented
│   ├── uncertainty_modules.py   # M4: MCUncertaintyEstimator
│   ├── protocol.py              # Pipeline + StratifiedKFold 协议
│   ├── splitters.py             # P-T网格采样与分层工具
│   ├── metrics.py               # 指标计算
│   └── viz.py                   # 可视化
│
├── tools/                       # 工具脚本（独立于main.py运行）
│   ├── run_stability_mc.py      # 稳定性测试 + MC不确定性量化
│   └── plot_offline_figures.py  # 离线绘图 + 特征重要性
│
├── results/                     # 实验输出目录
│   ├── metrics_summary.csv      # 汇总指标（mean ± CI）
│   ├── effect_table.csv         # 模块效应分析表
│   ├── models/                  # 保存的模型文件（.joblib）
│   └── figures/                 # 可视化图表
│
├── reference_files/             # 参考文件
│
└── old/                         # 旧版代码（已归档）
```

---

## 架构设计

### 整体工作流

```
main.py (主入口，仅运行实验)
    │
    ├── CONFIG 初始化
    │   └── 数据路径、特征集、CV参数、输出目录
    │
    ├── prepare_splits()
    │   └── P-T网格采样 → 固定测试集（用于工具脚本）
    │
    └── ExperimentMatrix.run_experiments()
        └── 24个实验 = 12基础配置 × 2特征集
            │
            └── StratifiedCVProtocol (10折CV)
                └── 每折: DataModule → ModelModule → CorrectionModule → 评估
                └── 保存模型到 results/models/*.joblib

tools/ (独立工具，按需运行)
    ├── plot_offline_figures.py  → 离线绘图 + 特征重要性
    └── run_stability_mc.py      → 稳定性测试 + MC不确定性
```

### 模块依赖关系

| 模块 | 依赖 | 说明 |
|------|------|------|
| `interfaces.py` | - | 抽象基类，无外部依赖 |
| `data_modules.py` | interfaces | M1 数据预处理实现 |
| `model_modules.py` | interfaces | M2 模型算法实现 |
| `correction_modules.py` | interfaces | M3 校正策略实现 |
| `uncertainty_modules.py` | interfaces | M4 不确定性量化 |
| `protocol.py` | 上述所有 | Pipeline组装与CV执行 |
| `metrics.py` | - | 纯函数指标计算 |
| `splitters.py` | - | P-T网格划分工具 |
| `viz.py` | - | 可视化绑定工具脚本使用 |

### 每折执行流程

1. `DataModule.fit_transform()` - 训练折内拟合标准化器/权重
2. `ModelModule.fit()` - 训练模型
3. `ModelModule.predict()` - 生成OOF预测
4. `CorrectionModule.fit()` - 在全训练集OOF上拟合校正器
5. 指标计算 (RMSE/MAE/R²/Slope等)
6. 模型保存至 `results/models/` (支持离线绘图/特征重要性)

### 工具脚本说明

| 工具 | 功能 | 典型用法 |
|------|------|----------|
| `plot_offline_figures.py` | 离线绘图、特征重要性 | `python tools/plot_offline_figures.py --exp-id E07_ert_augmented_none_liq` |
| `run_stability_mc.py` | 稳定性测试、MC不确定性 | `python tools/run_stability_mc.py --model-module ert --n-repeats 100` |

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 验证安装

```python
from src import (
    RawDataModule, BalancedDataModule, AugmentedDataModule,
    ExtraTreesModel, CatBoostModel, StrictOOFStacking,
    NoCorrection, ResidualRegressionCorrector, SegmentedLinearCorrector,
    Pipeline, StratifiedCVProtocol
)
print('OK')
```

### 3. 快速测试（2折 × 4实验）

```bash
python main.py --test
```

测试将运行前2个基础配置 × 2种特征集 = 4个实验

### 4. 运行完整实验矩阵（10折 × 24实验）

```bash
python main.py
```

完整运行包含12个基础配置 × 2种特征集 = 24个实验

---

## 特征集配置

### NoLiquid（无液相，9个特征）

仅使用单斜辉石（CPX）主要氧化物成分：

```python
['SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx', 'Cr2O3.cpx',
 'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'CaO.cpx', 'Na2O.cpx']
```

**适用场景**：仅有矿物成分数据，无共存熔体成分

### Liquid（有液相，18个特征）

CPX氧化物（9个）+ 共存熔体（LIQ）氧化物（9个）：

```python
# CPX部分（9个）
['SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx', 'Cr2O3.cpx',
 'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'CaO.cpx', 'Na2O.cpx',
 # LIQ部分（9个）
 'SiO2.liq', 'TiO2.liq', 'Al2O3.liq', 'FeO.liq',
 'MgO.liq', 'MnO.liq', 'CaO.liq', 'Na2O.liq', 'K2O.liq']
```

**适用场景**：同时具有矿物和熔体成分数据（推荐，精度更高）

**注意**：阳离子列（Si.cpx, Al.cpx等）、微量氧化物（K2O.cpx, NiO.cpx等）以及诊断列（cat.sum, Rm, kd）仅用于数据质量控制，不作为模型输入特征。

---

## 模块体系

### M1 数据模块 (DataModule)

| 模块 | 说明 | 设计要点 |
|------|------|----------|
| `RawDataModule` | 仅标准化 | 基线对照，权重=1 |
| `BalancedDataModule` | 分箱重加权 | 10 bins, quantile策略，逆频率权重 |
| `AugmentedDataModule` | EPMA 误差模型增强 | >1 wt% 3%误差, ≤1 wt% 8%误差，n=15 |

**设计亮点**：
- `fit_transform()` 返回 `DataModuleState` 封装拟合状态，实现训练/验证解耦
- 增强操作仅在训练折内执行，防止数据泄露
- EPMA 误差模型与地球化学电子探针分析实践对齐

### M2 模型模块 (ModelModule)

| 模块 | 说明 | 核心特点 |
|------|------|----------|
| `ExtraTreesModel` | 基线模型（高方差低偏差） | Bagging 集成，特征随机化 |
| `CatBoostModel` | 强单模型（Boosting 代表） | 梯度提升，内置正则化 |
| `StrictOOFStacking` | 严格 OOF 堆叠集成 | ERT+CatBoost+RF → Ridge |

#### 模型超参数配置

所有超参数采用**固定配置**，与已发表文献保持一致，不进行超参数调优。

##### ExtraTrees (ERT)

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_estimators` | 200 | 树的数量，Jorgenson et al. (2022) 推荐值 |
| `max_depth` | 15 | 最大深度，Jorgenson et al. (2022) 推荐值 |
| `min_samples_split` | 5 | 分裂最小样本数，使用 sklearn 默认 |
| `max_features` | - | 使用 sklearn 默认 (sqrt) |
| `random_state` | 42 | 随机种子，确保可复现 |

##### CatBoost

| 参数 | 值 | 说明 |
|------|-----|------|
| `iterations` | 1000 | 迭代次数 |
| `depth` | 6 | 树深度 |
| `learning_rate` | 0.03 | 学习率 |
| `loss_function` | 'RMSE' | 损失函数 |
| `random_seed` | 42 | 随机种子 |

##### RandomForest (RF) - Stacking 基模型

| 参数 | 值 | 说明 |
|------|-----|------|
| `n_estimators` | 200 | 与 ERT 一致 |
| `max_depth` | 15 | 与 ERT 一致 |
| `min_samples_split` | 5 | 与 ERT 一致 |
| `random_state` | 42 | 随机种子 |

##### Stacking 架构

| 组件 | 配置 | 说明 |
|------|------|------|
| **基模型** | ERT + CatBoost + RF | 3 个基模型，参数与单模型实验一致 |
| **元模型** | Ridge(alpha=1.0) | 线性元模型，防止过拟合 |
| **内层 CV** | 5 折 StratifiedKFold | 生成 OOF 元特征 |
| **元特征标准化** | StandardScaler | 标准化后输入元模型 |

> **设计说明**：Stacking 基模型参数与单模型实验**完全一致**，确保公平对比。

**设计亮点**：
- `_get_default_n_jobs()` 平台自适应（Windows: 1, Linux/Mac: -1）
- `_detect_catboost_gpu()` 自动检测GPU可用性
- Stacking内层使用独立StratifiedKFold生成OOF元特征，无数据泄露

### M3 校正模块 (CorrectionModule)

| 模块 | 说明 | 适用场景 |
|------|------|----------|
| `NoCorrection` | 无校正 | 默认配置 |
| `ResidualRegressionCorrector` | 残差回归校正（Ridge） | 系统性偏差修正（可选，保留接口） |
| `SegmentedLinearCorrector` | 分段线性校正 | 端元效应修正，训练集min-max clip |

**注意**：根据地质学ML传统实践，简单模型（ERT、CatBoost）的校正器使用in-sample预测，StrictOOFStacking使用严格OOF预测。

### M4 不确定性模块 (UncertaintyModule)

| 模块 | 说明 | 参数 |
|------|------|------|
| `MCUncertaintyEstimator` | MC 输入扰动 | n_mc=1000, EPMA误差模型 |

**设计亮点**：
- 使用EPMA误差模型生成输入扰动（与M1对齐）
- 输出多分位数（p5/p16/p50/p84/p95）
- 计算校准指标：PICP_68/PICP_90

---

### 2.1`protocol.py` - 协议执行器

| 类 | 职责 | 核心逻辑 |
|----|------|----------|
| `Pipeline` | 封装 DM+MM+CM 流程 | `fit()`, `predict()`, `set_correction()` |
| `StratifiedCVProtocol` | 主协议（P-T 分层 KFold） | 外层 CV + 全局校正器拟合 |
| `RandomSplitProtocol` | 对照协议 | 随机划分，评估乐观偏差 |
| `ExperimentConfig` | 实验配置 | dataclass 封装 |
| `ExperimentMatrix` | 批量运行 | 24 实验 × T/P 双目标 |

**辅助函数**:
- `_call_pipeline_factory(factory, seed)` - 使用 `inspect.signature()` 安全调用工厂函数
- `summarize_folds()` - 从 `metrics.py` 导入，计算折叠汇总和置信区间

**新增功能**:
- **模型保存**: 每个实验自动保存模型到 `results/models/{exp_id}_{target}_model.joblib`
- **分批保存**: `run_stability_repeats()` 支持 `checkpoint_interval` 参数，每 100 次自动保存 checkpoint

---

### 2.2 `splitters.py` - 数据划分

| 函数 | 功能 |
|------|------|
| `compute_pt_edges()` | 计算 P-T 网格边界（k = ceil(sqrt(n))） |
| `assign_pt_bins()` | 分配样本到 P-T 格子 |
| `select_test_indices()` | 每个非空格子随机选 1 个样本 |

---

### 2.3 `metrics.py` - 指标计算

| 函数 | 功能 |
|------|------|
| `rmse`, `mae`, `r2`, `mape`, `bias` | 基础指标 |
| `compute_slope_intercept` | 回归诊断 |
| `summarize_folds` | 折叠汇总 + CI 计算 |

---

### 2.4 `viz.py` - 可视化

| 函数 | 功能 |
|------|------|
| `plot_pred_vs_true` | 预测-真实散点图 |
| `plot_residuals` | 残差分布图 |
| `plot_fold_comparison` | 各折指标对比 |
| `plot_experiment_summary` | 实验汇总热力图 |
| `plot_stepwise_rmse_comparison` | 阶梯误差对比（论文图） |
| `plot_correction_effect` | 校正前后对比 |
| `plot_feature_importance` | 特征重要性图 |

---

## 实验矩阵（24组实验：12基础配置 × 2特征集）

### 基础配置（12组）

| Exp ID | M1 数据 | M2 模型 | M3 校正 | 目的             |
|-----|---------|---------|---------|----------------|
| E01 | Raw | ERT | None | 基线             |
| E02 | Raw | CatBoost | None | Boost基线        |
| E03 | Raw | Stacking | None | Stacking基线     |
| E04 | Balanced | ERT | None | M1效应(ERT)      |
| E05 | Balanced | CatBoost | None | M1效应(Boost)    |
| E06 | Balanced | Stacking | None | M1效应(Stacking) |
| E07 | Augmented | ERT | None | M1-Aug效应       |
| E08 | Augmented | CatBoost | None | M1-Aug效应       |
| E09 | Raw | CatBoost | Segmented | M3效应           |
| E10 | Balanced | ERT | Segmented | 完整流程(ERT)      |
| E11 | Balanced | CatBoost | Segmented | 理论配置           |
| E12 | Balanced | Stacking | Segmented | 边界探索           |

### 特征集扩展

每个基础配置生成两个实验：
- `E01_ert_raw_none_noliq` - NoLiquid特征集（9特征）
- `E01_ert_raw_none_liq` - Liquid特征集（18特征）

总计 **24个实验**，可直接对比特征集对模型性能的影响。

---

## 数据划分策略

### 测试集划分（P-T网格采样）

1. 基于文献公式计算分箱数：`k = ceil(sqrt(n))`
2. 构建P-T二维网格：
   - P bins: `seq(P_min-0.1, P_max+0.1, length=k)`，精度0.1 kbar
   - T bins: `seq(T_min-1, T_max+1, length=k)`，精度1°C
3. 从每个非空网格格子随机选择1个样本作为测试集
4. **测试集固定后，所有实验共用相同的train/test划分**

### 交叉验证策略（StratifiedKFold）

- **使用P-T bins作为分层标签**，确保每折的P-T分布平衡
- **不再使用Ref（文献来源）分组约束**
- 权衡决策：优先保证P-T分布平衡 > 文献分组完整性
- 数据泄露风险：同一文献的样本可能出现在训练集和验证集（已知限制）

---

## 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **基础** | RMSE, MAE, R2 | 精度指标 |
| **偏差** | MBE, Slope, Intercept | 系统性偏差诊断 |
| **分箱** | MAE_bin, MBE_bin | 端元效应诊断 |
| **不确定性** | PICP, Interval Width | 校准指标 |

---

## 输出文件

```
results/
├── metrics_summary.csv          # 汇总指标（mean ± CI，含T_test_*/P_test_*）
├── effect_table.csv             # 模块效应分析表
├── config_used.yaml             # 实验配置与测试集划分信息
├── {exp_id}_{T/P}_fold_metrics.csv    # 每折指标
├── {exp_id}_{T/P}_predictions.parquet # 逐样本预测（含raw/corr/残差/MC分位数）
├── models/                      # 保存的模型文件
│   └── {exp_id}_{target}_model.joblib
└── figures/                     # 可视化图表
```

---

## 核心协议约束

1. **外层 CV 使用 StratifiedKFold**：按P-T bins分层，保证各折P-T分布平衡
2. **所有拟合只在训练折内**：标准化器、权重、增强、校正器
3. **Stacking 必须 Strict OOF**：内层 StratifiedKFold 生成元特征，禁止泄露
4. **T/P 独立双链路**：温度和压力分别建模、预测、校正
5. **测试集固定策略**：基于P-T网格采样一次，所有实验共用
6. **特征集显式标识**：实验ID明确标注特征集（_noliq / _liq）
---

## 🔬 实验结果（V4.2，24组实验 10折交叉验证）

> 运行日期：2026-01-22 | 数据集：2079样本 | CV策略：P-T分层10折

### 最佳配置

| 目标 | 最佳实验 | RMSE | R² | 配置 |
|------|----------|------|----|------|
| **温度 T** | E07_ert_augmented_none_liq | **30.69 °C** | **0.934** | ERT + 数据增强 + Liquid |
| **压力 P** | E07_ert_augmented_none_liq | **1.91 kbar** | **0.920** | ERT + 数据增强 + Liquid |

### 全量实验结果（按 T_RMSE 排序）

| 实验ID | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|--------|-------------|------|---------------|------|
| **E07_ert_augmented_none_liq** ⭐ | **30.69** | **0.934** | **1.91** | **0.920** |
| E08_catboost_augmented_none_liq | 31.17 | 0.932 | 1.92 | 0.917 |
| E10_ert_balanced_segmented_liq | 31.82 | 0.929 | 1.99 | 0.911 |
| E01_ert_raw_none_liq | 32.16 | 0.927 | 2.03 | 0.909 |
| E03_stacking_raw_none_liq | 32.43 | 0.926 | 2.03 | 0.908 |
| E04_ert_balanced_none_liq | 32.52 | 0.926 | 2.06 | 0.906 |
| E12_stacking_balanced_segmented_liq | 32.69 | 0.925 | 2.04 | 0.908 |
| E06_stacking_balanced_none_liq | 32.83 | 0.924 | 2.07 | 0.905 |
| E09_catboost_raw_segmented_liq | 36.86 | 0.905 | 2.23 | 0.890 |
| E11_catboost_balanced_segmented_liq | 36.92 | 0.905 | 2.25 | 0.888 |
| E02_catboost_raw_none_liq | 36.96 | 0.904 | 2.26 | 0.887 |
| E05_catboost_balanced_none_liq | 37.00 | 0.904 | 2.30 | 0.883 |
| **E07_ert_augmented_none_noliq** | **52.42** | **0.809** | **2.31** | **0.882** |
| E10_ert_balanced_segmented_noliq | 53.67 | 0.799 | 2.39 | 0.874 |
| E04_ert_balanced_none_noliq | 54.35 | 0.795 | 2.43 | 0.870 |
| E01_ert_raw_none_noliq | 54.40 | 0.794 | 2.41 | 0.872 |
| E08_catboost_augmented_none_noliq | 54.82 | 0.791 | 2.37 | 0.877 |
| E03_stacking_raw_none_noliq | 54.85 | 0.791 | 2.46 | 0.867 |
| E12_stacking_balanced_segmented_noliq | 55.14 | 0.788 | 2.48 | 0.864 |
| E06_stacking_balanced_none_noliq | 55.35 | 0.787 | 2.50 | 0.862 |
| E09_catboost_raw_segmented_noliq | 59.84 | 0.750 | 2.64 | 0.846 |
| E02_catboost_raw_none_noliq | 60.04 | 0.749 | 2.65 | 0.844 |
| E11_catboost_balanced_segmented_noliq | 60.66 | 0.743 | 2.65 | 0.845 |
| E05_catboost_balanced_none_noliq | 60.88 | 0.742 | 2.68 | 0.842 |

### 模块效应分析

#### 特征集效应（最显著，↓40% RMSE）

| 特征集 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² | 改善幅度 |
|--------|-------------|------|---------------|------|----------|
| NoLiquid (9特征) | 56.37 ± 3.05 | 0.778 | 2.50 ± 0.13 | 0.862 | 基线 |
| **Liquid (18特征)** | **33.67 ± 2.49** | **0.920** | **2.09 ± 0.13** | **0.903** | T: **↓40%**, P: ↓16% |

> ⭐ **关键发现**：加入熔体成分使温度预测RMSE降低约40%（56.4→33.7°C），R²从0.78升至0.92

#### 模型效应（Liquid 特征集）

| 模型 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|------|-------------|------|---------------|------|
| **ERT** | **31.80 ± 0.79** | **0.929** | **2.00 ± 0.07** | **0.912** |
| CatBoost | 35.78 ± 2.58 | 0.910 | 2.19 ± 0.15 | 0.893 |
| Stacking | 32.65 ± 0.20 | 0.925 | 2.05 ± 0.02 | 0.907 |

> ERT 表现最优且方差最小（最稳定），CatBoost 在 Augmented 配置下接近 ERT

#### 数据模块效应（Liquid 特征集）

| 数据处理 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|----------|-------------|------|---------------|------|
| Raw | 34.61 ± 2.67 | 0.916 | 2.14 ± 0.12 | 0.899 |
| Balanced | 33.96 ± 2.35 | 0.919 | 2.12 ± 0.12 | 0.900 |
| **Augmented** | **30.93 ± 0.35** | **0.933** | **1.92 ± 0.01** | **0.919** |

> 数据增强带来稳定提升：T_RMSE ↓3-4°C，P_RMSE ↓0.2 kbar

#### 校正模块效应（Liquid 特征集）

| 校正方式 | T_RMSE (°C) | P_RMSE (kbar) |
|----------|-------------|---------------|
| **None** | **33.22 ± 2.43** | **2.07 ± 0.14** |
| Segmented | 34.57 ± 2.70 | 2.13 ± 0.13 |

> Segmented 校正在当前数据集上无改善，略有负面影响

### 结论与建议

1. **推荐配置**：`ERT + Augmented + Liquid` (E07)
2. **特征选择**：若有熔体成分数据，**必须使用Liquid特征集**（T_RMSE提升40%）
3. **数据模块**：Augmented带来稳定收益（T_RMSE ↓3-4°C）
4. **模型选择**：ERT 最优且最稳定，CatBoost 次之
5. **校正策略**：当前数据集上 Segmented 无收益，保持 None

---
## 变更日志

### 2026-01-22（V4.2）更新

**结果分析与验证**：
- 完成 24 组实验全量运行，验证所有修改正确实施
- 生成离线绘图（16张图表）和 MC 不确定性量化结果

**核心结论**：
1. **特征集效应最显著**：Liquid 使 T_RMSE ↓40%（56.4→33.7°C）
2. **数据增强稳定有效**：Augmented 使 T_RMSE ↓3-4°C
3. **模型选择影响有限**：ERT 最优且最稳定
4. **校正策略无显著收益**：Segmented 在当前数据集无改善

**推荐配置**：`ERT + Augmented + Liquid`（T_RMSE=30.69°C, R²=0.934）

### 2026-01-21（V4.1）更新

- MC 不确定性测试样本数从 100 调整为 10（扰动次数 n_mc=1000 保持不变）

### 2026-01-21（V4）更新

- 重命名 GroupCVProtocol → StratifiedCVProtocol，明确仅使用分层CV
- 主实验输出新增固定测试集指标（T_test_*/P_test_*）
- metrics_summary.csv 现采用追加+按 exp_id 去重策略（保留最新）
- summarize_results.py 将测试集指标合并至 experiment_summary.csv
- CatBoost GPU 自动检测 + n_jobs 自动配置（Windows 默认为 1）
- summarize_folds 默认计算置信区间（ddof=1, dropna）

### 2026-01-19（V3）更新

**主要修改**：
1. M1 数据增强对齐 EPMA 误差模型（>1 wt% 3%，<=1 wt% 8%），训练增强规模 n_perturbations_train=15。
2. M4 不确定性：MC 输入扰动使用 EPMA 误差模型，n_mc=1000；指标改用 MC 中位数，并输出 p16/p84。
3. M3 校正：仅 E09–E12 从 residual 改为 segmented，并按训练集 min–max 做 clip；其余保持 none。
4. 汇总输出：summarize_results 统一打印 R2，避免控制台编码问题。

**结果（10 折 CV，24 实验）**：
- 最佳配置：E08_catboost_augmented_none_liq
  - T_rmse=29.72 °C，T_r2=0.9378
  - P_rmse=1.85 kbar，P_r2=0.9237
- NoLiquid 最佳：E07_ert_augmented_none_noliq（T_rmse=52.42 °C，P_rmse=2.31 kbar）

**实验分析**：
- 特征集影响最大：Liquid 平均 T_rmse≈31.69 vs NoLiquid≈55.01；P_rmse≈1.98 vs 2.44。
- 数据增强稳定提升：augmented 相对 raw 平均约 -1.63 °C / -0.11 kbar。
- balanced 效果不稳定：平均略差（+0.13 °C / +0.02 kbar）。
- 分段校正总体对 T 无提升，对 P 仅小幅改善（约 -0.02 kbar），组合间存在正负波动。
- 模型对比：ERT 与 CatBoost 接近，stacking 相对较弱。

---


### 2026-01-19（V2）更新

- 新增双特征集支持（NoLiquid/Liquid），实验扩展至24组
- 移除Ref分组约束，改用P-T网格分层

##### 最佳配置

| 目标 | 最佳实验 | RMSE | R2 | 配置 |
|------|----------|------|-----|------|
| **温度 T** | E08_catboost_augmented_none_liq | **30.72 °C** | **0.934** | CatBoost + 数据增强 + Liquid |
| **压力 P** | E08_catboost_augmented_none_liq | **1.93 kbar** | **0.917** | CatBoost + 数据增强 + Liquid |

- **主要结论**：Liquid特征集使T_RMSE降低约42%

---

### 2026-01-18（V1）更新

- 完成12个基础实验矩阵（StratifiedKFold）
- 修复Windows兼容性问题（n_jobs=1, 编码问题）
- 生成初步结果汇总

- 最佳T模型：E11_catboost_balanced_residual（T_RMSE=42.65, T_R2=0.865）
- 最佳P模型：E08_catboost_augmented_none（P_RMSE=2.64, P_R2=0.826）

## 参考文献

Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

---

## License

MIT
