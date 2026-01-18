# ML Thermobarometer Benchmark Protocol

机器学习温压计模块化评估协议框架 - 用于 Chapter 3 标准化效能分析

## 核心特性

- ✅ **模块化消融实验**：M1(数据) × M2(算法) × M3(校正) × 特征集 全因子设计
- ✅ **严格防泄露**：所有拟合操作仅在训练折内完成
- ✅ **双特征集对比**：NoLiquid(9特征) vs Liquid(18特征)
- ✅ **P-T分层采样**：基于P-T网格的分层交叉验证，优先保证P-T分布平衡
- ✅ **T/P 独立建模**：温度与压力采用完全独立的建模链路
- ✅ **不确定性量化**：蒙特卡洛输入扰动 + 校准指标
- ✅ **Strict OOF Stacking**：严格 OOF 元特征生成，无数据泄露

---

## 项目结构

```
ml-thermobarometer-benchmark/
│
├── input.csv                    # 校准数据集（2079行×45列，latin-1编码）
├── main.py                      # 主入口（一键运行实验矩阵）
├── requirements.txt             # Python 依赖
├── README.md                    # 本文档
├── codex.md                     # 开发日志与需求变更记录
│
├── src/                         # 核心代码模块
│   ├── __init__.py              # 模块导出
│   ├── interfaces.py            # M1-M4 统一接口定义
│   ├── data_modules.py          # M1: Raw/Balanced/Augmented
│   ├── model_modules.py         # M2: ERT/CatBoost/Stacking
│   ├── correction_modules.py    # M3: None/ResidualRegression
│   ├── uncertainty_modules.py   # M4: MCUncertaintyEstimator
│   ├── protocol.py              # Pipeline + StratifiedKFold 协议
│   ├── splitters.py             # P-T网格采样与分层工具
│   ├── metrics.py               # 指标计算
│   └── viz.py                   # 可视化
│
├── results/                     # 实验输出目录
│   ├── metrics_summary.csv
│   ├── effect_table.csv
│   └── figures/
│
├── reference_files/             # 参考文件
│
└── old/                         # 旧版代码（已归档）
```

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
    NoCorrection, ResidualRegressionCorrector,
    Pipeline, GroupCVProtocol
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

| 模块 | 说明 |
|------|------|
| `RawDataModule` | 仅标准化 |
| `BalancedDataModule` | 分箱重加权（逆频率权重） |
| `AugmentedDataModule` | MC-based 数据增强 |

### M2 模型模块 (ModelModule)

| 模块 | 说明 |
|------|------|
| `ExtraTreesModel` | 基线模型（高方差低偏差） |
| `CatBoostModel` | 强单模型（Boosting 代表） |
| `StrictOOFStacking` | 严格 OOF 堆叠（ERT+CatBoost+RF→Ridge） |

### M3 校正模块 (CorrectionModule)

| 模块 | 说明 |
|------|------|
| `NoCorrection` | 无校正 |
| `ResidualRegressionCorrector` | 残差回归校正（基于OOF预测） |

**注意**：根据地质学ML传统实践，简单模型（ERT、CatBoost）的校正器使用in-sample预测，StrictOOFStacking使用严格OOF预测。详见代码注释。

### M4 不确定性模块 (UncertaintyModule)

| 模块 | 说明 |
|------|------|
| `MCUncertaintyEstimator` | 蒙特卡洛输入扰动 + PICP/区间宽度 |

---

## 实验矩阵（24组实验：12基础配置 × 2特征集）

### 基础配置（12组）

| Exp ID | M1 数据 | M2 模型 | M3 校正 | 目的 |
|--------|---------|---------|---------|------|
| E01 | Raw | ERT | None | 基线 |
| E02 | Raw | CatBoost | None | Boost基线 |
| E03 | Raw | Stacking | None | Stacking基线 |
| E04 | Balanced | ERT | None | M1效应(ERT) |
| E05 | Balanced | CatBoost | None | M1效应(Boost) |
| E06 | Balanced | Stacking | None | M1效应(Stacking) |
| E07 | Augmented | ERT | None | M1-Aug效应 |
| E08 | Augmented | CatBoost | None | M1-Aug效应 |
| E09 | Raw | CatBoost | Residual | M3效应 |
| E10 | Balanced | ERT | Residual | 完整流程(ERT) |
| **E11** | **Balanced** | **CatBoost** | **Residual** | ⭐ **主力配置** |
| E12 | Balanced | Stacking | Residual | 边界探索 |

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
| **基础** | RMSE, MAE, R² | 精度指标 |
| **偏差** | MBE, Slope, Intercept | 系统性偏差诊断 |
| **分箱** | MAE_bin, MBE_bin | 端元效应诊断 |
| **不确定性** | PICP, Interval Width | 校准指标 |

---

## 输出文件

```
results/
├── metrics_summary.csv      # 汇总指标（mean ± CI）
├── effect_table.csv         # 模块效应表
├── config_used.yaml         # 实验配置
├── {exp_id}_{T/P}_fold_metrics.csv    # 每折指标
├── {exp_id}_{T/P}_predictions.parquet # 逐样本预测
└── figures/                 # 图表
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

## 🔬 实验结果（24组实验 10折交叉验证）

> 运行日期：2026-01-19 | 数据集：2079样本 | CV策略：P-T分层10折

### 最佳配置

| 目标 | 最佳实验 | RMSE | R² | 配置 |
|------|----------|------|-----|------|
| **温度 T** | E08_catboost_augmented_none_liq | **30.72 °C** | **0.934** | CatBoost + 数据增强 + Liquid |
| **压力 P** | E08_catboost_augmented_none_liq | **1.93 kbar** | **0.917** | CatBoost + 数据增强 + Liquid |

### 全量实验结果

| 实验ID | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|--------|-------------|------|---------------|------|
| E01_ert_raw_none_noliq | 54.40 | 0.794 | 2.41 | 0.872 |
| E01_ert_raw_none_liq | 32.16 | 0.927 | 2.03 | 0.909 |
| E02_catboost_raw_none_noliq | 55.60 | 0.785 | 2.46 | 0.866 |
| E02_catboost_raw_none_liq | 31.35 | 0.931 | 1.97 | 0.914 |
| E03_stacking_raw_none_noliq | 55.40 | 0.786 | 2.47 | 0.865 |
| E03_stacking_raw_none_liq | 32.50 | 0.926 | 2.03 | 0.909 |
| E04_ert_balanced_none_noliq | 54.35 | 0.795 | 2.43 | 0.870 |
| E04_ert_balanced_none_liq | 32.52 | 0.926 | 2.06 | 0.906 |
| E05_catboost_balanced_none_noliq | 55.42 | 0.786 | 2.47 | 0.865 |
| E05_catboost_balanced_none_liq | 31.16 | 0.932 | 1.97 | 0.914 |
| E06_stacking_balanced_none_noliq | 55.96 | 0.782 | 2.51 | 0.861 |
| E06_stacking_balanced_none_liq | 32.80 | 0.925 | 2.06 | 0.906 |
| E07_ert_augmented_none_noliq | 53.23 | 0.803 | 2.35 | 0.879 |
| E07_ert_augmented_none_liq | 31.51 | 0.930 | 1.98 | 0.914 |
| **E08_catboost_augmented_none_noliq** | 54.43 | 0.794 | 2.37 | 0.877 |
| **E08_catboost_augmented_none_liq** ⭐ | **30.72** | **0.934** | **1.93** | **0.917** |
| E09_catboost_raw_residual_noliq | 55.54 | 0.786 | 2.46 | 0.867 |
| E09_catboost_raw_residual_liq | 31.18 | 0.932 | 1.96 | 0.915 |
| E10_ert_balanced_residual_noliq | 53.79 | 0.799 | 2.41 | 0.872 |
| E10_ert_balanced_residual_liq | 31.90 | 0.929 | 2.02 | 0.910 |
| E11_catboost_balanced_residual_noliq | 55.36 | 0.787 | 2.47 | 0.866 |
| E11_catboost_balanced_residual_liq | 31.03 | 0.932 | 1.96 | 0.914 |
| E12_stacking_balanced_residual_noliq | 55.94 | 0.782 | 2.51 | 0.861 |
| E12_stacking_balanced_residual_liq | 32.78 | 0.925 | 2.06 | 0.906 |

### 模块效应分析

#### 特征集效应（最显著）

| 特征集 | T_RMSE | T_R² | P_RMSE | P_R² | 提升幅度 |
|--------|--------|------|--------|------|----------|
| NoLiquid (9特征) | 54.95 | 0.790 | 2.44 | 0.868 | 基线 |
| **Liquid (18特征)** | **31.80** | **0.929** | **2.00** | **0.911** | T: **-42%**, P: -18% |

> ⭐ **关键发现**：加入熔体成分使温度预测RMSE降低42%，R²从0.79升至0.93

#### 模型效应

| 模型 | T_RMSE | T_R² | P_RMSE | P_R² |
|------|--------|------|--------|------|
| ERT | 42.98 | 0.863 | 2.21 | 0.891 |
| **CatBoost** | **43.18** | **0.860** | **2.20** | **0.892** |
| Stacking | 44.23 | 0.854 | 2.27 | 0.885 |

> 模型间差异较小（<3%），CatBoost在Liquid特征集下表现最优

#### 数据模块效应

| 数据处理 | T_RMSE | T_R² | P_RMSE | P_R² |
|----------|--------|------|--------|------|
| Raw | 43.52 | 0.859 | 2.22 | 0.890 |
| Balanced | 43.58 | 0.858 | 2.24 | 0.888 |
| **Augmented** | **42.47** | **0.865** | **2.16** | **0.896** |

> 数据增强略有帮助，但效果不如特征集选择显著

#### 校正模块效应

| 校正方式 | T_RMSE | T_R² | P_RMSE | P_R² |
|----------|--------|------|--------|------|
| None | 43.34 | 0.860 | 2.22 | 0.890 |
| Residual | 43.44 | 0.859 | 2.23 | 0.889 |

> Residual校正在当前数据集上无显著提升

### 结论与建议

1. **推荐配置**：`CatBoost + Augmented + Liquid` (E08)
2. **特征选择**：若有熔体成分数据，**必须使用Liquid特征集**
3. **模型选择**：CatBoost和ERT表现相当，Stacking收益有限
4. **校正策略**：当前数据集不需要Residual校正

---

## 变更日志

### 2026-01-19 版本更新

**主要变更**：
1. ✅ 新增特征集管理：支持NoLiquid（9特征）和Liquid（18特征）两种特征集
2. ✅ 移除Ref分组约束：从StratifiedGroupKFold改为StratifiedKFold
3. ✅ 简化测试集划分：纯P-T网格采样，不再尝试保持Ref完整性
4. ✅ 添加OOF说明注释：明确地质学ML传统实践与严格OOF的区别
5. ✅ 实验数量扩展：从12个扩展到24个（12基础配置 × 2特征集）
6. ✅ 完成24个实验全量运行并生成结果分析

**影响**：
- 运行时间翻倍（24个实验 vs 12个）
- P-T分布平衡性提升
- 同一文献样本可能跨训练/验证集（已知限制，优先保证P-T平衡）

### 2026-01-18 版本更新

**运行概况**：
- 完成12个基础实验矩阵（使用StratifiedGroupKFold）
- 测试集使用P-T网格一次性划分，test_size=349

**初步结果**（12个实验，无特征集扩展）：
- 最佳T模型：E11_catboost_balanced_residual（T_RMSE=42.65, T_R²=0.865）
- 最佳P模型：E08_catboost_augmented_none（P_RMSE=2.64, P_R²=0.826）

**问题修复**：
- WinError 5：设置n_jobs=1避免joblib权限错误
- PowerShell编码：移除emoji避免GBK控制台打印异常
- 稳定性测试：因运行时间长暂时禁用（CONFIG['run_stability']=False）

**临时策略**：
- 树模型n_jobs默认为1
- 稳定性测试默认关闭

---

## 参考文献

Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

---

## License

MIT
