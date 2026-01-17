# ML Thermobarometer Benchmark Protocol

机器学习温压计模块化评估协议框架 - 用于 Chapter 3 标准化效能分析

## 核心特性

- ✅ **模块化消融实验**：M1(数据) × M2(算法) × M3(校正) 全因子设计
- ✅ **严格防泄露**：所有拟合操作仅在训练折内完成
- ✅ **双协议对照**：GroupKFold(主) + RandomSplit(对照)
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
│
├── src/                         # 核心代码模块
│   ├── __init__.py              # 模块导出
│   ├── interfaces.py            # M1-M4 统一接口定义
│   ├── data_modules.py          # M1: Raw/Balanced/Augmented
│   ├── model_modules.py         # M2: ERT/CatBoost/RF/Stacking
│   ├── correction_modules.py    # M3: None/ResidualRegression
│   ├── uncertainty_modules.py   # M4: MCUncertaintyEstimator
│   ├── protocol.py              # Pipeline + GroupCV/Random 协议
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

### 3. 快速测试（2折 × 3实验）

```bash
python main.py --test
```

### 4. 运行完整实验矩阵（10折 × 12实验）

```bash
python main.py
```

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
| `ResidualRegressionCorrector` | 残差回归校正 |

### M4 不确定性模块 (UncertaintyModule)

| 模块 | 说明 |
|------|------|
| `MCUncertaintyEstimator` | 蒙特卡洛输入扰动 + PICP/区间宽度 |

---

## 实验矩阵（12组消融实验）

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

1. **外层 CV 必须用 GroupKFold**：按文献来源（Ref）分组，禁止随机划分
2. **所有拟合只在训练折内**：标准化器、权重、增强、校正器
3. **Stacking 必须 Strict OOF**：内层 KFold 生成元特征，禁止泄露
4. **T/P 独立双链路**：温度和压力分别建模、预测、校正

---

## 参考文献

Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

---

## License

MIT
