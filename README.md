# ML Thermobarometer Benchmark Protocol

机器学习温压计模块化评估协议框架 - 用于 Chapter 3 标准化效能分析

---

# 第一部分：项目概览

## 核心特性

- ✅ **模块化消融实验**：M1(数据) × M2(算法) × M3(校正) × 特征集 全因子设计
- ✅ **严格防泄露**：所有拟合操作仅在训练折内完成
- ✅ **双特征集对比**：NoLiquid(9特征) vs Liquid(18特征)
- ✅ **P-T分层采样**：基于P-T网格的分层交叉验证，优先保证P-T分布平衡
- ✅ **稀疏 bin 保护**：在学习曲线/稳定性测试中自动合并稀疏 bins，并按需降低折数以避免崩溃
- ✅ **T/P 独立建模**：温度与压力采用完全独立的建模链路
- ✅ **Strict OOF Stacking**：严格 OOF 元特征生成，无数据泄露
- ✅ **工具分离**：主实验与绘图/稳定性测试/不确定性量化分离，按需运行

## 项目结构

```
ml-thermobarometer-benchmark/
├── input.csv                    # 校准数据集（2079行，latin-1编码）
├── main.py                      # 主入口（运行实验矩阵）
├── config.py                    # 集中配置管理（见下方配置说明）
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
│   ├── perturbation.py          # 共用 EPMA 扰动模块
│   ├── metrics.py               # 指标计算
│   ├── logger.py                # 统一日志模块
│   └── viz.py                   # 可视化（含论文图件）
│
├── tests/                       # 测试目录
│   ├── conftest.py              # pytest fixtures
│   ├── test_data_modules.py     # M1 测试
│   ├── test_model_modules.py    # M2 测试
│   ├── test_correction_modules.py # M3 测试
│   ├── test_protocol.py         # 协议测试
│   ├── test_splitters.py        # 划分工具测试
│   └── test_metrics.py          # 指标测试
│
├── tools/                       # 工具脚本（独立运行）
│   ├── run_stability.py         # 稳定性测试
│   ├── run_error_propagation.py # 分析误差传播（MC不确定性）
│   ├── run_learning_curve.py    # 学习曲线分析
│   ├── plot_offline_figures.py  # 离线绘图 + 论文图件
│   └── checkpoint_manager.py    # 检查点管理工具
│
└── results/                     # 实验输出目录
    ├── metrics_summary.csv      # 汇总指标
    ├── effect_table.csv         # 模块效应表
    ├── config_used.yaml         # 配置记录
    ├── logs/                    # 日志文件
    ├── learning_curve/          # 学习曲线结果
    ├── stability/               # 稳定性测试结果
    ├── error_propagation/       # 分析误差传播结果
    ├── models/                  # 保存的模型
    └── figures/                 # 可视化图表
```

### 配置管理说明

`config.py` 是项目唯一配置来源，通过 `get_config_dict()` 获取配置。
模型/增强/不确定性参数在 `main.py` 中透传到对应模块。
配置分为两类：

**✅ 运行时配置（可根据需要修改）**
- `data_path`: 数据文件路径
- `n_splits`: CV 折数（默认 10）
- `random_seed`: 随机种子（默认 42）
- `output_dir`: 输出目录

**⚠️ 模型默认参数（经过调优，不建议随意修改）**
- `model_defaults.ert`: ExtraTrees 参数（n_estimators=200, max_depth=15）
- `model_defaults.catboost`: CatBoost 参数（iterations=1000, depth=6）
- `model_defaults.stacking`: Stacking 参数（inner_cv/use_meta_scaler）
- `model_defaults.stacking_base_defaults`: Stacking 基模型参数覆盖（可选）
- `augmentation.n_aug`: 增强副本数（默认 15，基于 Ágreda-López 2024）
- `uncertainty.n_mc / uncertainty.percentiles`: MC 不确定性默认配置


修改模型参数可能导致性能下降或与论文结果不可复现。如需调参，建议先使用学习曲线工具评估影响。

## 整体工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                         input.csv (2079 samples)                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   load_data() + prepare_splits()   │
                    │   P-T 网格采样 → 固定测试集        │
                    └─────────────┬─────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐       ┌───────────────┐       ┌───────────────┐
│   main.py     │       │run_learning   │       │run_stability  │
│ 24组实验矩阵  │       │  _curve.py    │       │    .py        │
│ 10-fold CV    │       │ 学习曲线分析  │       │ 稳定性测试    │
└───────┬───────┘       └───────┬───────┘       └───────┬───────┘
        │                       │                       │
        │               ┌───────▼───────┐               │
        │               │run_error_     │               │
        │               │propagation.py │               │
        │               │分析误差传播   │               │
        │               └───────┬───────┘               │
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────────────────────────────────────────────────────────┐
│                         results/                                   │
│  metrics_summary.csv | learning_curve/* | stability/*             │
│  error_propagation/* | *_predictions.parquet | models/*.joblib    │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │ plot_offline_figures  │
                    │   离线绘图（纯读取）  │
                    └───────────────────────┘
```

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
    NoCorrection, SegmentedLinearCorrector,
    Pipeline, StratifiedCVProtocol
)
print('OK')
```

### 3. 快速测试（2折 × 4实验，约2分钟）

```bash
python main.py --test
```

测试将运行前2个基础配置 × 2种特征集 = 4个实验

### 4. 运行完整实验（10折 × 24实验，约30分钟）

```bash
python main.py
```

完整运行包含12个基础配置 × 2种特征集 = 24个实验

### 5. 生成论文图件

```bash
python tools/plot_offline_figures.py
```

---

# 第二部分：详细技术文档

## 运行说明（并发/断点）

- 长耗时任务建议分段并行运行，所有分段必须使用一致的 `exp-id`、模型/特征/校正配置与 `output-dir`。
- 稳定性测试支持分段与断点：`--repeat-start/--repeat-end` 为闭区间；`--resume` 会跳过已完成的 `repeat_id`；完成后用 `--merge-dir` 合并并生成汇总。
- 学习曲线支持分段与断点：`--repeat-start/--repeat-end` 配合 `--resume`；全部结束后用 `--merge-dir` 汇总 runs 与 summary。

**稳定性并行示例（方案 A，4 段并行）**：

```bash
python tools/run_stability.py --exp-id E07_stability --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-repeats 1000 --repeat-start 0 --repeat-end 249 --resume
python tools/run_stability.py --exp-id E07_stability --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-repeats 1000 --repeat-start 250 --repeat-end 499 --resume
python tools/run_stability.py --exp-id E07_stability --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-repeats 1000 --repeat-start 500 --repeat-end 749 --resume
python tools/run_stability.py --exp-id E07_stability --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-repeats 1000 --repeat-start 750 --repeat-end 999 --resume

# 合并与汇总
python tools/run_stability.py --merge-dir results --exp-id E07_stability
```

## 特征集配置

### NoLiquid（9个特征）

仅使用单斜辉石（CPX）主要氧化物成分：

```python
['SiO2.cpx', 'TiO2.cpx', 'Al2O3.cpx', 'Cr2O3.cpx',
 'FeO.cpx', 'MgO.cpx', 'MnO.cpx', 'CaO.cpx', 'Na2O.cpx']
```

**适用场景**：仅有矿物成分数据，无共存熔体成分

### Liquid（18个特征）

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
| `AugmentedDataModule` | EPMA 误差模型增强 | 按列名映射误差率，主量 3%，低含量 8%，n=15 |

### M2 模型模块 (ModelModule)

| 模块 | 说明 | 核心特点 |
|------|------|----------|
| `ExtraTreesModel` | 基线模型 | Bagging 集成，n_estimators=200, max_depth=15 |
| `CatBoostModel` | Boosting 代表 | iterations=1000, depth=6, learning_rate=0.03 |
| `StrictOOFStacking` | 严格 OOF 堆叠 | ERT+CatBoost+RF → Ridge，内层5折 |

### M3 校正模块 (CorrectionModule)

| 模块 | 说明 | 适用场景 |
|------|------|----------|
| `NoCorrection` | 无校正 | 默认配置 |
| `ResidualRegressionCorrector` | 残差回归校正（Ridge） | 系统性偏差修正（可选，保留接口） |
| `SegmentedLinearCorrector` | 分段线性校正 | 端元效应修正|

**注意**：所有模型的校正器统一使用全量训练集的 OOF（Out-of-Fold）预测进行拟合，确保无数据泄露。

### M4 不确定性模块 (UncertaintyModule)

| 模块 | 说明 | 参数 |
|------|------|------|
| `MCUncertaintyEstimator` | MC 输入扰动 | n_mc=1000, EPMA误差模型 |

**说明**：
- 使用EPMA误差模型生成输入扰动（与M1对齐）
- 输出多分位数（p5/p16/p50/p84/p95）
- 计算校准指标：PICP_68/PICP_90
- 主实验会根据 feature_set 显式传入 feature_names
- 未显式传入 feature_names 时按特征数推断（18→Liquid，9→NoLiquid），否则直接报错，禁止默认 3% 降级

---

## 核心函数说明

### `protocol.py` - 协议执行器

| 类/函数 | 职责 |
|---------|------|
| `Pipeline` | 封装 DM+MM+CM 流程，`fit()`, `predict()` |
| `StratifiedCVProtocol` | P-T 分层 10 折 CV 协议 |
| `ExperimentConfig` | 实验配置 dataclass |
| `ExperimentMatrix` | 批量运行 24 实验 |

### `splitters.py` - 数据划分

| 函数 | 功能 |
|------|------|
| `compute_pt_edges()` | 计算 P-T 网格边界（k = ceil(sqrt(n))） |
| `assign_pt_bins()` | 分配样本到 P-T 格子 |
| `select_test_indices()` | 每个非空格子随机选 1 个样本 |

### `metrics.py` - 指标计算

| 函数 | 功能 |
|------|------|
| `rmse`, `mae`, `r2`, `mape`, `bias` | 基础指标 |
| `compute_slope_intercept` | 回归诊断 |
| `summarize_folds` | 折叠汇总 + CI 计算 |

### `viz.py` - 可视化

| 函数 | 功能 | 论文图件 |
|------|------|----------|
| `plot_pred_vs_true` | 预测-真实散点图 | - |
| `plot_residuals` | 残差分布图 | - |
| `plot_fold_comparison` | 各折指标对比 | - |
| `plot_full_report` | 全量诊断报告图 | - |
| `plot_correction_effect` | 校正前后对比 | - |
| `plot_feature_importance` | 特征重要性图 | - |
| `plot_pt_grid_cv_splits` | P-T 网格 CV 示意图 | **图 3-2** |
| `plot_feature_set_comparison_boxplot` | 特征集对比箱线图 | **图 3-3** |
| `plot_parity_comparison` | 1:1 预测对比图 | **图 3-4** |
| `plot_m1_ablation_stepwise` | M1 消融阶梯图 | **图 3-5** |
| `plot_performance_heatmap_matrix` | 性能热力图 | **图 3-6** |

---

## 工具脚本详解

### main.py - 主实验入口

**输入**：`input.csv`，命令行参数 `--test`（可选）

**输出**：
```
results/
├── metrics_summary.csv              # 汇总指标
├── effect_table.csv                 # 效应分析表
├── config_used.yaml                 # 配置记录
├── {exp_id}_{T/P}_fold_metrics.csv  # 每折指标（48个文件）
├── {exp_id}_{T/P}_predictions.parquet # 预测结果（48个文件）
└── models/{exp_id}_{target}_model.joblib # 模型文件（48个文件）
```

**工作流程**：
1. 加载数据 → 2. P-T网格划分测试集 → 3. 24组实验10折CV → 4. 保存结果

### run_learning_curve.py - 学习曲线分析

**输入参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--models` | `ert stacking` | 模型列表 |
| `--fractions` | `0.2 0.4 0.6 0.8 1.0` | 采样比例列表 |
| `--repeats` | `30` | 重复次数（跨随机种子） |
| `--repeat-start` | None | 分段运行 repeat 起始 |
| `--repeat-end` | None | 分段运行 repeat 结束 |
| `--merge-dir` | None | 合并 runs 文件并汇总 |
| `--resume` | False | 断点续跑（自动跳过已完成任务） |
| `--checkpoint-interval` | `20` | 每N个repeat保存检查点 |
| `--n-splits` | `10` | CV折数 |
| `--output-dir` | `results/learning_curve` | 输出目录 |

**输出**：
```
results/learning_curve/
├── learning_curve_checkpoint.csv        # 检查点文件（断点续跑用）
├── learning_curve_runs_rep_000_009.csv  # 分段运行时的 runs 文件
├── learning_curve_runs.csv              # 完整的 runs 文件
└── learning_curve_summary.csv           # 汇总统计结果

results/figures/
└── learning_curve_*.png                 # 学习曲线图（由 plot_offline_figures.py 生成）
```

**使用方式**：

```bash
# 方式1：一次性运行（适合小规模测试）
python tools/run_learning_curve.py --repeats 5

# 方式2：断点续跑（推荐，中断后可继续）
python tools/run_learning_curve.py --repeats 30 --resume

# 方式3：分段运行（适合并行或分批运行）
python tools/run_learning_curve.py --repeat-start 0 --repeat-end 9 --output-dir results/learning_curve
python tools/run_learning_curve.py --repeat-start 10 --repeat-end 19 --output-dir results/learning_curve
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 --output-dir results/learning_curve
python tools/run_learning_curve.py --merge-dir results/learning_curve

# 生成学习曲线图（离线绘图）
python tools/plot_offline_figures.py --learning-curve-dir results/learning_curve
```

### run_stability.py - 稳定性测试

**输入参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-id` | `stability_test` | 实验ID（输出文件名前缀） |
| `--model-module` | `ert` | 模型 |
| `--data-module` | `augmented` | 数据模块 |
| `--n-repeats` | `1000` | 重复次数（不同随机种子） |
| `--random-seed` | `42` | 基础随机种子（实际种子=base+repeat_id） |
| `--repeat-start` | None | 分段运行 repeat 起始 |
| `--repeat-end` | None | 分段运行 repeat 结束 |
| `--merge-dir` | None | 合并分段结果并汇总 |
| `--checkpoint-interval` | `20` | 每N次保存检查点 |
| `--resume` | False | 从检查点恢复 |

注：稳定性重复的随机种子采用 `base_seed + repeat_id` 派生。

**输出**：
```
results/stability/
├── {exp_id}_T_test_metrics.csv          # T目标完整结果
├── {exp_id}_P_test_metrics.csv          # P目标完整结果
├── {exp_id}_*_checkpoint_*.csv          # 检查点文件
└── stability_summary.csv                # 汇总统计
```

**使用方式**：

```bash
# 方式1：快速测试（100次）
python tools/run_stability.py --n-repeats 100

# 方式2：断点续跑（推荐，中断后可继续）
python tools/run_stability.py --n-repeats 1000 --resume

# 方式3：分段运行（适合并行）
python tools/run_stability.py --repeat-start 0 --repeat-end 199
python tools/run_stability.py --repeat-start 200 --repeat-end 399
python tools/run_stability.py --repeat-start 400 --repeat-end 599
python tools/run_stability.py --repeat-start 600 --repeat-end 799
python tools/run_stability.py --repeat-start 800 --repeat-end 999
python tools/run_stability.py --merge-dir results
```

### run_error_propagation.py - 分析误差传播

**设计理念**：评估 EPMA 分析误差经固定模型放大后的输出离散度（analysis-limited precision），而非模型的总体预测误差。

**核心原则**：
1. **模型固定**：在全部训练数据上拟合一个固定模型，不做 CV、不重新训练
2. **样本固定**：对固定的测试样本进行多次输入扰动
3. **仅扰动输入**：对输入组成（氧化物 wt%）添加 EPMA 误差模型噪声
4. **评估输出离散度**：计算输出分布的标准差、区间宽度等

**EPMA 误差模型**（Ágreda-López et al. 2024）：
- 按氧化物列名固定映射（而非按数值阈值）
- 主量元素（SiO2, Al2O3, FeO, MgO, CaO）：3%
- 低含量元素（TiO2, MnO, Na2O, Cr2O3, K2O）：8%
- 不做负值截断，保留完整正态分布
- 不做闭合约束，与训练数据预处理一致

**输入参数**：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-id` | 自动生成 | 实验ID（与主实验命名一致） |
| `--data-module` | `augmented` | 数据模块 |
| `--model-module` | `ert` | 模型 |
| `--corr-module` | `none` | 校正模块（none/residual/segmented） |
| `--feature-set` | `Liquid` | 特征集 |
| `--feature-names` | `None` | 自定义特征名（逗号/JSON/@file，非9/18时必填） |
| `--random-seed` | `42` | 模型/数据随机种子 |
| `--n-mc` | `1000` | MC 采样次数 |
| `--mc-sample-size` | `-1` | 测试样本数（-1 = 全部测试集） |
| `--mc-seed` | `42` | MC 随机种子 |

**输出**：
```
results/error_propagation/
├── {exp_id}_ep_meta.json           # 实验元数据（含设计说明/采样分布统计）
├── {exp_id}_ep_T_summary.csv       # 温度汇总统计
├── {exp_id}_ep_T_samples.csv       # 逐样本 T 预测与不确定度
├── {exp_id}_ep_P_summary.csv       # 压力汇总统计
└── {exp_id}_ep_P_samples.csv       # 逐样本 P 预测与不确定度
```

**输出指标说明**：
| 指标类型 | 指标名 | 含义 |
|----------|--------|------|
| 总误差 | `rmse/mae/mbe/r2` | 相对真值的误差（含模型误差）；`mbe = y_true - y_pred`，正值代表低估 |
| 回归诊断 | `slope/intercept/resid_std` | 线性拟合诊断与残差波动 |
| 分析误差 | `analysis_std_mean/median` | 输出离散度（标准差） |
| 分析误差 | `analysis_2mad_mean` | 传播误差 = 2×MAD（稳健形式） |
| 分析误差 | `analysis_interval_68/90_mean` | 输出分布区间宽度 |
| 对比 | `analysis_contribution_ratio` | 传播误差对总误差的比例（analysis_std / rmse，近似） |
| 对比 | `main_test_rmse` | 主实验独立测试集 RMSE（用于对比） |
| 对比 | `propagated_vs_test_ratio` | 传播误差 / test_RMSE 比例 |

> 注：analysis_* 仅反映输入扰动导致的离散度；total_* 为未扰动基线误差（含模型误差），两者口径不同。

**使用方式**：
```bash
# 默认配置（ERT + Augmented + Liquid，对应主实验 E07）
python tools/run_error_propagation.py --model-module ert

# 指定其他配置
python tools/run_error_propagation.py --model-module catboost --feature-set NoLiquid

# 启用校正模块
python tools/run_error_propagation.py --corr-module segmented

# 自定义特征名（非9/18特征时必须指定）
python tools/run_error_propagation.py --feature-names "@feature_names.json"
```
### plot_offline_figures.py - 离线绘图（V7.4）

**输入参数**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--results-dir` | `results` | 结果目录 |
| `--exp-id` | `E07_ert_augmented_none_liq` | SHAP/模型相关默认实验 |
| `--data-path` | `input.csv` | 数据路径 |
| `--stability-exp-id` | `E07_stability_nj4` | 稳定性实验 ID（全量绘图模式） |
| `--learning-curve-dir` | `results/learning_curve` | 学习曲线汇总目录 |
| `--fig-subdir` | `figures` | 图件输出子目录（默认输出到 `results/figures`） |
| `--correction-delta-exp-id` | `E10_ert_augmented_segmented_liq` | correction-delta 图使用的预测文件前缀 |
| `--enable-shap` | `False` | 是否启用 SHAP 绘图 |
| `--shap-max-samples` | `300` | SHAP 最大采样样本数 |
| `--shap-bg-k` | `50` | Kernel SHAP 背景聚类数 |
| `--shap-force` | `False` | 已存在 SHAP 图时是否强制覆盖 |
| `--selected-only` | `False` | 仅生成保留图件集合 |

**保留图件模式（`--selected-only`）输出**
- `pt_sampling_bias_overview.png`
- `parity_compare_TP.png`
- `learning_curve_TP.png`
- `{exp_id}_TP_SHAP_combined_{ModelName}.png`（启用 `--enable-shap` 时）
- `correction_delta_scatter_TP.png`

**关键函数行为（当前版本）**
- `plot_combined_shap_summary`：BayesV5 风格 SHAP dot+bar 叠加图；离线流程默认关闭总标题，保留坐标轴标题。
- `_plot_shap` + `_merge_shap_tp_images`：先生成 T/P，再合并成 TP；默认清理 SHAP 中间单图。
- `_plot_parity_compare` / `_plot_learning_curve`：生成 T/P 后合并为 `*_TP.png`，并在保留图件模式下清理中间单图。
- `plot_correction_delta_scatter_tp`：默认纯白背景，仅输出 PNG（无 JPG 流程），适配 Word 白底粘贴。

---

## 实验矩阵设计 (V5)

### 设计原则

- **E01-E03**：Raw 基线组
- **E04-E06**：Balanced 对比组（传统数据平衡）
- **E07-E09**：Augmented + M2 对比组（完整模型对比）
- **E10-E12**：Augmented + M3 对比组（验证校正效果）

**控制变量原则**：评估 M2/M3 时固定使用最佳 M1（Augmented）

### 基础配置（12组 × 2特征集 = 24实验）

| Exp | M1 数据 | M2 模型 | M3 校正 | 设计意图 |
|-----|---------|---------|---------|----------|
| E01 | Raw | ERT | None | 基线 |
| E02 | Raw | CatBoost | None | 基线 |
| E03 | Raw | Stacking | None | 基线 |
| E04 | Balanced | ERT | None | 传统方法对比 |
| E05 | Balanced | CatBoost | None | 传统方法对比 |
| E06 | Balanced | Stacking | None | 传统方法对比 |
| **E07** | **Augmented** | **ERT** | **None** | **最佳配置 ⭐** |
| E08 | Augmented | CatBoost | None | M2 对比 |
| E09 | Augmented | Stacking | None | M2 对比 |
| E10 | Augmented | ERT | Segmented | M3 对比 |
| E11 | Augmented | CatBoost | Segmented | M3 对比 |
| E12 | Augmented | Stacking | Segmented | M3 对比 |

---

## 核心协议约束

1. **P-T 分层 CV**：按 P-T bins 分层，保证各折温压分布平衡
2. **防泄露设计**：所有拟合（标准化、增强、校正）仅在训练折内
3. **Strict OOF Stacking**：内层 5 折生成元特征，无数据泄露
4. **T/P 独立建模**：温度和压力分别建模、预测、校正
5. **测试集固定**：基于 P-T 网格采样一次，所有实验共用
6. **主实验固定折数**：主实验不自动合并稀疏 bins，不降折

---

# 第三部分：实验结果

## 主实验结果（V7.3，24组实验）

> 运行日期：2026-02-03 | 数据集：1730样本 | CV策略：P-T分层10折

### 最佳配置

| 目标 | 最佳实验 | RMSE (CV mean) | R² | 配置 |
|------|----------|----------------|----|------|
| **温度 T** | E10_ert_augmented_segmented_liq | **30.51 °C** | **0.934** | ERT + Augmented + Segmented + Liquid |
| **压力 P** | E10_ert_augmented_segmented_liq | **1.89 kbar** | **0.921** | ERT + Augmented + Segmented + Liquid |

> **注**：E07 (None校正) 与 E10 (Segmented校正) 性能接近，E10 略优（T≈0.47°C、P≈0.05 kbar）。考虑复杂度，仍推荐 E07 作为实践基线。

### 全量实验结果（按 T_RMSE 排序）

| 实验ID | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|--------|-------------|------|---------------|------|
| E10_ert_augmented_segmented_liq | 30.51 | 0.934 | 1.89 | 0.921 |
| **E07_ert_augmented_none_liq** | 30.98 | 0.933 | 1.93 | 0.917 |
| E11_catboost_augmented_segmented_liq | 31.37 | 0.931 | 1.93 | 0.917 |
| E08_catboost_augmented_none_liq | 31.43 | 0.931 | 1.96 | 0.915 |
| E03_stacking_raw_none_liq | 31.62 | 0.930 | 2.00 | 0.911 |
| E12_stacking_augmented_segmented_liq | 31.64 | 0.930 | 1.99 | 0.912 |
| E06_stacking_balanced_none_liq | 31.67 | 0.930 | 2.02 | 0.910 |
| E09_stacking_augmented_none_liq | 31.93 | 0.928 | 2.01 | 0.910 |
| E01_ert_raw_none_liq | 32.13 | 0.928 | 2.04 | 0.908 |
| E04_ert_balanced_none_liq | 32.20 | 0.927 | 2.05 | 0.907 |
| E02_catboost_raw_none_liq | 36.31 | 0.908 | 2.24 | 0.889 |
| E05_catboost_balanced_none_liq | 36.53 | 0.906 | 2.29 | 0.884 |
| E10_ert_augmented_segmented_noliq | 52.33 | 0.809 | 2.33 | 0.880 |
| **E07_ert_augmented_none_noliq** | 52.92 | 0.805 | 2.35 | 0.878 |
| E12_stacking_augmented_segmented_noliq | 53.43 | 0.801 | 2.45 | 0.868 |
| E09_stacking_augmented_none_noliq | 53.60 | 0.800 | 2.45 | 0.867 |
| E03_stacking_raw_none_noliq | 53.73 | 0.799 | 2.41 | 0.872 |
| E06_stacking_balanced_none_noliq | 53.86 | 0.798 | 2.44 | 0.868 |
| E01_ert_raw_none_noliq | 54.35 | 0.795 | 2.43 | 0.870 |
| E04_ert_balanced_none_noliq | 54.43 | 0.794 | 2.46 | 0.867 |
| E11_catboost_augmented_segmented_noliq | 54.92 | 0.790 | 2.41 | 0.872 |
| E08_catboost_augmented_none_noliq | 54.98 | 0.789 | 2.41 | 0.871 |
| E02_catboost_raw_none_noliq | 59.43 | 0.754 | 2.63 | 0.848 |
| E05_catboost_balanced_none_noliq | 59.62 | 0.752 | 2.66 | 0.844 |

### 模块效应

#### 特征集效应

| 组别 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² | 变化幅度 |
|--------|-------------|------|---------------|------|----------|
| NoLiquid (9组) | 54.80 ± 2.34 | 0.791 | 2.45 ± 0.10 | 0.867 | 基准 |
| Liquid (18组) | 32.36 ± 1.95 | 0.926 | 2.03 ± 0.12 | 0.908 | T: ↓41%, P: ↓17% |

> 加入熔体成分后，温度 RMSE 下降约 41%，压力 RMSE 下降约 17%。

#### 模型效应（Liquid + Augmented）

| 模型 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|------|-------------|------|---------------|------|
| ERT | 30.75 ± 0.33 | 0.934 | 1.91 ± 0.03 | 0.919 |
| CatBoost | 31.40 ± 0.04 | 0.931 | 1.94 ± 0.02 | 0.916 |
| Stacking | 31.78 ± 0.21 | 0.929 | 2.00 ± 0.01 | 0.911 |

> ERT 略优于 CatBoost，Stacking 未体现增益且更复杂。

#### 数据模块效应（Liquid）

| 数据模块 | T_RMSE (°C) | T_R² | P_RMSE (kbar) | P_R² |
|----------|-------------|------|---------------|------|
| Raw | 33.35 ± 2.57 | 0.922 | 2.09 ± 0.13 | 0.903 |
| Balanced | 33.47 ± 2.67 | 0.921 | 2.12 ± 0.15 | 0.900 |
| Augmented | 31.31 ± 0.50 | 0.931 | 1.95 ± 0.04 | 0.915 |

> Augmented 在 T/P 两端均带来稳定改善。

#### 校正模块效应（Augmented + Liquid）

| 校正模块 | T_RMSE (°C) | P_RMSE (kbar) | 说明 |
|----------|-------------|---------------|------|
| None | 31.45 ± 0.47 | 1.97 ± 0.04 | 结构更简单 |
| Segmented | 31.17 ± 0.59 | 1.94 ± 0.05 | 略有提升但复杂度更高 |

### 结论

1. **实践基线**：`ERT + Augmented + None + Liquid` (E07) 兼顾性能与简洁性
2. **特征集**：显式加入 Liquid，T_RMSE ↓约 41%
3. **数据模块**：Augmented 提供稳定增益（T_RMSE ↓约 2.04°C）
4. **模型选择**：ERT 效果最佳，Stacking 无额外收益
5. **校正模块**：Segmented 提升有限，优先 None

## 学习曲线分析（模型复杂度边界）

> 运行日期：2026-02-05 | 数据模块：Augmented | 特征集：Liquid（18特征） | 校正：None | 模型：ERT/Stacking | repeats=8 | CV=5折

### 实验设计

评估训练数据从 20%→100% 增加时，ERT 与 Strict OOF Stacking 的泛化性能和稳定性变化。
- **数据模块**：Augmented（EPMA 误差模型增强）
- **特征集**：Liquid（18特征）
- **校正模块**：None
- **模型**：ERT + Stacking（Strict OOF）
- **采样策略**：嵌套采样（小比例为大比例的严格子集）
- **重复次数**：8
- **CV策略**：P-T 分层，稀疏 bins 自动合并；`n_splits_used=5`（无降折）

### 结果汇总

| 样本比例 | N_train | 模型 | T_RMSE (°C) | T_RMSE_std | T_95%CI | P_RMSE (kbar) | P_RMSE_std | P_95%CI |
|---------|---------|------|-------------|------------|---------|---------------|------------|---------|
| 20% | 463 | **ERT** | **47.10** | 1.15 | [46.14, 48.07] | **2.89** | 0.14 | [2.78, 3.01] |
| 20% | 463 | Stacking | 47.82 | 1.25 | [46.77, 48.86] | 2.93 | 0.16 | [2.80, 3.07] |
| 40% | 792 | **ERT** | **40.55** | 0.75 | [39.93, 41.18] | **2.48** | 0.07 | [2.43, 2.54] |
| 40% | 792 | Stacking | 41.45 | 0.82 | [40.76, 42.13] | 2.53 | 0.07 | [2.47, 2.59] |
| 60% | 1139 | **ERT** | **35.63** | 0.50 | [35.22, 36.05] | **2.24** | 0.04 | [2.21, 2.27] |
| 60% | 1139 | Stacking | 36.42 | 0.65 | [35.87, 36.96] | 2.28 | 0.04 | [2.25, 2.32] |
| 80% | 1468 | **ERT** | **33.05** | 0.55 | [32.59, 33.51] | **2.07** | 0.03 | [2.05, 2.09] |
| 80% | 1468 | Stacking | 33.67 | 0.55 | [33.21, 34.14] | 2.12 | 0.02 | [2.10, 2.14] |
| **100%** | **1730** | **ERT** | **31.37** | **0.34** | [31.09, 31.66] | **1.97** | **0.02** | [1.95, 1.99] |
| 100% | 1730 | Stacking | 32.16 | 0.30 | [31.91, 32.42] | 2.02 | 0.03 | [1.99, 2.05] |

### 关键发现

1. **ERT 全面优于 Stacking**：各样本量下 RMSE 均更低
   - 20%：T +0.71°C，P +0.04 kbar
   - 100%：T +0.79°C，P +0.05 kbar
2. **方差特征**：T_std 约 0.34–1.25°C，P_std 约 0.02–0.16 kbar，8 次重复已能稳定估计
3. **样本量收益**：ERT T_RMSE 47.10 → 31.37°C（↓33%），P_RMSE 2.89 → 1.97 kbar（↓32%），80%→100% 边际收益较小
4. **分层合并状态**：bins_merged_ratio=1.0（各比例均发生稀疏合并），n_effective_bins_mean 13→84，`n_splits_used=5`

### 结论

- **模型复杂度边界已确认**：在 ~2000 样本规模下，Stacking 的额外复杂度未带来性能提升
- **ERT 仍是更稳健的选择**：性能更好、训练更快、调参更简单
- **数据规模收益趋于饱和**：80%→100% 的边际改善已较小，更多数据的收益有限

## 分析误差传播（V7.3，独立测试集口径）

> 运行日期：2026-02-04 | 实验：E07_ert_augmented_none_liq | corr=none | n_mc=1000 | 测试集=349（P-T网格采样，Liquid 18特征）

### 结果摘要

| 目标 | test_RMSE | MBE | analysis_std_mean | analysis_2mad_mean | interval_68_mean | interval_90_mean | propagated/test |
|------|-----------|-----|-------------------|--------------------|------------------|------------------|-----------------|
| T | 54.65 °C | +5.24 °C | 6.83 °C | 8.96 °C | 13.34 °C | 22.18 °C | 12.5% |
| P | 3.23 kbar | +1.07 kbar | 0.446 kbar | 0.580 kbar | 0.866 kbar | 1.449 kbar | 13.8% |

### 解释与结论

1. **口径区分**：analysis_* 仅反映输入扰动导致的离散度；total_* 为未扰动基线误差（含模型误差）。
2. **贡献比例**：propagated/test 约 12–14%，说明测量误差是显著来源但非主导。
3. **偏差方向**：MBE 为正，表示整体存在低估。
4. **可追溯性**：meta 中记录 test_indices_sampled 与 test_sample_stats（y_T/y_P 分布统计）。

## 稳定性验证（V7.3，1000 repeats）

> 运行日期：2026-02-05 | 实验：E07_stability_nj4 | ERT + Augmented + None + Liquid | repeats=1000 | stability_test_size=0.3 | CV=10折

### 结果摘要

| 目标 | RMSE_mean | RMSE_std | RMSE_95%CI | MAE_mean | MBE_mean | R²_mean |
|------|-----------|----------|------------|----------|----------|---------|
| T | 57.66 °C | 1.99 °C | [57.53, 57.78] | 37.65 °C | +6.26 °C | 0.908 |
| P | 3.48 kbar | 0.13 kbar | [3.48, 3.49] | 2.30 kbar | +1.15 kbar | 0.839 |

### 解释与结论

1. **稳健性范围**：RMSE 标准差约 3–4%，显示在 70% 训练子样本扰动下性能波动有限。
2. **偏差方向**：MBE 为正，说明整体存在低估，与 slope>1 的压缩趋势一致。
3. **稀疏合并**：n_splits_used 全为 10；bins_merged_rate=1.0（n_bins_raw≈207 → n_bins_merged=24），子样本稀疏合并为常态。

---
# 第四部分：变更日志

### V7.4.0 - 保留图件流程收敛与可视化一致性

- 仅保留并收敛 5 张核心图件输出：`pt_sampling_bias_overview`、`parity_compare_TP`、`learning_curve_TP`、`E07_ert_augmented_none_liq_TP_SHAP_combined_ExtraTreesRegressor`、`correction_delta_scatter_TP`。
- 新增 `--selected-only` 流程：仅生成上述保留图件，并将输出统一落盘到 `results/{fig-subdir}`（默认 `results/figures`）。
- SHAP 绘图迁移为 BayesV5 参考画法：`plot_combined_shap_summary` 采用 dot+bar 叠加；离线流程默认去除总标题，保留坐标轴标题。
- SHAP 合并策略固定为 TP 主图输出，默认清理中间 T/P 单图，避免中间产物残留。
- `correction_delta_scatter_TP` 重构为参考实现样式：联合散点 + 平滑趋势 + 95% 区间 + 分段边界（q33/q67）；背景统一纯白，默认仅输出 PNG。
- 新增/扩展离线绘图参数：`--enable-shap`、`--shap-max-samples`、`--shap-bg-k`、`--shap-force`、`--correction-delta-exp-id`。
- 清理测试输出约定：不再将最终图件落到测试子目录，统一汇入 `results/figures`。

### V7.3.0 - 误差链路统一与随机隔离

- MBE 定义统一为 `y_true - y_pred`（正值代表低估）
- 删除 `clip_min` 与 `error_model` 配置，移除多路径分支
- 完全移除 Group 逻辑，协议/学习曲线/稳定性工具不再走分组分支
- 默认按特征数推断 `feature_names`（9/18），非 9/18 必须显式提供，禁止默认 3% 降级
- 温压随机性隔离：P 目标种子显式偏移（offset=1000）
- 误差传播链路补齐：按 T/P 目标独立训练与采样，输出 summary/samples 文件；对比口径统一为独立测试集 RMSE，支持 corr-module 与自定义特征名
- 版本号更新至 `v7.3.0`

### V7.2.0 - 配置单一源与指标统一

- 移除 `config.yaml` 覆盖逻辑，配置仅来自 `config.py`
- 工具脚本参数链路与主实验对齐（含 Stacking 细节）
- `compute_all_metrics` 统一迁移到 `src/metrics.py`
- 版本号更新至 `v7.2.0`

### V7.1 - 接口清理与版本同步

- 版本号同步至 `7.2.0`（`src/__init__.py`）
- 移除 `groups` 参数：从所有模块接口中移除未使用的 groups 参数
  - `DataModule.fit_transform()`
  - `ModelModule.fit()`
  - `Pipeline.fit()`
  - `StratifiedCVProtocol.run()`
  - `ExperimentMatrix.__init__()`
- 更新 `main.py`：移除 `ExperimentMatrix` 调用中的 groups 参数
- 整合 `docs/` 文档到 README 后删除 docs 文件夹

### V7.0 - 代码审计与清理

- 移除 `main.py` 中无效参数 `run_random_split=False`
- 审计确认种子派生机制（`seed + fold_idx`）正确实现
- 审计确认 OOF 校正策略文档与代码一致
- 审计确认 `_infer_feature_names()` 已在基类统一实现

## V6 系列（2026-02）

### V6.4 - EPMA 误差模型重构

**核心变更**：rel_err 计算方式从"按数值阈值"改为"按氧化物列名映射"

- 新增 `src/perturbation.py`：共用扰动模块，统一数据增强和误差传播的扰动逻辑
- 新增 `src/perturbation.py`：按列名映射的 EPMA 误差配置与统一扰动逻辑
- 修复 TiO2 在高含量时被错误分配 3% 误差的问题

**误差映射规则**：
- 主量元素（SiO2, Al2O3, FeO, MgO, CaO）：3%
- 低含量元素（TiO2, MnO, Na2O, Cr2O3, K2O）：8%

### V6.3 - 分析误差传播工具

- 重命名 `run_mc_uncertainty.py` → `run_error_propagation.py`
- 设计理念：评估 EPMA 分析误差经模型放大后的输出离散度
- 新增指标：`analysis_std`, `analysis_2mad`, `propagated_vs_test_ratio`（原 `propagated_vs_cv_ratio`）

### V6.2 - 稀疏分组合并

- 学习曲线与稳定性测试支持稀疏 bins 自动合并
- 100% 样本使用完整 10 折 CV

### V6.1 - 配置集中化

- `config.py` 成为项目唯一配置来源
- 新增 CatBoost GPU 控制（`task_type: auto/CPU/GPU`）

### V6.0 - 架构重构

- 新增 `config.py`：集中配置管理 + 版本信息收集
- 新增 `src/logger.py`：统一日志模块
- 新增 `tests/`：正式测试框架（7 个测试文件，87 用例）

---

## 历史版本摘要

| 版本   | 日期 | 主要变更 |
|------|------|----------|
| V7.4.0 | 2026-02-24 | 保留5图流程、SHAP画法迁移、Correction Delta白底PNG、输出统一到figures |
| V7.3.0 | 2026-02-03 | 误差链路统一、随机隔离、特征名严格推断 |
| V7.2.0 | 2026-02-03 | 配置单一源、参数链路统一、指标口径合并 |
| V7.1 | 2026-02-02 | 接口清理：移除未使用的 groups 参数、版本号同步至 7.2.0 |
| V7.0 | 2026-02-02 | 代码审计与清理：移除无效参数、确认架构一致性 |
| V6.5 | 2026-02-02 | 配置单一源重构、MBE方向统一、随机性修复、参数链路完整传递 |
| V5.5 | 2026-01-30 | 修复 viz.py 箱线图和热力图 bug |
| V5.4 | 2026-01-30 | 学习曲线嵌套采样，100% 样本使用完整 10 折 |
| V5.3 | 2026-01-30 | 24 组实验完整重跑，确认 E07 最佳 |
| V5.2 | 2026-01-30 | ERT vs Stacking 学习曲线实验（30 重复） |
| V5.1 | 2026-01-28 | 论文图件函数（图 3-2 至 3-6） |
| V5.0 | 2026-01-28 | 实验矩阵重构，新增学习曲线工具 |
| V4.x | 2026-01-21 | StratifiedCVProtocol，固定测试集 |
| V3   | 2026-01-19 | M1 对齐 EPMA 误差模型，M3 改用 Segmented |
| V2   | 2026-01-19 | 双特征集支持，P-T 网格分层 |
| V1   | 2026-01-18 | 初始版本，12 个基础实验 |

---

## 参考文献

Jorgenson et al. (2022). A Machine Learning‐Based Approach to Clinopyroxene Thermobarometry: Model Optimization and Distribution for Use in Earth Sciences. *Journal of Geophysical Research*.

---

## License

MIT
