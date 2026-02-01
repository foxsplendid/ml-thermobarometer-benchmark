# ML Thermobarometer Benchmark - 项目逻辑梳理

## 一、main.py - 主实验入口

### 输入
| 输入项 | 来源 | 说明 |
|--------|------|------|
| `input.csv` | 项目根目录 | 2079 行 CPX-LIQ 校准数据集 |
| `--test` | 命令行参数 | 可选，快速测试模式 |

### 输出
```
results/
├── metrics_summary.csv              # 所有实验的汇总指标（RMSE/MAE/R²等）
├── effect_table.csv                 # 模块效应分析表
├── config_used.yaml                 # 实验配置记录
├── {exp_id}_{T/P}_fold_metrics.csv  # 每折指标（24×2=48个文件）
├── {exp_id}_{T/P}_predictions.parquet # 逐样本预测结果（48个文件）
└── models/
    └── {exp_id}_{target}_model.joblib # 保存的模型（48个文件）
```

### 工作流程
```
1. 加载数据
   └── load_data() → 读取 input.csv，提取特征(X)、目标(y_T, y_P)、分组(groups)

2. 准备数据划分
   └── prepare_splits() → P-T 网格采样生成固定测试集
       ├── compute_pt_edges() → 计算 P-T bin 边界
       ├── assign_pt_bins() → 分配样本到 bins
       └── select_test_indices() → 每个 bin 选 1 个样本作为测试集

3. 获取实验配置
   └── get_experiment_configs() → 生成 24 个实验配置
       └── 12 个基础配置 × 2 种特征集(NoLiquid/Liquid)

4. 按特征集分组运行实验
   └── ExperimentMatrix.run_experiments()
       └── 对每个实验配置：
           ├── 构建 Pipeline(DataModule + ModelModule + CorrectionModule)
           ├── StratifiedCVProtocol 执行 10 折 CV
           │   └── 每折：fit_transform → fit → predict → 评估
           ├── 保存 fold_metrics.csv 和 predictions.parquet
           └── 保存模型到 models/

5. 汇总结果
   ├── 合并所有特征集的结果
   ├── compute_effect_table() → 生成效应分析表
   └── save_config() → 保存配置信息
```

### 实验矩阵设计 (V5)
| 组别 | 实验 | M1 数据 | M2 模型 | M3 校正 | 设计意图 |
|------|------|---------|---------|---------|----------|
| 基线组 | E01-E03 | Raw | ERT/CatBoost/Stacking | None | 基线对照 |
| 传统方法 | E04-E06 | Balanced | ERT/CatBoost/Stacking | None | 传统数据平衡 |
| M2 对比 | E07-E09 | Augmented | ERT/CatBoost/Stacking | None | 模型对比 |
| M3 对比 | E10-E12 | Augmented | ERT/CatBoost/Stacking | Segmented | 校正效果验证 |

---

## 二、tools/run_learning_curve.py - 学习曲线工具

### 功能说明
分析模型在不同训练数据量下的性能变化，论证模型复杂度收益边界（ERT vs Stacking）。

### 输入参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--feature-set` | `liq` | 特征集 (liq/noliq) |
| `--data-module` | `augmented` | 数据模块名称 |
| `--corr-module` | `none` | 校正模块名称 |
| `--models` | `ert stacking` | 待评估模型列表 |
| `--fractions` | `0.2 0.4 0.6 0.8 1.0` | 采样比例列表 |
| `--repeats` | `30` | 每个比例的重复次数（完整运行） |
| `--repeat-start` | None | 分段运行起始 repeat 索引（含） |
| `--repeat-end` | None | 分段运行结束 repeat 索引（含） |
| `--merge-dir` | None | 合并多段 runs 文件后生成汇总 |
| `--n-splits` | `10` | CV 折数 |
| `--seed` | `42` | 随机种子 |
| `--resume` | False | 从检查点断点续跑 |
| `--no-plot` | False | 跳过绘图 |

### 输出
```
results/
├── learning_curve_runs_rep_000_009.csv  # 分段运行：该段的 runs 文件
├── learning_curve_runs.csv              # 完整运行：所有 runs
├── learning_curve_summary.csv           # 汇总统计（均值/标准差/CI）
└── figures/learning_curve_*.png         # 学习曲线图
```

### 分段运行与合并机制
- **分段运行**：使用 `--repeat-start/--repeat-end` 指定 repeat 范围，输出带后缀的 runs 文件
- **合并汇总**：使用 `--merge-dir` 读取所有分段 runs，合并后生成 summary 与图表

### 核心工作流程
```
1. 加载数据与划分（复用 main.py 的 load_data/prepare_splits）
2. 对 repeats × fractions × models × targets 四层循环：
   ├── 为每个 repeat 生成嵌套采样索引（小比例是大比例的严格子集）
   ├── 根据 bins 稀疏程度自动降级 CV 折数
   └── 运行分层 CV，记录指标
3. 保存 runs 文件；若非分段模式则生成 summary 和图表
```

### 关键设计
- **嵌套采样**：同一 repeat 内不同 fraction 的索引严格嵌套，避免随机性引入的伪差异
- **折数自动降级**：稀疏 bins 会被合并，折数按需降低以避免 StratifiedKFold 报错
- **检查点**：`--resume` 可从检查点文件恢复，跳过已完成的任务

---

## 三、tools/run_stability.py - 稳定性测试工具

### 功能说明
评估模型在不同随机训练集划分下的性能稳定性，输出 RMSE/R² 的均值、标准差、置信区间。

### 输入参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-id` | `stability_test` | 实验 ID（输出文件名前缀） |
| `--data-module` | `augmented` | 数据模块名称 |
| `--model-module` | `ert` | 模型模块名称 |
| `--corr-module` | `none` | 校正模块名称 |
| `--feature-set` | `Liquid` | 特征集 |
| `--n-repeats` | `1000` | 重复次数（完整运行默认 1000） |
| `--repeat-start` | None | 分段运行起始 repeat 索引（含） |
| `--repeat-end` | None | 分段运行结束 repeat 索引（含） |
| `--merge-dir` | None | 合并分段文件后生成汇总 |
| `--checkpoint-interval` | `100` | 每 N 次保存检查点 |
| `--resume` | False | 从最新检查点继续 |

### 输出
```
results/stability/
├── {exp_id}_{T/P}_test_metrics_rep_000_199.csv   # 分段运行：该段指标
├── {exp_id}_{T/P}_test_metrics.csv               # 完整/合并后：全部指标
├── {exp_id}_{T/P}_checkpoint_*.csv               # 检查点文件
└── stability_summary.csv                         # 汇总统计
```

### 分段运行与合并机制
- **分段运行**：使用 `--repeat-start/--repeat-end` 指定 repeat 范围，输出带后缀的 test_metrics
- **合并汇总**：使用 `--merge-dir` 读取所有分段文件，去重后生成 summary

---

## 四、tools/run_mc_uncertainty.py - MC 不确定性

### 输入
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-id` | `mc_test` | 实验 ID |
| `--data-module` | `augmented` | 数据模块 |
| `--model-module` | `ert` | 模型模块 |
| `--corr-module` | `none` | 校正模块 |
| `--feature-set` | `Liquid` | 特征集 |
| `--n-mc` | `1000` | MC 采样次数 |
| `--mc-sample-size` | `10` | MC 测试样本数 |

### 输出
```
results/
  uncertainty/
    {exp_id}_mc_T_samples.csv
    {exp_id}_mc_P_samples.csv
    {exp_id}_mc_T_summary.csv
    {exp_id}_mc_P_summary.csv
    {exp_id}_mc_meta.json
```

### 工作流程
```
1. 构建实验配置
   └── _build_config() -> ExperimentConfig

2. 加载数据和划分（复用 main.py）

3. 训练与校正
   └── StratifiedCVProtocol.run() -> corr_model（全量 OOF）

4. MC 不确定性估计
   └── MCUncertaintyEstimator.predict_distribution()
       ├── 对测试子集生成 n_mc 次输入扰动
       └── 计算分位数与校准指标
```

### 用途
- **MC 不确定性**：量化单样本预测的不确定性（分位数区间与校准）

---

## 五、tools/plot_offline_figures.py - 离线绘图工具

### 输入
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--results-dir` | `results` | 结果目录 |
| `--exp-id` | `E07_ert_augmented_none_liq` | 默认实验 ID |
| `--fig-subdir` | `figures/plot_smoke_test` | 图片输出子目录 |
| `--data-path` | `input.csv` | 数据文件路径（用于图 3-2） |

### 输出
```
results/figures/plot_smoke_test/
├── {exp_id}_T_pred_vs_true.png    # 预测-真实散点图
├── {exp_id}_P_pred_vs_true.png
├── {exp_id}_T_residuals.png       # 残差分布图
├── {exp_id}_P_residuals.png
├── {exp_id}_full_report.png       # 完整报告图
├── {exp_id}_T_correction_effect.png # 校正前后对比
├── {exp_id}_P_correction_effect.png
├── {exp_id}_T_fold_rmse.png       # 各折指标对比
├── {exp_id}_P_fold_rmse.png
├── {exp_id}_T_importance.png      # 特征重要性
├── {exp_id}_P_importance.png
├── experiment_summary_heatmap.png # 实验汇总热力图
├── pt_grid_cv_splits.png          # 图 3-2：P-T 网格 CV 示意图
├── feature_set_boxplot_T.png      # 图 3-3：特征集对比箱线图
├── feature_set_boxplot_P.png
├── parity_compare_T.png           # 图 3-4：1:1 预测对比
├── parity_compare_P.png
├── m1_ablation_T.png              # 图 3-5：M1 消融阶梯图
├── m1_ablation_P.png
├── performance_heatmap_T.png      # 图 3-6：性能热力图
├── performance_heatmap_P.png
├── residual_compare_T.png         # 残差分布对比
└── residual_compare_P.png
```

### 工作流程
```
1. 基础绘图（单实验）
   ├── _plot_basic() → 预测散点图、残差图、完整报告、校正效果
   ├── _plot_fold() → 各折指标对比
   └── _plot_importance() → 特征重要性

2. 汇总绘图（全局）
   └── _plot_summary() → 实验汇总热力图

3. 论文图件（V5.1 新增）
   ├── _plot_pt_grid_cv() → 图 3-2：P-T 网格 CV 示意图
   ├── _plot_feature_set_boxplot() → 图 3-3：特征集对比箱线图
   ├── _plot_parity_compare() → 图 3-4：1:1 预测对比
   ├── _plot_stepwise() → 图 3-5：M1 消融阶梯图
   ├── _plot_heatmap_matrix() → 图 3-6：性能热力图
   └── _plot_residual_compare() → 残差分布对比
```

### 特点
- **纯离线**：只读取已有的 results/ 文件，不进行模型训练
- **论文导向**：包含第三章所需的图 3-2 至 3-6

---

## 五、数据流总览

```
                    ┌─────────────────┐
                    │   input.csv     │
                    │  (2079 samples) │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   load_data()   │
                    │  X, y_T, y_P    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ prepare_splits()│
                    │  P-T Grid Split │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   main.py     │   │ run_learning  │   │ run_stability │
│ 24 实验矩阵   │   │  _curve.py    │   │    .py        │
│ 10-fold CV    │   │ 学习曲线分析  │   │ 稳定性测试    │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│ results/      │   │ results/      │   │ results/      │
│ metrics_*.csv │   │ learning_     │   │ stability/    │
│ predictions_* │   │ curve_*.csv   │   │ *_summary.csv │
│ models/*.jlib │   │ figures/lc_*  │   └───────────────┘
└───────┬───────┘   └───────────────┘
        │
        ▼
┌───────────────────────┐
│ plot_offline_figures  │
│    离线绘图工具       │
│  (读取 results/ 绘图) │
└───────────────────────┘
```

---

## 六、关键设计原则

1. **严格防泄露**：所有拟合（标准化、增强、校正）仅在训练折内完成
2. **P-T 分层**：使用 P-T 网格保证验证集覆盖整个温压空间
3. **T/P 独立建模**：温度和压力分别建模、预测、评估
4. **模块化设计**：M1(数据) × M2(模型) × M3(校正) 可独立替换
5. **工具分离**：主实验与绘图/稳定性测试分离，按需运行

---

## 七、模块关系汇总

| 脚本 | 职责 | 依赖 main.py | 是否训练模型 |
|------|------|-------------|-------------|
| `main.py` | 运行 24 组实验矩阵 | - | ✅ 是 |
| `run_learning_curve.py` | 学习曲线分析 | 复用数据加载/划分 | ✅ 是 |
| `run_stability.py` | 稳定性测试 | 复用数据加载/划分 | ✅ 是 |
| `run_mc_uncertainty.py` | MC 不确定性 | 复用数据加载/划分 | ✅ 是 |
| `plot_offline_figures.py` | 离线绘图 | 仅读取 results/ | ❌ 否 |

## 八、典型使用顺序

```bash
# 1. 运行主实验（生成所有结果）
python main.py

# 2. 生成论文图件
python tools/plot_offline_figures.py

# 3. 学习曲线 - 分段运行（推荐）
python tools/run_learning_curve.py --repeat-start 0 --repeat-end 9 --output-dir results/lc
python tools/run_learning_curve.py --repeat-start 10 --repeat-end 19 --output-dir results/lc
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 --output-dir results/lc
# 合并分段结果
python tools/run_learning_curve.py --merge-dir results/lc --output-dir results

# 4. 稳定性测试 - 分段运行（推荐）
python tools/run_stability.py --repeat-start 0 --repeat-end 199 --output-dir results/stability
python tools/run_stability.py --repeat-start 200 --repeat-end 399 --output-dir results/stability
# 合并分段结果
python tools/run_stability.py --merge-dir results/stability --output-dir results

# 5. MC 不确定性分析（可选）
python tools/run_mc_uncertainty.py --model-module ert --n-mc 1000 --mc-sample-size 10
```

---

## 九、长时运行实验机制详解

由于学习曲线和稳定性测试的运行时间较长（数小时至数十小时），本项目设计了 **分段保存** 和 **断点续跑** 机制，确保实验可控、可恢复。

### 9.1 两种运行模式对比

| 模式 | 适用场景 | 特点 |
|------|----------|------|
| **完整运行** | 小规模测试、资源充足 | 一次完成所有 repeats，直接输出 summary |
| **分段运行** | 大规模实验、需中断恢复 | 多次运行各自保存，最后合并为 summary |

### 9.2 学习曲线分段运行详解

#### 运行逻辑流程图
```
┌─────────────────────────────────────────────────────────────────┐
│  学习曲线工具运行逻辑                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │ 判断运行模式                              │                  │
│  │ --merge-dir → 合并模式                   │                  │
│  │ --repeat-start/end → 分段模式            │                  │
│  │ 其他 → 完整模式                          │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     ▼                                           │
│  ┌────────────┬─────┴─────┬────────────┐                       │
│  ▼            ▼           ▼            ▼                       │
│ 合并模式    分段模式    完整模式    断点续跑模式                │
│  │           │           │            │                        │
│  │  读取所有  │ 只运行    │ 运行所有   │ 跳过已完成             │
│  │  分段文件  │ 指定范围  │ repeats    │ 的任务                 │
│  │     │      │    │      │    │       │    │                  │
│  │     ▼      │    ▼      │    ▼       │    ▼                  │
│  │ 合并+汇总  │ 保存带    │ 保存完整   │ 从checkpoint          │
│  │ +绘图      │ 后缀的    │ runs.csv   │ 恢复进度              │
│  │           │ runs文件  │ +summary   │                        │
│  └───────────┴───────────┴────────────┴────────────────────────┤
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 文件命名规则
| 模式 | 输出文件名 | 说明 |
|------|------------|------|
| 分段 | `learning_curve_runs_rep_000_009.csv` | 带 repeat 范围后缀 |
| 完整 | `learning_curve_runs.csv` | 无后缀 |
| 合并 | `learning_curve_runs.csv` + `learning_curve_summary.csv` | 合并所有分段 |

#### 典型使用流程
```bash
# 步骤 1: 分多次运行（可在不同时间/机器执行）
python tools/run_learning_curve.py --repeat-start 0 --repeat-end 9 --output-dir results/lc
python tools/run_learning_curve.py --repeat-start 10 --repeat-end 19 --output-dir results/lc
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 --output-dir results/lc

# 步骤 2: 合并所有分段结果
python tools/run_learning_curve.py --merge-dir results/lc --output-dir results

# 或：使用断点续跑（自动跳过已完成任务）
python tools/run_learning_curve.py --repeats 30 --resume --output-dir results
```

### 9.3 稳定性测试分段运行详解

#### 运行逻辑流程图
```
┌─────────────────────────────────────────────────────────────────┐
│  稳定性测试工具运行逻辑                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────┐                  │
│  │ 判断运行模式                              │                  │
│  │ --merge-dir → 合并模式                   │                  │
│  │ --repeat-start/end → 分段模式            │                  │
│  │ 其他 → 完整模式                          │                  │
│  └──────────────────┬───────────────────────┘                  │
│                     ▼                                           │
│  ┌────────────┬─────┴─────┬────────────┐                       │
│  ▼            ▼           ▼            ▼                       │
│ 合并模式    分段模式    完整模式    断点续跑模式                │
│  │           │           │            │                        │
│  │ 合并所有   │ 只运行    │ 运行所有   │ 从checkpoint          │
│  │ 分段文件   │ 指定范围  │ repeats    │ 恢复进度              │
│  │     │      │    │      │    │       │    │                  │
│  │     ▼      │    ▼      │    ▼       │    ▼                  │
│  │ 去重+汇总  │ 保存带    │ 保存完整   │ 跳过已有的            │
│  │           │ 后缀的    │ metrics    │ repeat_id             │
│  │           │ metrics   │ +summary   │                        │
│  └───────────┴───────────┴────────────┴────────────────────────┤
│                                                                 │
│  检查点机制:                                                    │
│  - 每 checkpoint_interval 次保存一次（默认100）                 │
│  - 文件名: {exp_id}_{T/P}_checkpoint_rep_XXX_YYY_ZZZ.csv       │
│  - --resume 时自动加载最新检查点                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### 文件命名规则
| 模式 | 输出文件名 | 说明 |
|------|------------|------|
| 分段 | `{exp_id}_{T/P}_test_metrics_rep_000_199.csv` | 带 repeat 范围后缀 |
| 完整 | `{exp_id}_{T/P}_test_metrics.csv` | 无后缀 |
| 检查点 | `{exp_id}_{T/P}_checkpoint_rep_XXX_YYY_ZZZ.csv` | ZZZ 为当前完成数 |

#### 典型使用流程
```bash
# 方案 A: 分段运行（推荐用于长时间实验）
python tools/run_stability.py --repeat-start 0 --repeat-end 199 --output-dir results/stability
python tools/run_stability.py --repeat-start 200 --repeat-end 399 --output-dir results/stability
# ...更多分段...
python tools/run_stability.py --merge-dir results/stability --output-dir results

# 方案 B: 断点续跑（自动恢复）
python tools/run_stability.py --n-repeats 1000 --resume --checkpoint-interval 50
# 中断后再次运行相同命令，会自动跳过已完成的 repeats
```

### 9.4 关键设计说明

| 设计点 | 说明 |
|--------|------|
| **幂等性** | 同一 repeat 范围多次运行会覆盖，合并时自动去重 |
| **任务粒度** | 学习曲线以 (repeat, fraction, model, target) 为单位；稳定性以 repeat_id 为单位 |
| **随机种子** | 由 `base_seed + repeat_id` 派生，保证可复现 |
| **输出目录隔离** | 分段运行建议使用独立子目录，避免与最终结果混淆 |


