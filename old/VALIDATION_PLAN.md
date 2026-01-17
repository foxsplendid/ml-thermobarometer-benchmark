# 全量代码验证方案

## 验证目标

验证 `ml-thermobarometer-benchmark` 项目的所有核心模块是否正常工作，包括：
1. 模块导入验证
2. 数据加载验证
3. 模型创建验证
4. 单折实验验证
5. 完整实验验证（5组实验矩阵）

---

## 验证步骤

### 步骤1：清空现有输出

```bash
# 删除 outputs 目录下的所有实验结果
rm -rf outputs/exp1_baseline outputs/exp2_aug_only outputs/exp3_corr_only outputs/exp4_aug_corr outputs/exp5_stacking outputs/test outputs/cache outputs/figures
```

### 步骤2：模块导入验证

```python
# 验证所有模块是否可正常导入
print("=" * 60)
print("步骤2：模块导入验证")
print("=" * 60)

try:
    from src import models, runner, metrics, viz, preprocessing, correction
    print("✅ src 包导入成功")
except Exception as e:
    print(f"❌ src 包导入失败: {e}")
    raise

try:
    from src.models import (
        BaseThermoModel, CatBoostWrapper, ExtraTreesWrapper,
        XGBoostWrapper, GroupAwareStacker, get_model, create_default_stacker
    )
    print("✅ models.py 导入成功")
except Exception as e:
    print(f"❌ models.py 导入失败: {e}")
    raise

try:
    from src.runner import ExperimentConfig, ExperimentRunner, SingleTargetRunner
    print("✅ runner.py 导入成功")
except Exception as e:
    print(f"❌ runner.py 导入失败: {e}")
    raise

try:
    from src.correction import (
        BiasCorrector, LinearBiasCorrector, IdentityCorrector,
        PolynomialBiasCorrector, get_corrector
    )
    print("✅ correction.py 导入成功")
except Exception as e:
    print(f"❌ correction.py 导入失败: {e}")
    raise

try:
    from src.preprocessing import (
        load_data, prepare_data, get_feature_cols, augment_data,
        CPX_OXIDE_COLS, LIQ_OXIDE_COLS, CPX_CATION_COLS
    )
    print("✅ preprocessing.py 导入成功")
except Exception as e:
    print(f"❌ preprocessing.py 导入失败: {e}")
    raise

try:
    from src.metrics import (
        rmse, mae, r2, compute_metrics, compute_slope_intercept,
        compute_bias_stats, summarize_folds
    )
    print("✅ metrics.py 导入成功")
except Exception as e:
    print(f"❌ metrics.py 导入失败: {e}")
    raise

try:
    from src.viz import (
        plot_pred_vs_true, plot_residuals, plot_full_report,
        plot_stepwise_rmse_comparison, plot_correction_effect,
        plot_feature_importance, plot_residual_distribution_comparison
    )
    print("✅ viz.py 导入成功")
except Exception as e:
    print(f"❌ viz.py 导入失败: {e}")
    raise

try:
    from config import EXPERIMENT_CONFIGS, N_SPLITS
    print("✅ config.py 导入成功")
    print(f"   - 实验配置数量: {len(EXPERIMENT_CONFIGS)}")
    print(f"   - 默认折数: {N_SPLITS}")
except Exception as e:
    print(f"❌ config.py 导入失败: {e}")
    raise

print("\n✅ 所有模块导入验证通过！")
```

### 步骤3：数据加载验证

```python
print("\n" + "=" * 60)
print("步骤3：数据加载验证")
print("=" * 60)

import os

# 检查数据文件是否存在
data_path = 'input.csv'
if not os.path.exists(data_path):
    print(f"❌ 数据文件 {data_path} 不存在")
    raise FileNotFoundError(data_path)
print(f"✅ 数据文件存在: {data_path}")

# 加载数据
try:
    df = load_data(data_path, encoding='latin-1')
    print(f"✅ 数据加载成功")
    print(f"   - 数据形状: {df.shape}")
    print(f"   - 列数: {len(df.columns)}")
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    raise

# 准备数据
try:
    data = prepare_data(df, feature_mode='cpx_liq')
    print(f"✅ 数据准备成功")
    print(f"   - 特征形状: {data['X'].shape}")
    print(f"   - 温度范围: {data['y_T'].min():.0f} - {data['y_T'].max():.0f} ℃")
    print(f"   - 压力范围: {data['y_P'].min():.2f} - {data['y_P'].max():.2f} kbar")
    print(f"   - 分组数量: {len(set(data['groups']))}")
except Exception as e:
    print(f"❌ 数据准备失败: {e}")
    raise

print("\n✅ 数据加载验证通过！")
```

### 步骤4：模型创建验证

```python
print("\n" + "=" * 60)
print("步骤4：模型创建验证")
print("=" * 60)

# 验证 CatBoost
try:
    model_cb = get_model('catboost', iterations=10, depth=4, silent=True)
    print(f"✅ CatBoostWrapper 创建成功: {type(model_cb).__name__}")
except Exception as e:
    print(f"❌ CatBoostWrapper 创建失败: {e}")
    raise

# 验证 ExtraTrees
try:
    model_et = get_model('extratrees', n_estimators=10, max_depth=4)
    print(f"✅ ExtraTreesWrapper 创建成功: {type(model_et).__name__}")
except Exception as e:
    print(f"❌ ExtraTreesWrapper 创建失败: {e}")
    raise

# 验证 XGBoost
try:
    model_xgb = get_model('xgboost', n_estimators=10, max_depth=4)
    print(f"✅ XGBoostWrapper 创建成功: {type(model_xgb).__name__}")
except Exception as e:
    print(f"❌ XGBoostWrapper 创建失败: {e}")
    raise

# 验证 Stacking
try:
    stacker = create_default_stacker(cache_dir=None, inner_cv=2, use_heterogeneous=True)
    print(f"✅ GroupAwareStacker 创建成功")
    print(f"   - 基模型数量: {len(stacker.base_models)}")
    print(f"   - 元模型: {type(stacker.meta_model).__name__}")
except Exception as e:
    print(f"❌ GroupAwareStacker 创建失败: {e}")
    raise

# 验证校正器
try:
    corrector_linear = get_corrector('linear')
    corrector_identity = get_corrector('identity')
    print(f"✅ 校正器创建成功: LinearBiasCorrector, IdentityCorrector")
except Exception as e:
    print(f"❌ 校正器创建失败: {e}")
    raise

print("\n✅ 模型创建验证通过！")
```

### 步骤5：单折快速验证

```python
print("\n" + "=" * 60)
print("步骤5：单折快速验证")
print("=" * 60)

import config
original_n_splits = config.N_SPLITS
config.N_SPLITS = 1  # 临时改为1折

try:
    # 创建实验配置
    test_config = ExperimentConfig(
        exp_name='test_validation',
        model_type='catboost',
        model_params={'iterations': 50, 'depth': 4},
        augment=False,
        correct=False,
        n_splits=1
    )

    # 运行实验
    runner = ExperimentRunner(test_config)
    results = runner.run_experiment(
        X=data['X'],
        y_T=data['y_T'],
        y_P=data['y_P'],
        groups=data['groups'],
        row_ids=data['row_ids'],
        refs=data['refs']
    )

    # 验证输出文件
    metrics_path = f"outputs/test_validation/metrics.csv"
    preds_path = f"outputs/test_validation/preds.parquet"

    if os.path.exists(metrics_path):
        import pandas as pd
        metrics_df = pd.read_csv(metrics_path)
        print(f"✅ metrics.csv 生成成功")
        print(f"   - 列: {list(metrics_df.columns)}")
    else:
        print(f"❌ metrics.csv 未生成")
        raise FileNotFoundError(metrics_path)

    if os.path.exists(preds_path):
        preds_df = pd.read_parquet(preds_path)
        print(f"✅ preds.parquet 生成成功")
        print(f"   - 形状: {preds_df.shape}")
        print(f"   - 列: {list(preds_df.columns)}")
    else:
        print(f"❌ preds.parquet 未生成")
        raise FileNotFoundError(preds_path)

    # 验证指标键格式
    required_cols = ['T_rmse', 'T_mae', 'T_r2', 'T_slope', 'T_intercept',
                     'P_rmse', 'P_mae', 'P_r2', 'P_slope', 'P_intercept']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"⚠️ 缺失指标列: {missing_cols}")
    else:
        print(f"✅ 所有必需指标列存在")

    print(f"\n✅ 单折快速验证通过！")
    print(f"   - T_RMSE: {results.get('T_rmse_mean', 'N/A'):.2f}")
    print(f"   - P_RMSE: {results.get('P_rmse_mean', 'N/A'):.3f}")

except Exception as e:
    print(f"❌ 单折验证失败: {e}")
    import traceback
    traceback.print_exc()
    raise
finally:
    config.N_SPLITS = original_n_splits  # 恢复原设置
```

### 步骤6：完整实验矩阵验证（可选，耗时较长）

```python
print("\n" + "=" * 60)
print("步骤6：完整实验矩阵验证")
print("=" * 60)

import pandas as pd

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']

results_list = []

for exp_name in exp_names:
    print(f"\n--- 运行 {exp_name} ---")
    try:
        exp_config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
        runner = ExperimentRunner(exp_config)
        results = runner.run_experiment(
            X=data['X'],
            y_T=data['y_T'],
            y_P=data['y_P'],
            groups=data['groups'],
            row_ids=data['row_ids'],
            refs=data['refs']
        )
        results_list.append(results)
        print(f"✅ {exp_name} 完成")
    except Exception as e:
        print(f"❌ {exp_name} 失败: {e}")
        import traceback
        traceback.print_exc()

# 汇总结果
if results_list:
    summary_df = pd.DataFrame(results_list)
    summary_df.to_csv('outputs/summary_all.csv', index=False)
    print(f"\n✅ 完整实验矩阵验证完成！")
    print(f"   - 成功实验数: {len(results_list)}/{len(exp_names)}")
    print(f"   - 汇总文件: outputs/summary_all.csv")

    # 打印结果表
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    display_cols = ['exp_name', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))
```

---

## 预期输出

### 成功验证的输出示例

```
============================================================
步骤2：模块导入验证
============================================================
✅ src 包导入成功
✅ models.py 导入成功
✅ runner.py 导入成功
✅ correction.py 导入成功
✅ preprocessing.py 导入成功
✅ metrics.py 导入成功
✅ viz.py 导入成功
✅ config.py 导入成功
   - 实验配置数量: 5
   - 默认折数: 5

✅ 所有模块导入验证通过！

============================================================
步骤3：数据加载验证
============================================================
✅ 数据文件存在: input.csv
✅ 数据加载成功
   - 数据形状: (2079, 45)
   - 列数: 45
✅ 数据准备成功
   - 特征形状: (2079, 36)
   - 温度范围: 750 - 1380 ℃
   - 压力范围: 0.00 - 21.00 kbar
   - 分组数量: XX

✅ 数据加载验证通过！

...（后续步骤类似）
```

---

## 验证清单

| 步骤 | 验证项 | 预期结果 |
|------|--------|---------|
| 1 | 清空输出目录 | outputs/ 下无旧实验结果 |
| 2 | 模块导入 | 所有模块无报错导入 |
| 3 | 数据加载 | input.csv 正常加载，形状 (2079, 45) |
| 4 | 模型创建 | CatBoost/ExtraTrees/XGBoost/Stacking 创建成功 |
| 5 | 单折验证 | metrics.csv 和 preds.parquet 正常生成 |
| 6 | 完整实验 | 5组实验全部完成，生成 summary_all.csv |

---

## 审批后执行

请审批此验证方案后，我将按步骤执行验证。

验证选项：
1. **快速验证**（步骤1-5）：约2-3分钟
2. **完整验证**（步骤1-6）：约15-30分钟（取决于硬件）

请确认要执行哪个验证级别。
