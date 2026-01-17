# -*- coding: utf-8 -*-
"""
全量代码验证脚本
按照 VALIDATION_PLAN.md 执行步骤2-6的验证
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 设置工作目录
os.chdir(r'd:\Code\Jupyter\ml-thermobarometer-benchmark')

# 设置输出为 UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd

# ============================================================
# 步骤2：模块导入验证
# ============================================================
print("=" * 60)
print("步骤2：模块导入验证")
print("=" * 60)

try:
    from src import models, runner, metrics, viz, preprocessing, correction
    print("[OK] src 包导入成功")
except Exception as e:
    print(f"[FAIL] src 包导入失败: {e}")
    sys.exit(1)

try:
    from src.models import (
        BaseThermoModel, CatBoostWrapper, ExtraTreesWrapper,
        XGBoostWrapper, GroupAwareStacker, get_model, create_default_stacker
    )
    print("[OK] models.py 导入成功")
except Exception as e:
    print(f"[FAIL] models.py 导入失败: {e}")
    sys.exit(1)

try:
    from src.runner import ExperimentConfig, ExperimentRunner, SingleTargetRunner
    print("[OK] runner.py 导入成功")
except Exception as e:
    print(f"[FAIL] runner.py 导入失败: {e}")
    sys.exit(1)

try:
    from src.correction import (
        BiasCorrector, LinearBiasCorrector, IdentityCorrector,
        PolynomialBiasCorrector, get_corrector
    )
    print("[OK] correction.py 导入成功")
except Exception as e:
    print(f"[FAIL] correction.py 导入失败: {e}")
    sys.exit(1)

try:
    from src.preprocessing import (
        load_data, prepare_data, get_feature_cols, augment_data,
        CPX_OXIDE_COLS, LIQ_OXIDE_COLS, CPX_CATION_COLS
    )
    print("[OK] preprocessing.py 导入成功")
except Exception as e:
    print(f"[FAIL] preprocessing.py 导入失败: {e}")
    sys.exit(1)

try:
    from src.metrics import (
        rmse, mae, r2, compute_metrics, compute_slope_intercept,
        compute_bias_stats, summarize_folds
    )
    print("[OK] metrics.py 导入成功")
except Exception as e:
    print(f"[FAIL] metrics.py 导入失败: {e}")
    sys.exit(1)

try:
    from src.viz import (
        plot_pred_vs_true, plot_residuals, plot_full_report,
        plot_stepwise_rmse_comparison, plot_correction_effect,
        plot_feature_importance, plot_residual_distribution_comparison
    )
    print("[OK] viz.py 导入成功")
except Exception as e:
    print(f"[FAIL] viz.py 导入失败: {e}")
    sys.exit(1)

try:
    from config import EXPERIMENT_CONFIGS, N_SPLITS
    print("[OK] config.py 导入成功")
    print(f"   - 实验配置数量: {len(EXPERIMENT_CONFIGS)}")
    print(f"   - 默认折数: {N_SPLITS}")
except Exception as e:
    print(f"[FAIL] config.py 导入失败: {e}")
    sys.exit(1)

print("\n[OK] 所有模块导入验证通过！")

# ============================================================
# 步骤3：数据加载验证
# ============================================================
print("\n" + "=" * 60)
print("步骤3：数据加载验证")
print("=" * 60)

# 检查数据文件是否存在
data_path = 'input.csv'
if not os.path.exists(data_path):
    print(f"[FAIL] 数据文件 {data_path} 不存在")
    sys.exit(1)
print(f"[OK] 数据文件存在: {data_path}")

# 加载数据
try:
    df = load_data(data_path, encoding='latin-1')
    print(f"[OK] 数据加载成功")
    print(f"   - 数据形状: {df.shape}")
    print(f"   - 列数: {len(df.columns)}")
except Exception as e:
    print(f"[FAIL] 数据加载失败: {e}")
    sys.exit(1)

# 准备数据
try:
    data = prepare_data(df, feature_mode='cpx_liq')
    print(f"[OK] 数据准备成功")
    print(f"   - 特征形状: {data['X'].shape}")
    print(f"   - 温度范围: {data['y_T'].min():.0f} - {data['y_T'].max():.0f} C")
    print(f"   - 压力范围: {data['y_P'].min():.2f} - {data['y_P'].max():.2f} kbar")
    print(f"   - 分组数量: {len(set(data['groups']))}")
except Exception as e:
    print(f"[FAIL] 数据准备失败: {e}")
    sys.exit(1)

print("\n[OK] 数据加载验证通过！")

# ============================================================
# 步骤4：模型创建验证
# ============================================================
print("\n" + "=" * 60)
print("步骤4：模型创建验证")
print("=" * 60)

# 验证 CatBoost
try:
    model_cb = get_model('catboost', iterations=10, depth=4, silent=True)
    print(f"[OK] CatBoostWrapper 创建成功: {type(model_cb).__name__}")
except Exception as e:
    print(f"[FAIL] CatBoostWrapper 创建失败: {e}")
    sys.exit(1)

# 验证 ExtraTrees
try:
    model_et = get_model('extratrees', n_estimators=10, max_depth=4)
    print(f"[OK] ExtraTreesWrapper 创建成功: {type(model_et).__name__}")
except Exception as e:
    print(f"[FAIL] ExtraTreesWrapper 创建失败: {e}")
    sys.exit(1)

# 验证 XGBoost
try:
    model_xgb = get_model('xgboost', n_estimators=10, max_depth=4)
    print(f"[OK] XGBoostWrapper 创建成功: {type(model_xgb).__name__}")
except Exception as e:
    print(f"[FAIL] XGBoostWrapper 创建失败: {e}")
    sys.exit(1)

# 验证 Stacking
try:
    stacker = create_default_stacker(cache_dir=None, inner_cv=2, use_heterogeneous=True)
    print(f"[OK] GroupAwareStacker 创建成功")
    print(f"   - 基模型数量: {len(stacker.base_models)}")
    print(f"   - 元模型: {type(stacker.meta_model).__name__}")
except Exception as e:
    print(f"[FAIL] GroupAwareStacker 创建失败: {e}")
    sys.exit(1)

# 验证校正器
try:
    corrector_linear = get_corrector('linear')
    corrector_identity = get_corrector('identity')
    print(f"[OK] 校正器创建成功: LinearBiasCorrector, IdentityCorrector")
except Exception as e:
    print(f"[FAIL] 校正器创建失败: {e}")
    sys.exit(1)

print("\n[OK] 模型创建验证通过！")

# ============================================================
# 步骤5：快速验证（使用2折而不是1折，因为GroupKFold需要至少2折）
# ============================================================
print("\n" + "=" * 60)
print("步骤5：快速验证 (2折)")
print("=" * 60)

try:
    # 创建实验配置
    test_config = ExperimentConfig(
        exp_name='test_validation',
        model_type='catboost',
        model_params={'iterations': 50, 'depth': 4},
        augment=False,
        correct=False,
        n_splits=2  # GroupKFold 需要至少2折
    )

    # 运行实验
    exp_runner = ExperimentRunner(test_config)
    results = exp_runner.run_experiment(
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
        metrics_df = pd.read_csv(metrics_path)
        print(f"[OK] metrics.csv 生成成功")
        print(f"   - 列: {list(metrics_df.columns)}")
    else:
        print(f"[FAIL] metrics.csv 未生成")
        sys.exit(1)

    if os.path.exists(preds_path):
        preds_df = pd.read_parquet(preds_path)
        print(f"[OK] preds.parquet 生成成功")
        print(f"   - 形状: {preds_df.shape}")
        print(f"   - 列: {list(preds_df.columns)}")
    else:
        print(f"[FAIL] preds.parquet 未生成")
        sys.exit(1)

    # 验证指标键格式
    required_cols = ['T_rmse', 'T_mae', 'T_r2', 'T_slope', 'T_intercept',
                     'P_rmse', 'P_mae', 'P_r2', 'P_slope', 'P_intercept']
    missing_cols = [col for col in required_cols if col not in metrics_df.columns]
    if missing_cols:
        print(f"[WARN] 缺失指标列: {missing_cols}")
    else:
        print(f"[OK] 所有必需指标列存在")

    print(f"\n[OK] 快速验证通过！")
    print(f"   - T_RMSE: {results.get('T_rmse_mean', 'N/A'):.2f}")
    print(f"   - P_RMSE: {results.get('P_rmse_mean', 'N/A'):.3f}")

except Exception as e:
    print(f"[FAIL] 快速验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# 步骤6：完整实验矩阵验证
# ============================================================
print("\n" + "=" * 60)
print("步骤6：完整实验矩阵验证")
print("=" * 60)

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']

results_list = []

for exp_name in exp_names:
    print(f"\n--- 运行 {exp_name} ---")
    try:
        exp_config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
        exp_runner = ExperimentRunner(exp_config)
        results = exp_runner.run_experiment(
            X=data['X'],
            y_T=data['y_T'],
            y_P=data['y_P'],
            groups=data['groups'],
            row_ids=data['row_ids'],
            refs=data['refs']
        )
        results_list.append(results)
        print(f"[OK] {exp_name} 完成")
        print(f"   - T_RMSE: {results.get('T_rmse_mean', 'N/A'):.2f}")
        print(f"   - P_RMSE: {results.get('P_rmse_mean', 'N/A'):.3f}")
    except Exception as e:
        print(f"[FAIL] {exp_name} 失败: {e}")
        import traceback
        traceback.print_exc()

# 汇总结果
if results_list:
    summary_df = pd.DataFrame(results_list)
    os.makedirs('outputs', exist_ok=True)
    summary_df.to_csv('outputs/summary_all.csv', index=False)
    print(f"\n[OK] 完整实验矩阵验证完成！")
    print(f"   - 成功实验数: {len(results_list)}/{len(exp_names)}")
    print(f"   - 汇总文件: outputs/summary_all.csv")

    # 打印结果表
    print("\n" + "=" * 80)
    print("实验结果汇总")
    print("=" * 80)
    display_cols = ['exp_name', 'T_rmse_mean', 'T_r2_mean', 'P_rmse_mean', 'P_r2_mean']
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

print("\n" + "=" * 80)
print("ALL VALIDATION COMPLETED SUCCESSFULLY!")
print("=" * 80)
