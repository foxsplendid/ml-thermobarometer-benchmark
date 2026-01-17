# 机器学习温压计标准化评估协议

面向 Jupyter 编排的可复用 Python 工程骨架，用于辉石地质温压计的机器学习模型标准化评估。

## 核心特性

- **GroupKFold 外层评估**：按文献来源 (`Ref`) 分组，严禁随机划分
- **T/P 独立双链路**：温度与压力采用独立建模链路
- **Fold-safe 设计**：标准化、增强、校正均在训练折内完成
- **可审计输出**：每折输出 `metrics.csv` + `preds.parquet`
- **Group-aware Stacking**：inner CV 使用 GroupKFold

## 项目结构

```
ml-thermobarometer-benchmark/
├── config.py                    # 全局配置
├── requirements.txt             # Python 依赖
├── input.csv                    # 校准数据集
├── src/
│   ├── __init__.py              # 模块导出
│   ├── models.py                # CatBoost、Stacking 模型
│   ├── runner.py                # 实验运行器
│   ├── correction.py            # 偏差校正器
│   ├── preprocessing.py         # 数据预处理
│   ├── metrics.py               # 指标计算
│   └── viz.py                   # 可视化
├── notebooks/
│   └── run_experiments.ipynb    # Jupyter 一键运行入口
└── outputs/                     # 实验输出
    └── {exp_name}/
        ├── metrics.csv          # 各折指标
        ├── preds.parquet        # 逐样本预测
        └── summary.csv          # 实验汇总
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键运行（Jupyter）

打开 `notebooks/run_experiments.ipynb`，按顺序执行所有单元格。

### 3. 命令行运行

```python
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner

# 加载数据
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

# 配置实验
config = ExperimentConfig(
    exp_name='exp1_catboost_base',
    model_type='catboost',
    model_params={'iterations': 1000, 'depth': 6},
    augment=False,
    correct=False
)

# 运行
runner = ExperimentRunner(config)
results = runner.run_experiment(
    X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
    groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
)
```

## 实验矩阵

| 实验 | 模型 | 增强 | 校正 |
|------|------|------|------|
| exp1_catboost_base | CatBoost | ❌ | ❌ |
| exp2_catboost_aug | CatBoost | ✅ | ❌ |
| exp3_catboost_aug_corr | CatBoost | ✅ | ✅ |
| exp4_stacking_aug_corr | Stacking | ✅ | ✅ |

## 协议约束

1. **外层评估**: 必须使用 `GroupKFold`，groups 来自 `Ref` 列
2. **训练折隔离**: 标准化、增强、超参、stacking inner CV、偏差校正均在训练折内
3. **偏差校正**: 校正器只能用训练折 OOF 拟合（fold-safe）
4. **Stacking**: inner CV 必须使用 `GroupKFold`

## 输出格式

### metrics.csv

| 列名 | 说明 |
|------|------|
| fold_id | 折索引 |
| rmse_T / mae_T / r2_T | 温度指标 |
| rmse_P / mae_P / r2_P | 压力指标 |

### preds.parquet

| 列名 | 说明 |
|------|------|
| row_id | 样本索引 |
| Ref | 文献来源 |
| T_true / T_pred_raw / T_pred_corr | 温度真值/原始预测/校正预测 |
| P_true / P_pred_raw / P_pred_corr | 压力真值/原始预测/校正预测 |
| fold_id | 折索引 |
| exp_name | 实验名称 |

## License

MIT
