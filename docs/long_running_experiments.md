# 长时间实验运行指南

本文档说明如何运行学习曲线和稳定性测试等长时间实验，包括分段运行、断点续跑、结果合并等功能。

---

## 一、运行模式对比

| 模式 | 适用场景 | 命令示例 |
|------|----------|----------|
| **一次性运行** | 小规模测试 | `--repeats 5` |
| **断点续跑** | 可能中断的长时间运行 | `--repeats 30 --resume` |
| **分段运行** | 并行运行或分批提交 | `--repeat-start 0 --repeat-end 9` |

---

## 二、学习曲线 (`run_learning_curve.py`)

### 2.1 快速测试

```bash
# 5次重复，约10分钟
python tools/run_learning_curve.py --repeats 5 --models ert
```

### 2.2 断点续跑（推荐）

```bash
# 首次运行（可能中断）
python tools/run_learning_curve.py --repeats 30 --resume

# 中断后继续（自动跳过已完成任务）
python tools/run_learning_curve.py --repeats 30 --resume
```

**原理**：
- 运行过程中定期保存检查点到 `learning_curve_checkpoint.csv`
- `--resume` 选项会读取检查点，跳过已完成的 (repeat, fraction, model, target) 组合

### 2.3 分段运行

适合在多台机器上并行运行，或分批提交到计算集群。

```bash
# 创建输出目录
mkdir results/lc_segments

# 分段运行（可并行）
python tools/run_learning_curve.py --repeat-start 0 --repeat-end 9 --output-dir results/lc_segments
python tools/run_learning_curve.py --repeat-start 10 --repeat-end 19 --output-dir results/lc_segments
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 --output-dir results/lc_segments

# 合并结果
python tools/run_learning_curve.py --merge-dir results/lc_segments --output-dir results
```

### 2.4 输出文件说明

```
results/
├── learning_curve_checkpoint.csv        # 检查点（断点续跑用，运行完成后可删除）
├── learning_curve_runs_rep_000_009.csv  # 分段结果（仅分段模式）
├── learning_curve_runs.csv              # 完整原始数据
├── learning_curve_summary.csv           # 汇总统计
└── figures/
    ├── learning_curve_T.png             # 温度学习曲线图
    └── learning_curve_P.png             # 压力学习曲线图
```

---

## 三、稳定性测试 (`run_stability.py`)

### 3.1 快速测试

```bash
# 100次重复，约30分钟
python tools/run_stability.py --n-repeats 100 --model-module ert
```

### 3.2 断点续跑（推荐）

```bash
# 首次运行
python tools/run_stability.py --n-repeats 1000 --resume

# 中断后继续
python tools/run_stability.py --n-repeats 1000 --resume
```

**原理**：
- 每 100 次保存检查点（可通过 `--checkpoint-interval` 调整）
- `--resume` 选项会查找最新的检查点或已完成的 test_metrics 文件

### 3.3 分段运行

```bash
# 分5段运行（可并行）
python tools/run_stability.py --repeat-start 0 --repeat-end 199 --exp-id E07
python tools/run_stability.py --repeat-start 200 --repeat-end 399 --exp-id E07
python tools/run_stability.py --repeat-start 400 --repeat-end 599 --exp-id E07
python tools/run_stability.py --repeat-start 600 --repeat-end 799 --exp-id E07
python tools/run_stability.py --repeat-start 800 --repeat-end 999 --exp-id E07

# 合并结果
python tools/run_stability.py --merge-dir results --exp-id E07
```

### 3.4 输出文件说明

```
results/stability/
├── {exp_id}_T_test_metrics.csv              # T目标所有重复结果
├── {exp_id}_P_test_metrics.csv              # P目标所有重复结果
├── {exp_id}_T_test_metrics_rep_000_199.csv  # 分段结果（仅分段模式）
├── {exp_id}_T_checkpoint_rep_000_199_100.csv # 检查点（可删除）
└── stability_summary.csv                    # 汇总统计
```

---

## 四、常见问题

### Q1: 运行中断后如何继续？

**断点续跑模式**（推荐）：
```bash
# 添加 --resume 选项重新运行相同命令
python tools/run_learning_curve.py --repeats 30 --resume
```

**分段模式**：
```bash
# 检查哪些分段已完成
ls results/*.csv

# 只运行未完成的分段
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 ...
```

### Q2: 如何估算运行时间？

| 实验 | 单次时间 | 100次 | 1000次 |
|------|----------|-------|--------|
| 学习曲线（ERT） | ~3s | ~5min | ~50min |
| 学习曲线（Stacking） | ~15s | ~25min | ~4h |
| 稳定性（ERT） | ~2s | ~3min | ~30min |
| 稳定性（Stacking） | ~10s | ~17min | ~3h |

### Q3: 如何并行运行？

使用分段模式，在多个终端/机器上同时运行不同分段：

```bash
# 终端1
python tools/run_learning_curve.py --repeat-start 0 --repeat-end 9 --output-dir results/seg1

# 终端2
python tools/run_learning_curve.py --repeat-start 10 --repeat-end 19 --output-dir results/seg2

# 终端3
python tools/run_learning_curve.py --repeat-start 20 --repeat-end 29 --output-dir results/seg3
```

然后合并：
```bash
python tools/run_learning_curve.py --merge-dir results --output-dir results/final
```

### Q4: 检查点文件可以删除吗？

可以。检查点文件仅用于断点续跑，实验完成后可安全删除：

```bash
# 删除检查点文件
rm results/*_checkpoint*.csv
rm results/learning_curve_checkpoint.csv
```

---

## 五、最佳实践

1. **长时间实验务必使用 `--resume`**：即使预计不会中断，也建议加上此选项
2. **定期检查输出**：查看检查点文件确认进度
3. **分段大小建议**：
   - 学习曲线：10 repeats/段（约30分钟/段）
   - 稳定性：200 repeats/段（约1小时/段）
4. **合并前备份**：分段结果合并前建议备份原始文件
