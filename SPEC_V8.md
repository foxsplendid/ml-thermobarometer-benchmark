# SPEC V8 — ml-thermobarometer-benchmark

> 本文件描述 V8 相对 V7 的设计变更、动机、实现细节与验证方法。每条改动均附 **Why / How / Verification / Back-compat** 四要素,便于后续操作与审稿追溯。
>
> 实施顺序: **第 1 步** S6 + H1 + H2 + 回归测试 → **第 2 步** S1/S2 → **后续** H3–H7 / S3–S5。

---

## 0. 总览

| ID | 类别 | 名称 | 风险 | 影响数值 |
|----|----|----|----|----|
| S1 | 科学 | 校正器嵌套评估 | 中 | 是 (E10–E12 CV) |
| S2 | 科学 | 统一稀疏 bin 处理 | 低 | 是 (轻微) |
| S3 | 科学 | EPMA 扰动非负截断 | 低 | 是 (轻微) |
| S4 | 科学 | 区分缺测与真零 | 低 | 否 (当前数据无 NaN) |
| S5 | 科学 | 校正器边段外推 | 低 | 是 (尾部) |
| S6 | 规范 | 配置真源 + 复现性元数据 | 极低 | 否 |
| H1 | 性能 | 硬件探测层 `runtime.py` | 极低 | 否 |
| H2 | 性能 | 并行预算分配器 | 低 | 否 (期望加速) |
| H3 | 性能 | CatBoost 自动设备选择 | 低 | 否 |
| H4 | 性能 | 模型快速档 | 低 | 否 (`--quick` 才用) |
| H5 | 性能 | 增广内存/权重模拟 | 中 | 视开关 |
| H6 | 性能 | 缓存层 | 低 | 否 |
| H7 | 性能 | Stacking 内层可选并行 | 低 | 否 |

---

## 1. 工作目录约定 (2026-06-11 起改为分支模式)

- **仓库**: `D:/Code/Jupyter/ml-thermobarometer-benchmark/`,git 管理。
- **`main` 分支**: V7 终态,**受保护——禁止直接提交**。
- **`V8` 分支**: 所有 V8 改动在此分支发生 (初始提交 778bfe9 来自已删除的
  `ml-thermobarometer-benchmark-v8/` 目录)。
- **结果隔离**: `results/` 是 V7 黄金基线 (回归测试的对照),V8 完整重跑前不覆盖;
  快速测试落 `results_test/`。
- **运行环境**: `D:\anaconda3\envs\cpx\python.exe` (catboost 1.2.7);base Anaconda 无 catboost。
- **依赖**: 与 V7 共用 `requirements.txt`,新增依赖在本文件 §10 列出。

---

## 2. S6 — 配置真源与复现性元数据

### Why
- V7 代码注释和测试引用了不存在的 `config.yml`,造成"双真源"错觉。
- `results/config_used.yaml` 没有持久化代码状态(git commit / file sha256),无法把结果绑回具体代码版本。

### How
1. **`src/repro.py`** (新增):
   - `code_fingerprint(root) -> dict`: 对 `src/**/*.py` + `main.py` + `config.py` 计算 sha256,返回 `{file: sha256}`。
   - `combined_sha(fingerprint) -> str`: 串联后再 sha256,给出单一 16 字符指纹。
   - `git_state(root) -> dict`: 已存在于 `config.get_version_info`,迁入此模块统一。
2. **`config.py`**:
   - 删除所有"sync with config.yml"注释。
   - `get_version_info()` 调用 `repro.code_fingerprint` + `repro.combined_sha`。
3. **`protocol.ExperimentMatrix.save_config`**:
   - 在 `config_used.yaml` 添加顶层字段 `code_fingerprint` 和 `code_sha`。
4. **`src/model_modules.py`**:
   - 删除 `_CATBOOST_DEFAULTS` 上方关于 `config.yml` 的过时注释,改为"single source: `config.CatBoostConfig`"。

### Verification
- `python -c "from src.repro import code_fingerprint, combined_sha; ..."` 输出稳定指纹。
- `pytest tests/test_repro.py` 验证两次调用指纹一致、改一个文件后变化。

### Back-compat
- 完全向后兼容; 仅新增字段。

---

## 3. H1 — 硬件探测层 `src/runtime.py`

### Why
- V7 各模块独立调用 `os.cpu_count()`,无法协调跨进程线程数,容易超订。
- GPU 探测散落 (`catboost.utils.get_gpu_device_count`)。

### How
新增 `src/runtime.py`,导出冻结 dataclass + 模块级缓存单例:

```python
@dataclass(frozen=True)
class Runtime:
    n_physical_cores: int     # psutil.cpu_count(logical=False), 失败回退 os.cpu_count()
    n_logical_cores: int
    has_cuda_gpu: bool
    n_gpu_devices: int
    gpu_mem_gb: float | None
    ram_gb: float | None      # psutil 不可用时为 None
    platform: str             # sys.platform

def get_runtime() -> Runtime: ...   # lazy + cached
def runtime_summary_str() -> str: ...
```

GPU 探测优先级: `torch.cuda.is_available` → `catboost.utils.get_gpu_device_count` → 否则 False。所有探测均 `try/except` 静默回退,不阻塞模块加载。

### Verification
- `python -m src.runtime` 打印检测结果。
- `tests/test_runtime.py`: 断言字段类型、缓存幂等、即使 GPU 探测失败也能返回。

### Back-compat
- 纯新增模块,不修改既有调用。

---

## 4. H2 — 并行预算分配器

### Why
- V7 用 `ML_N_JOBS` 单环境变量,所有模型一视同仁。Stacking 在内层 5 折 × 3 基模型每个都用满核时严重超订,反而比 4 线程慢。
- 用户当前**同时跑其他 CC 进程**,需要给底线预留资源。

### How
1. **`src/runtime.py`** 增加:
   ```python
   def suggest_n_jobs(context: str = "model", reserve_cores: int = 0) -> int:
       """
       context:
         'model'        — 单模型主路径; 默认 cores - reserve_cores
         'inner_loop'   — Stacking 内层基模型; 默认 cores - reserve_cores
         'cross_proc'   — 多进程编排器单进程内部; 默认 max(1, cores // env('ML_OUTER_PROCS', 1))
       reserve_cores 可由 ENV ML_RESERVE_CORES 全局指定 (默认 0)。
       上界由 ENV ML_N_JOBS 强制 (兼容 V7)。
       下界 1。
       """
   ```
2. **`src/model_modules.py`**:
   - `_get_default_n_jobs()` 改为 `return runtime.suggest_n_jobs("model")`,保留 `ML_N_JOBS` 兼容路径。
3. **环境变量约定** (新增,文档化):
   - `ML_N_JOBS`: 单模型最大线程数 (V7 旧约定,优先生效)。
   - `ML_RESERVE_CORES`: 给系统/其他进程预留的核数 (V8 新增)。默认 0。**用户多任务并发场景建议 `set ML_RESERVE_CORES=2`**。
   - `ML_OUTER_PROCS`: 外层并行进程数,告诉分配器把核分摊给多少个 sibling 进程。默认 1。
4. **CatBoost 路径**: 若用户未传 `thread_count`,在 `fit()` 时经
   `_resolve_runtime_params(n_samples)` 注入 `suggest_n_jobs("model")` (仅 CPU 路径;
   M3/H3 将设备决策推迟到 fit,因此线程注入也随之后移)。

### Verification
- `tests/test_runtime.py::test_suggest_n_jobs_respects_reserve`:
  - 设 `ML_RESERVE_CORES=physical`, 期望返回 1。
  - 设 `ML_N_JOBS=2`, 期望返回 ≤ 2。
- 手动: `ML_RESERVE_CORES=2 python main.py --test`,日志打印生效核数。

### Back-compat
- `ML_N_JOBS` 行为不变; 不设新变量则等价于 V7 默认。

---

## 5. S1 — 校正器嵌套评估

### Why (审查 #1 详述)
V7 `StratifiedCVProtocol.run` 用全部 OOF 预测拟合校正器,然后作用回**同一批**预测计算 fold metrics——in-sample 评估,系统性低估校正后误差。E10–E12 的 CV 改善被高估。

### How
1. **`protocol.StratifiedCVProtocol.__init__`** 增加参数:
   ```python
   nested_correction: bool = True          # V8 默认 True
   inner_correction_cv: int = 5            # 内层折数
   ```
2. **`StratifiedCVProtocol.run`** 流程:
   - **阶段 A** (不变): 外层 10 折,收集 `oof_pred_raw`。
   - **阶段 B** (新): 若 `nested_correction and corr_module is not NoCorrection`:
     - 对每个**外层验证折** `k`,在外层训练折索引 (`train_idx_k = all - val_idx_k`) 内做 `inner_correction_cv` 折切分,得到内层 OOF 预测 `inner_oof_k`,用它 + 真值拟合 `corr_model_k`。
     - 用 `corr_model_k` 校正**外层验证折**的 `y_pred_raw[val_idx_k]`,得到 `y_pred_corr`。
   - **阶段 C** (兼容): 若 `nested_correction=False`,走 V7 路径(全局 OOF 拟合)。
   - **聚合校正器** (供持久化和留出测试集使用): 仍在全部 OOF 上拟合一个最终 `corr_model`,保存到 `results['corr_model']`——这是给 test set 用的,留出评估本身不受影响。
3. **新字段** 写入 `results['summary']`:
   - `correction_eval_mode`: `"nested" | "in_sample"`
   - `correction_inner_cv`: 内层折数
4. **`ExperimentMatrix.run_experiments`**: 暴露 `nested_correction` 参数,默认 True。
5. **重要细节**: 外层验证折内层 CV 时,继承外层的 `stratify_labels[train_idx_k]`; 若内层稀疏到无法 K 折分层,自动回退到 KFold。

### Cost
- 额外训练成本 ≈ `inner_correction_cv` × `n_outer_folds` 次基模型拟合。10 外 × 5 内 = 50 次 vs V7 的 10 次,**5×**。
- 缓解: 仅当 `corr_module != NoCorrection` 时启用 (E10–E12 才付出代价,E01–E09 零开销)。

### Verification
- **数值回归**: E01–E09 (none 校正) 在 V8 跑出的数字必须 bit-identical 到 V7 (走相同路径)。
- **方向性测试**: `tests/test_nested_correction.py` 构造一个偏差明显的合成数据集,断言 nested 评估的 RMSE ≥ in-sample 评估的 RMSE。
- **可复现 V7**: `nested_correction=False` 跑 E10–E12,数字应 bit-identical 到 V7。

### Back-compat
- 通过 `nested_correction=False` 完全复现 V7。论文 V7 数字仍可重跑。

---

## 6. S2 — 统一稀疏 bin 处理

### Why (审查 #2)
V7 主基准对单样本 P-T bin 不合并,直接喂给 `StratifiedKFold(10)`,sklearn 内部告警,分层退化; 子工具 (`run_stability`, `run_learning_curve`) 反而做了 `_merge_sparse_bins`。两者不一致。

### How
1. **`StratifiedCVProtocol.__init__`** 增加:
   ```python
   merge_sparse_bins: bool = True
   min_samples_per_bin: int | None = None   # 默认 = n_splits
   ```
2. **`run()`** 在切分前调用 `_merge_sparse_bins(stratify_labels, min_samples_per_bin or n_splits)`。
3. **`ExperimentMatrix.run_experiments`** 暴露这两参数,默认 True。
4. **日志**: 合并前后报告 bin 数变化。

### Verification
- 跑主基准前后比对: 合并应消除 sklearn "least populated class" 告警。
- E01 数值会有**轻微变化** (折成员变化),纳入 V8 的新基线,文档说明。

### Back-compat
- `merge_sparse_bins=False` 复现 V7 行为。

---

## 7. 回归测试 `tests/test_regression_vs_v7.py`

### Why
确保 H1/H2/S6 等"不该改变数字"的改动确实没改变 E01–E09 的输出。

### How
- 读 V7 `results/metrics_summary.csv` (V8 复制过来的镜像) 作为黄金参考。
- 测试矩阵: 只跑 E01 Liquid (单实验,2–3 分钟)。
- 断言 V8 在**等价配置**下 (`nested_correction=False, merge_sparse_bins=False`) 跑出的 `T_rmse_mean / P_rmse_mean / T_r2_mean / P_r2_mean` 与 V7 偏差 < 1e-6。
- 测试体内强制 `ML_RESERVE_CORES=0, ML_N_JOBS=2` 以稳定时序对种子的影响 (sklearn 在不同线程数下树构造有时数值微抖,严格 bit-identical 需要单线程; 因此实际阈值放宽到 1e-4 相对误差,文档说明)。

### Mark
- `@pytest.mark.slow`,需要 `pytest -m slow` 显式触发。
- 默认 CI 路径只跑快测 (`pytest.ini` 注册 marker 并设 `addopts = -m "not slow"`)。
- 状态: 已交付 `tests/test_regression_vs_v7.py` (E01 Liquid,`nested_correction=False,
  merge_sparse_bins=False`,相对容差 1e-4)。

---

## 8. 文件改动清单 (第 1 步范围)

| 文件 | 动作 |
|----|----|
| `src/runtime.py` | 新增 (H1, H2) |
| `src/repro.py` | 新增 (S6) |
| `src/model_modules.py` | 改 `_get_default_n_jobs`,清理注释 (H2, S6) |
| `config.py` | `get_version_info` 调用 `repro`,清理注释 (S6) |
| `src/protocol.py` | `save_config` 增字段; `run` 增 nested + merge 参数 (S1, S2, S6) |
| `tests/test_runtime.py` | 新增 |
| `tests/test_repro.py` | 新增 |
| `tests/test_nested_correction.py` | 新增 |
| `tests/test_regression_vs_v7.py` | 新增,`@pytest.mark.slow` |
| `SPEC_V8.md` | 本文件 |
| `CHANGES_FROM_V7.md` | 新增,逐 commit 记 |

---

## 9. 实施顺序与里程碑

### Milestone M1 — 基础设施 (完成)
- [x] SPEC_V8.md (本文件)
- [x] H1 `runtime.py` + 单测
- [x] H2 `suggest_n_jobs` + 接入 model_modules + 单测
- [x] S6 `repro.py` + 清理 config 注释 + 接入 save_config + 单测
- [x] 回归测试 `tests/test_regression_vs_v7.py` (M4 批补交)
- [x] 跑现有 `pytest tests/` 确认无回归

### Milestone M2 — 科学修正 (代码完成,V8 基线待跑)
- [x] S1 nested correction
- [x] S2 unified sparse-bin merge
- [ ] 跑一次完整主基准得到 V8 基线,与 V7 对比写入 `CHANGES_FROM_V7.md`

### Milestone M3 — 性能优化 (H3 完成,其余移至 M3.2)
- [x] H3 CatBoost 设备策略
- [ ] H4 模型快速档 (M3.2)
- [ ] H5 增广内存优化 (M3.2)
- [ ] H6 缓存层 (M3.2)
- [ ] H7 stacking 内层并行 (M3.2)

### Milestone M4 — 数据/校正语义 (完成,详见 §13–15)
- [x] S3 EPMA 非负截断
- [x] S4 缺测/零区分
- [x] S5 校正器边段外推

---

## 10. 新增依赖

- `psutil >= 5.9` — 物理核数 + 内存检测 (H1)
- (可选) `pynvml` — GPU 显存检测; 没有也能跑

将在 `requirements.txt` 第 1 步末尾追加。

---

## 11. 并发与硬件礼让约定 (重要)

用户当前**多 CC 进程并发**。所有训练入口在 V8 中遵守 (M4 批全部落地):

1. 任何 CV 跑前打印 `runtime_summary_str()` 与 `effective n_jobs`
   (`main.py` 完整横幅; `tools/run_stability.py`、`tools/run_learning_curve.py`、
   `tools/run_error_propagation.py` 打印硬件摘要行)。
2. 默认 `ML_RESERVE_CORES=0` 不改变现状; README 建议并发场景设 `2`。
3. 测试套件: `tests/conftest.py` 对 `ML_N_JOBS` 做 `setdefault("2")`,
   避免本地开发把机器跑满 (单测内 monkeypatch 不受影响)。
4. `main.py --test`: 对 `ML_N_JOBS` 做 `setdefault("4")` (横幅打印前生效)。
5. 完整 `main.py` 不强制覆盖,尊重用户环境。

---

## 12. 评审清单 (给后续维护者)

提 PR 前对照:
- [ ] 新代码无对 `config.yml` 的引用
- [ ] 所有 `os.cpu_count()` 调用迁移到 `runtime.suggest_n_jobs`
- [ ] 任何改变 CV 路径的改动都暴露开关 + 默认值 + back-compat 测试
- [ ] 新跑产出的 `config_used.yaml` 含 `code_sha` + `code_fingerprint` + `runtime` 字段
  (注意: `results/config_used.yaml` 当前是 V7 时代工件,完整重跑后才会携带新字段)
- [ ] `CHANGES_FROM_V7.md` 已更新本次改动

---

## 13. S3 — EPMA 扰动非负截断 (M4,已完成)

### Why
`epma_perturb` 的乘性高斯噪声 (σ = rel_err·|x|) 理论上可产生负氧化物 wt%,
物理上不可能。默认 3–8% 相对误差下负值概率为 Φ(-1/rel_err) < 4e-36 (永不触发),
但用户自定义大误差 (>~0.2) 的鲁棒性实验会真实出现负值。

### How
- `epma_perturb(..., clip_negative=True)`: 加噪后 `np.maximum(x, 0)`。
  RNG 流不变,默认误差下逐位等同 V7。
- `perturbation_with_repeats` 同样新增 `clip_negative=True`,两个分支均截断。
- `tools/run_error_propagation.py` 的 ep_meta JSON 说明从 `no_clip` 改为 `clip`
  (历史结果文件中的旧说明不改——它们如实描述当时的行为)。
- 弃选方案: 截断正态重采样/对数正态会改变 RNG 流,破坏与 V7 的逐位可复现性,收益为零。

### Verification
`tests/test_perturbation.py` (5 用例): 大误差下截断生效且未截断路径确实产生负值、
默认误差下 clip/no-clip 逐位相等、零值列保持零、两分支形状与非负、
截断概率 ≈ Φ(-2) 点质量检验。

### Back-compat
`clip_negative=False` 完全复现 V7。默认 True 在默认误差下数值零影响。

---

## 14. S4 — 区分缺测与真零 (M4,已完成)

### Why
V7 在 `main.load_data` 用 `np.nan_to_num(X, nan=0.0)` 把缺测静默填 0,
将"未测量"与"真零浓度"混为一谈。实测 `input.csv` 特征列 **0 个 NaN**
(该行为当前是恒等变换),但大量字面 0 是上游"未分析/低于检出限"约定
(Cr2O3.cpx 1102/2079 行、MnO.liq 311、MnO.cpx 296、K2O.liq 177 等),
这一信息在上游已不可恢复。

### How
- 删除 `nan_to_num`;特征或目标含 NaN 时 **loud-fail** (ValueError 列出列名与计数),
  错误信息指明: 本基准无插补策略,树模型/CatBoost 原生容忍 NaN 但 Ridge 与
  StandardScaler 语义不容忍,未来数据含 NaN 须显式决策。
- 文档化零语义: 字面 0 按上游约定保留为数值零;`epma_perturb` 对 0 不加噪
  (σ = rel_err·0 = 0),与该约定自洽。

### Verification
`tests/test_load_data.py` (4 用例): 真实 input.csv 双特征集全有限值守卫、
特征 NaN 报错并点名列、目标 NaN 报错、零值保留。

### Back-compat
当前数据无 NaN → 数组逐位不变,E01–E12 输出不受影响。
唯一行为变化: 未来含 NaN 数据从静默填零变为显式报错 (有意为之)。

---

## 15. S5 — 校正器边段外推 (M4,已完成)

### Why
V7 `SegmentedLinearCorrector.apply` 的段掩码只覆盖 `[boundaries[0], boundaries[-1]]`:
**界外预测完全不校正** (直通),在边界处产生跳变不连续,且树模型收缩偏差最大的
尾部恰好得不到校正。E10 实测界外暴露: P 1/1730 (0.058%)、T 2/1730 (0.116%)。

### How
- `edge_mode='offset'` (V8 默认): 界外以斜率 1 延拓,携带边段在边界处的
  校正量作为常数加性偏移 `f(x) = x + (f_seg(b) − b)`。边界连续、保序、无新超参。
- `edge_mode='raw'`: 复现 V7 直通;持久化 corr_model 字典缺 `edge_mode` 键时
  (V7 时代 joblib) 默认按 `'raw'` 回放。
- `fit` 返回字典与 `get_correction_params` 均携带 `edge_mode`;
  边段样本不足 (model=None) 时该侧保持直通,与既有 None 段语义一致。
- 偏移延拓在 `clip_to_train_range` 截断**之前**执行,截断仍是最终护栏。
- 弃选方案: 输入钳位 (压扁排序信息)、斜率阻尼 (引入需辩护的超参)。

### Verification
`tests/test_correction_modules.py::TestEdgeExtrapolation` (7 用例): 界内两模式
逐位一致、边界连续性 (1e-6)、斜率 1 与保序、raw 模式直通、缺键字典按 raw 回放、
clip 交互、非法 edge_mode 报错。

### Back-compat
`edge_mode='raw'` 复现 V7;旧持久化模型自动回放旧行为。
数值影响仅尾部 (E10–E12 的 ~0.06–0.12% 样本),E01–E09 不受影响。
