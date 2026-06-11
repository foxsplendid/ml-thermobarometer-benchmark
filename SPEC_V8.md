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
| H4 | 性能 | 模型快速档 | 低 | 否 (`--test` 快速档才用, §17) |
| H5 | 性能 | 增广内存/权重模拟 | — | 否 (评估后弃置, §16) |
| H6 | 性能 | 缓存层 (外层 CV 拟合复用) | 低 | 否 (默认关, §19) |
| H7 | 性能 | Stacking 内层可选并行 | 低 | 否 (默认关, §18) |

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

### Milestone M2 — 科学修正 (完成)
- [x] S1 nested correction
- [x] S2 unified sparse-bin merge
- [x] 跑一次完整主基准得到 V8 基线,与 V7 对比写入 `CHANGES_FROM_V7.md`
  (2026-06-11,墙钟 ~7.5 h;对比见 CHANGES M2 节"V8 基线实测对比")

### Milestone M3 — 性能优化 (H3 完成,其余移至 M3.2)
- [x] H3 CatBoost 设备策略
- [x] H4 模型快速档 (M3.2,§17)
- [x] H5 增广内存优化 (M3.2,评估后弃置,§16)
- [x] H6 缓存层 (M3.2,外层 CV 拟合复用,§19)
- [x] H7 stacking 内层并行 (M3.2,§18)

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

---

## 16. H5 — 增广内存/权重模拟 (M3.2,评估后弃置)

### Why
M3 遗留项 H5 假设 `AugmentedDataModule` 物化 `(1+n_aug)x` 训练行 (n_aug=15 → 16x float64)
造成内存/时间负担,候选方案为 float32 存储、流式生成、或以样本权重替代复制。M3.2 立项时
先按项目真实数据规模量化,结论是该负担不存在,三个候选方案均不成立,**弃置且不引入任何开关**。

### 实测证据 (2026-06-11,cpx 环境,n_jobs=2)
主基准外层折训练规模 1557 行 (≈1730 × 0.9) × 18 特征 (Liquid 集):

| 量 | 实测值 |
|---|---|
| 增广矩阵 X_all (24912 × 18, float64) | **3.59 MB** (y+w 另 0.40 MB) |
| `fit_transform` 峰值分配 (tracemalloc) | **11.0 MB** |
| `fit_transform` 耗时 (15 次 `epma_perturb` + vstack + scaler) | **11.9 ms** |
| ERT fit (raw 1557 行 / aug 24912 行,项目默认超参) | 0.225 s / **2.43 s** |
| 增广生成占增广管线拟合成本 | **0.5%** |
| 训出的 ERT 模型 pickle 体积 | **37.2 MB** (数据的 10 倍) |

推论:
- **内存大头在树模型 (37 MB),不在数据 (3.6 MB)**;H5 三个方案均不触及模型内存。
- **时间大头是 16x 行数的模型拟合 (10.8x)**——这是"在扰动副本上训练"的科学方法本体,
  工程上不可消除,只能改 `n_aug` (科学决策,不属于性能项)。
- 最重负载 `run_stability` (1000 重复 × 2 目标 × 11 次拟合 = 22000 次): 生成总开销
  ≈ 4.4 分钟,对以模型拟合为主的多小时总时长占比 <1%。即便 `ML_OUTER_PROCS=14` 并发,
  数据峰值合计 ≈ 154 MB,对本机内存可忽略。

### 三个候选方案逐一否决
- **权重模拟**: 科学上不可行。15 份副本的 X 各不相同 (`epma_perturb` 乘性高斯噪声),
  样本权重只能加权既有行,无法表达扰动后的新 X;能被权重化的"精确复制"场景本项目已由
  `BalancedDataModule` 以权重实现。
- **float32 存储**: 每次拟合省约 5 MB、省时 ≈0,却**影响数值**——`scaler.transform` 在
  float32 算术下与"float64 计算后转换"存在末位差,可翻转 ERT 分裂点;CatBoost 特征分箱
  边界也可能移动。即必须配 opt-in 开关 + 回归验证,接口成本远超收益,违反简洁原则。
- **流式生成**: `ExtraTreesRegressor.fit` / `catboost.Pool` 均为批式接口,需要完整矩阵
  驻留内存;流式至多省去 3.6 MB 的临时列表,无意义。

### Verification
无代码改动 → 无新增测试。既有覆盖保持: `tests/test_data_modules.py` 已验证增广形状、
fold_seed 确定性、不同 seed 产生不同扰动。本节实测脚本一次性运行,数字记录于此,不入库。

### Back-compat
不改任何代码、不新增任何参数/环境变量,全部路径逐位不变。若未来 `n_aug` 或数据量增长
≥50x (增广矩阵进入百 MB 级) 再重启本项,届时优先复测本节表格。stability 工具墙钟时间的
真实杠杆是外层进程并行 (`ML_OUTER_PROCS`) 与 §18 的 stacking 内层并行,不在 H5。

---

## 17. H4 — 模型快速档 (M3.2)

### Why
- `main.py --test` 是唯一的端到端冒烟入口,但它沿用科学档模型参数 (ERT 200 树/深 15、
  CatBoost 1000 迭代/深 6)。实测 (1730 行 Liquid 18 特征、2 线程、机器有底载): CatBoost
  单拟合 2.31 s、ERT 0.27 s;quick test 共 24 次拟合 (E01/E02 × 2 特征集 × 2 目标 ×
  [2 折 CV + 1 全量拟合]),纯拟合预算 ~31 s,是 quick test 墙钟的大头。
- 冒烟测试只验证管线连通与指标有限性,不消费数值质量;为它支付科学档训练成本是纯浪费。
- 范围相对最初设想 (`ert_fast` 注册键、独立 `--quick` 入口) **收缩**: 不加新模型注册键、
  不加环境变量、不进 tools/*。重负载 (stability / learning curve / error propagation)
  是科学路径,**按设计不允许**使用快速档——H4 对它们零收益是目标而非缺陷。
  (SPEC §0 表中原写 `--quick`;实际入口是既有 `--test`,以 `--tier` 选档。)

### How
1. **`src/experiment_params.py`** 新增模块常量 `FAST_TIER_OVERRIDES` (快速档唯一真源):
   ert/rf `n_estimators 200→50, max_depth 15→10`;catboost `iterations 1000→200,
   depth 6→4, learning_rate 0.03→0.1` (lr 提高部分补偿迭代削减,冒烟指标肉眼可判读,
   不构成质量承诺);stacking `inner_cv 5→3`。
2. **`build_model_params(..., tier="full")`**: 仅接受 `'full' | 'fast'`,否则 `ValueError`。
   `tier='fast'` 在复制 config 默认后 `update(FAST_TIER_OVERRIDES[key])`;stacking 分支对
   三个基模型逐个套用覆盖 (在 `stacking_base_defaults` 之后,快速档最终生效) 并强制
   `inner_cv=3`。未覆盖键一律保持 config 默认。**未知模型模块无快速覆盖**: `tier='fast'`
   对 `{}` 回退分支不做任何事,与 `'full'` 行为一致 (测试钉死)。`tier='full'` (默认)
   分支零代码改动,返回字典与改动前逐位相同。
3. **`main.py`**: `get_experiment_configs(tier='full')` 透传;`run_quick_test(tier='fast')`
   横幅打印 tier、`save_config` extra_info 增 `model_tier`;argparse 新增
   `--tier {fast,full}` (默认 None),不带 `--test` 时 `parser.error`。`main()` **没有 tier
   形参**,快速档对 E01–E12 完整基准在代码层面不可达。
4. **不设环境变量** (有意决策,亦为全项目开关习语边界): 环境变量会泄漏给子进程/兄弟进程
   (本机常态多 CC 并发),CLI 标志作用域恰好是单次调用。**ML_\* env 只留给核数/线程预算
   与设备选择** (含既有 H3 的 `ML_CATBOOST_GPU_MIN_SAMPLES`;完整登记处为
   `src/runtime.py` 模块 docstring);**新行为开关一律走 CLI** (`--tier`、`--reuse-fits`)。
5. **tools/\* 与 `run_stability_repeats` 零改动**: 四个科学调用点全部不传 `tier`,grep 可证。

### Cost / Benefit (实测,1730×18,2 线程,机器有底载)
| 拟合 | 科学档 | 快速档 | 加速 |
|----|----|----|----|
| ERT 单拟合 | 0.27 s | 0.063 s | 4.3× |
| CatBoost 单拟合 | 2.31 s | 0.18 s | 12.8× |
| quick test 拟合预算 (24 次) | ~31 s | ~3 s | ~10.7× |

quick test 端到端 (含 CSV 加载、parquet/joblib 落盘等固定开销) 预期 ~2.5–4×。

### Verification
`tests/test_experiment_params.py` (7 用例,全部快速、确定性、无 GPU): 默认档与 `'full'`
逐键相等且钉死科学值;快速档逐键断言 (含 stacking 基模型继承、未覆盖键保持);非法 tier
报错;未知模块 fast 保持 `{}`;`get_experiment_configs()` 24 配置钉死科学档 (泄漏防护
回归锚);`get_experiment_configs(tier='fast')` E01/E02 为快速值;快速档参数实例化
ExtraTrees/CatBoost 并 fit + predict 全有限 (防覆盖键名与构造器失配)。
手动: `python main.py --test` 横幅显示 `model tier: fast`;`--test --tier full` 复现
此前 quick test 行为;`--tier` 不带 `--test` 报参数错误。

### Back-compat
- 科学路径: 仅多一个默认 `tier='full'` 的关键字参数,默认分支返回字典逐位不变,RNG 零接触。
- quick test 数字会变 (本就非科学产物,落 `results_test/`,无黄金对照): `--tier full`
  一键复现旧行为;`config_used.yaml` 记录 `model_tier` 供溯源。
- 旧 joblib 工件: 模型类与 predict 路径零改动。

---

## 18. H7 — Stacking 内层可选并行 (M3.2)

### Why
- `StrictOOFStacking.fit` 顺序执行 `inner_cv × n_base` (默认 5×3=15) 次基模型拟合 +
  3 次全量重拟合。每次拟合内部虽占满 `suggest_n_jobs("model")` 线程,但 CatBoost 在本
  项目数据规模 (augmented 内层折 ≈ 20k×18) 下约 4 线程即饱和——实测 `thread_count=14`
  1.51 s vs `=4` 1.59 s,>4 的核基本闲置。把这部分浪费换成任务级吞吐: 实测 3 个 CatBoost
  fit 顺序@14 线程 3.96 s vs 并发 W=3@4 线程 1.74 s = **2.28×**。
- ERT/RF 是粗粒度树级并行,线程扩展性好,外层并行对它们 ≈1×;CatBoost 占 augmented 内层
  fit 约 1/3 时间 → 单次 stacking fit 期望 **1.2–1.5×**。
- 收益集中: E12 的 nested correction (S1) 把每 target 的 stacking fit 提高到 61 次,是
  主基准最贵单项;`run_learning_curve` 默认含 stacking (3000 次 fit);`run_stability
  --model-module stacking` 为 22000 次 fit。

### How
1. **`StrictOOFStacking.__init__`** 增加 `inner_parallel: Optional[int] = None`:
   None → 经 `runtime._env_int("ML_STACKING_PARALLEL", 1)` 读 ENV (该变量已登记进
   `runtime.py` 模块 docstring 的 env 清单);解析失败或 ≤1 一律视为关闭 (顺序)。
   **>1 时 worker 数钳制到 `suggest_n_jobs('inner_loop')`** (per_fit 下界为 1,若不钳制
   workers,`W > budget` 时 W×1 会突破单拟合包络)。不提供 "auto" 档、不加 CLI flag。
2. **线程预算注入** (仅 `base_model_params` 构造路径): `inner_parallel > 1` 时对每份
   params **复制后** `setdefault`:
   `per_fit = max(1, suggest_n_jobs("inner_loop") // inner_parallel)`;
   ert/rf → `n_jobs`,catboost → `thread_count` (进入 `_user_kwargs`,自动抑制 fit 期
   注入)。**显式传入的线程值不被覆盖** (契约),因此不超订保证**有条件**: 项目 config 给
   ert 钉了显式 `n_jobs=4` (`config.ModelDefaults.ert`),它不会被除——**对照 config 默认
   应保持 `W ≤ budget // 4`** (14 核建议 W=3);显式值超过 per_fit 份额时 logger.warning
   明示可能超订。直接传 `base_models` 实例列表时不调整 (logger.warning),责任在调用方。
   **预算语义更正**: `suggest_n_jobs('inner_loop')` 尊重 `ML_N_JOBS` / `ML_RESERVE_CORES`;
   **只有 `'cross_proc'` 上下文除以 `ML_OUTER_PROCS`**,`'inner_loop'` 不除。多兄弟进程
   并发跑 stability 分段时,请按既有约定用 `ML_N_JOBS` 为每进程设上限。
3. **`StrictOOFStacking.fit`** 双分支: `≤1` 时现有顺序循环**逐字保留** (默认路径零代码
   变化);`>1` 时先 `splits = list(split_iter)` (K 折器产出与惰性消费逐位相同),把
   (fold × base) 摊平为任务,`joblib.Parallel(backend="threading")` 派发,主线程统一写
   `oof_meta[val_idx, j]` (各任务切片两两不相交,结果与完成顺序无关);最终全量重拟合循环
   同法并行。backend 取 threading: 基模型 fit 均为释放 GIL 的原生代码,零数据拷贝,避免
   Windows loky spawn 开销;joblib 已是既有依赖。
4. **RNG 不变性**: 每次基模型 fit 用构造期固定整数种子**新建** estimator,无跨 fit 共享
   顺序 RNG,`model_modules.py` 无任何 `np.random` 全局调用 → 执行顺序不影响任何种子消费。
5. **共享可变状态**: 并发 fit 时唯一写操作是信息性的 `_training_time` (良性竞态,docstring
   注明并行模式下该字段无意义);`self.params` / `_user_kwargs` 只读。
6. **与 nested_correction (S1) 零代码交互**: `protocol._compute_inner_oof` 经 pipeline
   factory 构造全新 stacking 实例,ENV 默认值自动生效;protocol 外层循环保持串行,任意
   时刻总并发 = W × per_fit ≤ budget 仍成立。`protocol.py` 无需改动。
7. **可观测性**: `main.py` 启动横幅 env 行追加 `ML_STACKING_PARALLEL`。
8. **GPU 守卫 (与 `ML_CATBOOST_GPU_MIN_SAMPLES` 同读)**: augmented 内层折 ≈20k ≥ 阈值
   5000,`task_type='auto'` 在 GPU 机器上会选 GPU,而并发 worker 争用同一设备可能更慢
   甚至 OOM——因此并行模式 (`inner_parallel > 1`) 下 **`'auto'` 一律钉到 CPU**;显式
   `task_type='GPU'` 保留并 logger.warning (责任在调用方)。顺序模式 (默认) 行为不变。

### Cost / 实测收益 (2026-06-11 空闲机验收,16 核 reserve 2,CatBoost 双模式均强制 CPU)
设计期预测 (单次 augmented fit 1.2–1.5×、E12 wall −15~25%) **被空闲机实测部分推翻**:

| 数据规模 | 顺序 | W=2 | W=3 | W=4 |
|---|---|---|---|---|
| raw 1730×18 (E03/E06 场景) | 7.5 s | **5.8 s (1.29×)** | 6.0 s (1.25×) | — |
| augmented 27.7k×18 (E09/E12 场景) | 45.4 s | **41.4 s (1.10×)** | 49.2 s (0.92×) | 55.2 s (0.82×) |

- 结论: **收益集中在 CatBoost 占比高的小矩阵** (raw/balanced stacking ~1.3×);augmented
  大矩阵上至多 1.10× (W=2),**W≥3 反而变慢**——设计假设"森林外层并行 ≈1×"在 27.7k 行上
  不成立: RF/ERT 砍线程的损失 + 内存带宽争用盖过 CatBoost 的任务级收益。
- **建议值修正: `set ML_STACKING_PARALLEL=2`** (而非设计期的 3);augmented 重负载
  (stability --model-module stacking) 上收益有限,启用前按本表预期。

### Verification
1. `tests/test_stacking_parallel.py` (8 用例,合成小数据,秒级): 默认关;ENV 启用;
   非法 ENV 回退 1;**workers 钳制到预算** (`ML_N_JOBS=2` + `inner_parallel=16` → 2);
   `inner_parallel=1` vs `3` 在基模型 `n_jobs=1` 钉死下 `_oof_meta_features` 与
   `predict` **逐位相等**;`ML_N_JOBS=2`+W=2 → 注入 `n_jobs=1`;显式 `n_jobs=3` 不被
   覆盖;并行模式 catboost `'auto'` 钉 CPU / 显式 `'GPU'` 保留。
2. 实测数值佐证 (20k×18): CatBoost `thread_count` 14 vs 4 预测**逐位相等**;ERT `n_jobs`
   14 vs 4 仅 predict 求和顺序差 ≤1 ulp (2.2e-16)。
3. 空闲机计时复测: **已执行 (2026-06-11)**,结果见上表;真实 stacking 配置 (config 默认
   超参,含 ert 显式 n_jobs=4) 直接 fit 实测,每模式单次测量。

### Back-compat
- 默认 (`ML_STACKING_PARALLEL` 未设或 ≤1) 走原顺序循环,逐字未动 → 全部科学路径
  bit-identical。
- 开启后为**纯性能开关但非逐位等同**: per-fit 线程数改变 → ERT/RF 预测有 ≤1 ulp 的浮点
  求和顺序差,CatBoost 实测逐位相等;统计上零影响,但**论文数字生成必须保持默认关闭**,
  或重跑整套基线后注明。
- 持久化 joblib 工件不含 inner_parallel 状态,新旧互放不受影响 (predict 路径未改)。

---

## 19. H6 — 外层 CV 拟合复用 (主基准缓存层,M3.2)

### Why
- 主基准 E10–E12 与 E07–E09 仅差校正模块,而校正是 OOF 之后的后处理,**完全不进入
  `Pipeline.fit`**。同一次 `run_experiments` 调用内,两组实验的 data_params /
  model_params / target_seed / 折切分逐项相同 → 外层 10 折拟合、全量拟合 (持久化用)
  与测试集原始预测是**逐位相同的重复计算**。
- 实测 (V7 黄金 `metrics_summary.csv` 的 `*_total_training_time`): E10–E12 双特征集
  T+P 外层折拟合合计 **10,690 s,占全部折拟合时间 25,756 s 的 41.5%** (E09_liq 6,411 s
  vs E12_liq 6,262 s 近似相等,佐证为同一计算)。V8 默认嵌套校正下 E10–E12 另付 ~4.5×
  内层拟合 (不可缓存),复用外层拟合仍省全基准 **~14–16%** 墙钟;`nested_correction=False`
  重放模式下省 **~40%**。
- 其余负载**无可缓存项** (核查结论,不做缓存): `run_stability_repeats` 每 repeat 种子
  唯一;`run_learning_curve` fraction=1.0 时行序按 repeat_seed 重排、折成员有意变化;
  `--test` 只跑 E01/E02 (无校正孪生,故 `--reuse-fits` 与 `--test` 互斥,parser.error)。

### How
1. **`StratifiedCVProtocol.run`** 新增 **keyword-only** 参数
   `precomputed_folds: Optional[List[Dict]] = None` (签名以 `*` 隔开,防位置传参破坏):
   - 非 None 时: 跳过 splitter 构造与外层折循环,采用瘦记录列表 (每条含
     `fold_id/train_idx/val_idx/y_val/y_pred_raw/training_time`,无 `pipeline`/`X_val`
     键),由 `val_idx + y_pred_raw` 重建 `oof_pred_raw`;**既有 "OOF prediction contains
     NaN" 完整性检查显式保留**,是瘦记录覆盖全样本的唯一结构性守卫。后续聚合校正器拟合、
     嵌套校正 (内层照常经 `pipeline_factory` 真实重训,种子流不变)、fold metrics、
     predictions_df 全部走既有代码。
   - 守卫: `precomputed_folds` 与 `uncertainty_module` 互斥 → `ValueError` (不确定度
     路径需要活 pipeline 并会 `set_correction` 原地改它)。
2. **`ExperimentMatrix.run_experiments`** 新增 **keyword-only** 参数
   `reuse_identical_fits: bool = False`:
   - 函数体内局部 `fit_cache` 字典,生命周期=单次调用 (X/y/stratify/n_splits/random_seed
     调用内恒定,无需进 key)。
   - key = `_fit_cache_key(config, target_name)` = (data_module 名, model_module 名,
     `_canon(data_params)`, `_canon(model_params)`, target);`_canon` 为递归规范化
     (dict→按键排序 tuple),**标量叶子带类型标签** (1 / 1.0 / True 在 Python 等值但对
     sklearn 是不同拟合,不得碰撞);**corr_module / corr_params 不进 key**。
     `run_uncertainty=True` 的实验既不产出也不消费缓存。
   - **命中**: `protocol.run(..., precomputed_folds=瘦记录)`;跳过 `full_pipeline.fit`,
     改为 `joblib.load(生产者工件)` 后仅替换 `corr_model` 与 `config` 再 dump 到本实验
     路径;测试集指标用缓存的 `y_test_pred_raw` + 本实验聚合校正器计算。打印
     `[reuse] outer-CV fits reused from {producer_exp_id}`。
   - **未命中**: 走现行路径,结束后存瘦条目 (各数组 copy 防共享可变状态,<1 MB/key)。
     命中失败永远安全——退化为全量计算 (如消费者排在生产者之前,只是缓存 miss)。
3. **`main.py`**: argparse 新增 `--reuse-fits` (默认 False,help 注明 timing 列语义),
   `main(reuse_fits=False)` 透传。`run_quick_test` 与三个 tools **不接**此参数 (无重复
   计算,接了是虚假接口面)。
4. **弃选方案**: 盘上缓存 (失效键复杂度不成比例);缓存活 pipeline 对象 (GB 级内存 +
   uncertainty 原地改);给 tools 加缓存 (收益为 0);实验重排序 (改变 metrics_summary
   行序,属默认行为变化)。

### Verification
`tests/test_fit_reuse.py` (5 用例,ert-only 合成小数据 + n_splits=3,全 CPU 确定性,
默认套件可跑): reuse on/off 的 summary 与 fold_metrics 除 `*training_time*` 外逐位相等
(嵌套校正开启,连带验证嵌套在 precomputed 路径上照常工作);monkeypatch 计数
`ExtraTreesModel.fit` 证明 nested off 时 reuse on = 8 次 (仅生产者 2 目标 × (3 折 + 1
全量)) vs off = 16 次;消费者 joblib 工件与生产者模型预测逐位相同且 `corr_model`/`config`
为本实验所有;仅 `n_estimators` 不同 → 缓存 miss、双方全量拟合;`precomputed_folds` +
`uncertainty_module` → `ValueError`。
手动: `python main.py --reuse-fits` 与默认跑的 `metrics_summary.csv` 对比,除 timing
列外逐位一致;日志出现 12 行 `[reuse]` (双特征集 × E10–E12 × T/P 双目标)。

### Back-compat
- **默认关**: 两个新参数均为 keyword-only 且默认 None/False,不进入任何新代码路径,
  全部科学路径输出逐位不变。
- 开启时: 科学指标在确定性路径 (ExtraTrees/RF/Ridge/CatBoost-CPU) 上逐位等于不开;
  CatBoost-GPU 重训本就不保证逐位确定,复用反而消除 E08/E11 间的随机差异——文档注明。
  **`training_time` / `total_training_time` 列在消费者实验中为生产者的测量值** (同一
  计算的真实耗时);不同开关状态下的 timing 列不可互相对比 (H7 同理)。
- 持久化工件: 消费者 joblib 与其全量重训产物内容等价 (同种子同数据的确定性拟合);
  reused 运行的 `results['fold_records']` 无 `pipeline`/`X_val` 键——仓库内无消费方
  (uncertainty 路径已被守卫排除),第三方代码若在 `--reuse-fits` 下窥探该字段需注意。
