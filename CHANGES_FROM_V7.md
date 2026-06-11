# Changes from V7

> 按时间倒序记录 V8 相对 V7 的逐项改动。每条注明 SPEC 编号与是否影响数值。

## M1 — 基础设施 (进行中)

- **[S6]** 新增 `src/repro.py`: 代码指纹与 git 状态采集,统一 `config.get_version_info` 实现。
- **[H1]** 新增 `src/runtime.py`: 硬件探测层 (物理核数 / GPU / 内存 / 平台)。
- **[H2]** 新增 `runtime.suggest_n_jobs(context, reserve_cores)`; `_get_default_n_jobs` 改为委托给它。新增 ENV `ML_RESERVE_CORES`, `ML_OUTER_PROCS`,兼容旧 `ML_N_JOBS`。
- **[S6]** `model_modules.py` 删除 `config.yml` 失效注释,标注 `config.CatBoostConfig` 为唯一真源。
- **[S6]** `protocol.ExperimentMatrix.save_config` 在 `config_used.yaml` 增 `code_sha` / `code_fingerprint`。
- **[meta]** 新增 `SPEC_V8.md`, `CHANGES_FROM_V7.md`。
- **[meta]** `requirements.txt` 追加 `psutil>=5.9`。

数值影响: **无**。E01–E09 + E10–E12 输出应与 V7 一致 (默认 `nested_correction=False, merge_sparse_bins=False`)。

## M2 — 科学修正 (代码就位,基线重跑待执行)

- **[S1]** `StratifiedCVProtocol` 新增 `nested_correction: bool = True` 与 `inner_correction_cv: int = 5`。当校正模块非 `NoCorrection` 时,在每个外层验证折之上做内层 K 折 OOF 来拟合**专属的** `corr_model_fold`,消除 V7 的 in-sample 评估偏差。聚合 `corr_model` (供持久化和测试集使用) 仍在全部 OOF 上拟合,因此持久化的模型与测试集指标向后兼容。
- **[S1]** `ExperimentMatrix.run_experiments` 暴露 `nested_correction` / `inner_correction_cv` 参数,默认 True / 5。精确复现 V7 数字需**同时**设 `nested_correction=False, merge_sparse_bins=False` (后者影响折成员)。
- **[S2]** `StratifiedCVProtocol` 新增 `merge_sparse_bins: bool = True` 与 `min_samples_per_bin: Optional[int] = None`。主基准默认走 `_merge_sparse_bins`,与 `run_stability` / `run_learning_curve` 子工具一致,消除 sklearn "least populated class" 告警。
- **[meta]** `summary` 新增 `correction_eval_mode` ∈ {`nested`, `in_sample`, `none`}, `correction_inner_cv`, `merge_sparse_bins` 三字段,持久化到 `metrics_summary.csv` 便于审稿溯源。
- **[test]** `tests/test_nested_correction.py` (7 用例): 验证模式字符串、no-corr 短路、`nested RMSE ≥ in_sample RMSE` 方向性、聚合 corr_model 仍返回、back-compat 路径。

数值影响:
- E01–E09 (NoCorrection): **无变化** —— nested 分支被短路,merge_sparse_bins 在数据已有稀疏 bin 时会改变折成员; 主基准实际会**轻微变动**,但与 V7 评估精度等价。
- E10–E12 (SegmentedLinearCorrector): **CV 数字会变高**,反映真实泛化误差; 留出测试集数字保持不变。

完整测试套件: **105/105 通过**。

## M3 — 性能优化 (已做: H3 + 启动期摘要; H4/H5/H7 留待后续)

- **[H3]** CatBoost 设备决策从 `__init__` 推迟到 `fit()`,基于训练集大小自动选 CPU/GPU。新增 `_resolve_runtime_params(n_samples)` 方法,逻辑:
  - `task_type='CPU'` / `'GPU'` 显式指定 -> 直接生效。
  - `'auto'` -> 无 CUDA 设备走 CPU; 有设备且 `n_samples >= ML_CATBOOST_GPU_MIN_SAMPLES` (默认 5000) 才走 GPU; 否则走 CPU (GPU 启动开销在小数据上不划算)。
  - V7 callers (`n_samples=None`) 走旧行为,完全向后兼容。
- **[H3 影响]** 本项目训练集 ~1730 样本,远低于 5000,因此 CatBoost 现在**默认在 CPU 上跑**,与 V7 在大显存机器上"被迫切 GPU 又不快"的行为相比是净加速。需要时设 `ML_CATBOOST_GPU_MIN_SAMPLES=1000` 即可强制 GPU。
- **[startup]** `main.py` 启动期打印 `runtime_summary_str()` + `suggest_n_jobs(model/inner_loop/cross_proc)` + 关键环境变量。多进程并发时一眼能看出是否超订。
- **[test]** `tests/test_catboost_device.py` (9 用例): 显式 CPU/GPU、auto 数据大小阈值、env 覆盖、init 不锁定设备、`n_samples=None` 兼容。
- **[defer]** H4 (`ert_fast`)、H5 (增广 float32 / 流式)、H6 (cache 层)、H7 (stacking 内层并行) 暂缓,优先级低且改动接口面较大,留作 M3.2。

数值影响: **无** (CatBoost CPU/GPU 结果在 RMSE 数千分位级别一致; 主基准跑出的数字与 V7 等价)。

完整测试套件: **114/114 通过**。

## M4 — 数据/校正语义 + 全面核验修复 (2026-06-11)

> 本批分两部分: (a) M4 三项科学修正 S3/S4/S5; (b) 分支化后对 M1–M3 全部前序工作
> 的多代理核验 (9 个审查/设计代理) 暴露的 8 个 major / 27 个 minor 问题的修复。
> 详细设计见 SPEC_V8.md §13–15。

### M4 科学修正

- **[S3]** `epma_perturb` / `perturbation_with_repeats` 新增 `clip_negative=True`:
  扰动后氧化物 wt% 截断至 0。RNG 流不变;默认 3–8% 误差下截断概率 < 4e-36,
  数值逐位等同 V7。`tools/run_error_propagation.py` ep_meta 说明同步更新。
- **[S4]** `main.load_data` 删除静默 `np.nan_to_num(X, nan=0.0)`,改为特征/目标含
  NaN 时 ValueError loud-fail。实测 input.csv 特征列 0 个 NaN → 当前数据逐位无变化。
  字面 0 (Cr2O3.cpx 53% 行等) 按上游"未分析/低于检出限"约定保留,文档化。
- **[S5]** `SegmentedLinearCorrector` 新增 `edge_mode='offset'` (V8 默认):
  界外预测以斜率 1 + 边界校正偏移延拓 (V7 为完全不校正的直通,边界跳变)。
  `edge_mode='raw'` 复现 V7;旧持久化 corr_model 缺键自动按 raw 回放。
  数值影响仅 E10–E12 尾部 (~0.06–0.12% 样本)。

### 核验修复 (major)

- **[S1 修复]** `run_stability_repeats` 与 `tools/run_learning_curve.py` 显式传
  `nested_correction=False`: 前者的 CV 逐折校正器本就被丢弃 (只用聚合 corr_model),
  V8 默认会带来 ~6x 纯浪费;后者保持 V7 in-sample 语义与成本,避免学习曲线数字
  未文档化漂移。嵌套评估升级仅作用于主基准。
- **[S6 修复]** `save_config` 补写顶层 `code_fingerprint` (逐文件 sha256 字典,
  SPEC §2 承诺但此前未持久化);`run_quick_test` 现在也调用 `save_config`
  (此前 results_test/ 永远没有 config_used.yaml)。
- **[测试交付]** 补交 SPEC §7 承诺但缺失的 `tests/test_regression_vs_v7.py`
  (E01 Liquid vs V7 黄金数字,rel 1e-4,`@pytest.mark.slow`);新增 `pytest.ini`
  注册 slow marker 且默认 `-m "not slow"`。
- **[README]** 新增 V8 说明: 环境变量、新模块、嵌套/合并默认值、并发礼让建议。

### 核验修复 (minor,择要)

- **[种子]** 内层管线种子改为 `random_seed + 500000 + outer_fold*100 + inner_fold`:
  旧公式 `outer_fold*1000` 与 `P_SEED_OFFSET=1000` 碰撞,导致 T 折 k+1 与 P 折 k
  的内层 RNG 流相同,违反 T/P 独立性设计意图 (仅影响 nested 模式,E10–E12)。
- **[SPEC §5.5]** `_compute_inner_oof` 实现承诺的 KFold 回退: 分层 bin 过稀
  (最小 bin < k) 时自动退为普通 KFold 并告警;内层折数被缩减时记录 warning。
- **[清理]** 移除死字段 `fold_records['stratify_train']`、`config.py` 孤儿
  `import subprocess`、`tests/test_model_modules.py` 两处过期 `config.yml` 字样。
- **[H3 测试]** `test_auto_no_gpu_falls_back_to_cpu` 原为永真断言,改为 monkeypatch
  强制无 GPU 分支;补 `thread_count` 不被覆盖、用户 kwargs 优先两用例。
- **[H1 测试]** 补 GPU 探测失败注入测试 (SPEC §3 承诺): 伪造损坏 torch/catboost,
  断言降级为 no-GPU 而非抛错。
- **[礼让]** `tests/conftest.py` 对 `ML_N_JOBS` setdefault("2") (SPEC §11.3);
  `main.py --test` setdefault("4") (§11.4);三个 tools 入口打印硬件摘要行 (§11.1)。
- **[文档]** SPEC §1 重写为分支模式工作流 (main 受保护,V8 分支开发,cpx 环境);
  §3 `ram_gb` 标注 Optional;§4 thread_count 注入时机更正为 fit();§9 里程碑
  勾选与实际进度同步;M2 措辞修正 (复现 V7 需同时关两个开关)。

### 已知未改 (有意保留,避免破坏 back-compat)

- 全量拟合路径 (`run_experiments` 持久化模型) 仍用未合并的原始分层标签喂
  StackingModel 内层 → sklearn 告警仍会出现一次;改它会改变持久化模型与
  测试集指标,违背 M2 的 back-compat 保证。完整重跑 V8 基线时再统一。
- `_merge_sparse_bins` 的"最近 bin"按展平 id 距离度量,P 行边界处可能跨行合并
  (V7 子工具既有行为,确定性可复现)。
- V7 时代旧 pickle 的 CatBoostModel 缺新属性,re-fit 会 AttributeError;
  仓库内所有消费方仅 predict,不受影响。

数值影响:
- S3/S4: **无** (当前数据与默认误差下逐位不变)。
- S5: E10–E12 尾部样本 (~0.06–0.12%) 校正值变化,聚合 RMSE 预期变化 < 0.01。
- 种子修复: 仅 nested 模式 (E10–E12 CV) 数字微变 (该模式 V8 新增,无既有基线)。
- E01–E09 与 V7 等价配置路径: **不变** (回归测试可证)。

完整测试套件: **133/133 通过** (含 19 个新用例),slow 回归测试默认 deselected。

## M3.2 — 进一步性能优化 (待做: H4/H5/H6/H7)

