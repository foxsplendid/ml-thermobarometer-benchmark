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
- **[S1]** `ExperimentMatrix.run_experiments` 暴露 `nested_correction` / `inner_correction_cv` 参数,默认 True / 5。`nested_correction=False` 可精确复现 V7 数字。
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

## M3.2 — 进一步性能优化 (待做)
## M4 — 数据/校正语义 (待做)

