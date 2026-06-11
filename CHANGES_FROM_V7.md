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

### V8 基线实测对比 (2026-06-11 完成重跑)

运行事实: 24/24 实验,V8 分支 f029931 (代码指纹 `code_sha=a3611e4fbc890602` 已随
`config_used.yaml` 落盘,S6 元数据首次完整生效),cpx 环境,`ML_RESERVE_CORES=2`,
墙钟 ~7.5 h (12:54–20:25)。`results/` 已由 V8 基线覆盖;V7 黄金数字保留于 git
(main 分支) 与 `tests/golden_v7_metrics_summary.csv` 镜像 (回归测试改读镜像)。

**与 V7 的逐实验对比 (CV RMSE):**

| 组 | T_rmse 变化 | P_rmse 变化 | 归因 |
|---|---|---|---|
| ERT / Stacking, raw+balanced (E01/03/04/06) | −0.01% ~ −4.3% | −0.4% ~ −4.8% | S2 折成员变化 (+stacking 含 CatBoost 基模型的设备效应) |
| **CatBoost raw+balanced (E02/E05)** | **−6.8% ~ −15.1%** | **−6.1% ~ −13.8%** | **H3 设备切换**: V7 auto→GPU,V8 小数据 (1557 < 5000) 走 CPU |
| CatBoost/含 CB 实验, augmented (E07–E09) | −0.2% ~ +1.9% | −1.4% ~ +1.7% | augmented 折 ≈24.9k ≥ 5000 → V8 仍走 GPU (已核实工件 task_type),变化 = 折成员 + GPU 非确定性 |
| E10–E12 (segmented, nested 评估) | −2.0% ~ +2.1% | −2.3% ~ +1.8% | nested + 折成员;**未系统性变差** |

要点:
- **修正 M3 的预期**: "CatBoost CPU/GPU 结果数千分位一致"在小数据上**不成立**——CPU
  显著更好 (E02_liq T_rmse 36.31→31.19, −14%)。V7 时代 CatBoost 的 raw/balanced 数字
  被 GPU 模式拖低,V8 数字反映 CPU 真实水平;CatBoost 与 ERT/stacking 的相对排序在
  raw/balanced 组因此改变,**论文中涉及 E02/E05 的结论需对照新基线复核**。
- **嵌套校正评估 (S1) 实测影响很小**: E10–E12 CV 变化 ±2%,V7 的 in-sample 评估偏差
  实际不大;校正收益在诚实评估下保持 (V8 内部 E10 vs E07 仍改善)。
  `correction_eval_mode=nested` / `correction_inner_cv=5` / `merge_sparse_bins=True`
  三个新列已落盘 `metrics_summary.csv`。
- **留出测试集如设计般稳定**: E10–E12 测试集 RMSE |Δ| ≤ 0.42% (聚合校正器向后兼容 +
  S5 边段仅动尾部 + GPU 漂移),验证 M2 的 back-compat 承诺。
- **性能**: 外层折拟合总时长 25,756 s → 8,914 s (**−65%**),主因 CatBoost 小数据
  CPU 化 (H3)。墙钟大头是 E10–E12 嵌套校正的内层拟合 (不计入 total_training_time;
  E12 单实验 ~2 h/特征集)——正是 M3.2 的 H6 (拟合复用) 与 H7 (stacking 内层并行)
  的目标场景。

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

## M3.2 — 进一步性能优化 (2026-06-11,H4/H6/H7 落地,H5 评估后弃置)

> 设计经 4 个独立勘察代理 + 1 个横向评审代理裁决 (approve-with-changes ×3,
> defer ×1) 后实施,实施后再经 4 个对抗性复审代理 + 逐发现独立核实 (8 项确认全部修复)。
> 详细设计见 SPEC §16–19。开关习语边界自此固定: **ML_\* 环境变量只留给核数/线程预算与
> 设备选择 (含既有 H3 的 `ML_CATBOOST_GPU_MIN_SAMPLES`;`src/runtime.py` docstring 为
> 完整登记处),新行为开关一律走 CLI flag**。

- **[H5] 评估后弃置 (零代码)**: 实测增广矩阵仅 3.59 MB / 生成占增广管线拟合成本 0.5%,
  内存大头是树模型本身 (pickle 37 MB);权重模拟科学上不可行 (扰动副本 X 各不相同),
  float32 影响数值且收益 ~5 MB,流式对批式学习器无意义。证据表与重启条件 (数据增长
  ≥50x) 记录于 SPEC §16。数值影响: **无**。
- **[H4] 模型快速档**: `experiment_params.FAST_TIER_OVERRIDES` + `build_model_params`
  新增 `tier='full'|'fast'` 关键字参数;`main.py --test` 默认走 fast 档 (ERT 50 树/深 10、
  CatBoost 200 it/深 4/lr 0.1、stacking inner_cv 3),`--test --tier full` 复现旧 quick
  test;`--tier` 不带 `--test` 报错;`results_test/config_used.yaml` 记录 `model_tier`。
  实测拟合预算 ~31 s → ~3 s (~10.7×)。**有意不设环境变量** (防泄漏给并发兄弟进程)。
  快速档对 E01–E12 基准代码层面不可达 (`main()` 无 tier 形参,
  `test_main_matrix_never_fast` 钉死 24 配置)。数值影响: 仅 quick test (非科学产物);
  科学路径返回字典逐位不变。
- **[H7] Stacking 内层可选并行**: 新 ENV `ML_STACKING_PARALLEL` (int,默认 1=顺序,
  已登记进 `runtime.py` env 清单) / 构造参数 `StrictOOFStacking(inner_parallel=None)`。
  >1 时 worker 数**钳制到 `suggest_n_jobs('inner_loop')`**,把 (fold × base) 摊平为任务
  经 `joblib(backend='threading')` 派发,并对每基模型 `setdefault` 注入
  `per_fit = budget // workers` (ert/rf → `n_jobs`,catboost → `thread_count`)。
  **不超订保证有条件**: 显式线程值不被覆盖 (config ert 钉了 `n_jobs=4`),对照 config
  默认应保持 `W ≤ budget // 4` (14 核建议 3),超额时 logger.warning;并行模式下 CatBoost
  `task_type='auto'` 钉到 CPU (并发 worker 不得争用同一 GPU),显式 `'GPU'` 保留并告警。
  注意 `'inner_loop'` 预算尊重 `ML_N_JOBS`/`ML_RESERVE_CORES` 但**不**除以
  `ML_OUTER_PROCS` (仅 `'cross_proc'` 除);多进程并发请仍用 `ML_N_JOBS` 限每进程。
  **空闲机实测 (验收,推翻设计期 1.2–1.5× 预测的 augmented 部分)**: raw 1730×18 上
  W=2 得 **1.29×** (CatBoost 占比高,收益真实);augmented 27.7k×18 上至多 1.10× (W=2),
  **W≥3 反而变慢** (0.92×/0.82×,森林砍线程损失 + 带宽争用盖过 CatBoost 任务级收益)。
  建议 `ML_STACKING_PARALLEL=2`,详表见 SPEC §18。
  数值影响: 默认关,顺序循环逐字保留 → bit-identical;开启后 ERT/RF 仅 ≤1 ulp 求和顺序差,
  论文数字生成保持默认关。
- **[H6] 外层 CV 拟合复用 (opt-in 缓存层)**: `StratifiedCVProtocol.run` 新增 keyword-only
  `precomputed_folds`;`ExperimentMatrix.run_experiments` 新增 keyword-only
  `reuse_identical_fits=False`;`main.py --reuse-fits` (与 `--test` 互斥)。依据: 校正
  不进入 `Pipeline.fit`,故 E10–E12 与 E07–E09 的外层折拟合/全量拟合/测试集原始预测逐位
  相同——实测占 V7 全部折拟合时间 **41.5%** (10,690 s / 25,756 s)。命中时跳过外层拟合,
  joblib 工件由生产者工件替换 `corr_model`/`config` 而来;嵌套校正内层照常真实重训。
  `run_uncertainty` 实验不产出不消费;OOF NaN 完整性检查在 precomputed 路径保留。
  V8 默认嵌套下预计省主基准墙钟 ~14–16%,`nested_correction=False` 重放省 ~40%。
  **真实数据验收 (E07/E10 型 ERT 对,Liquid,10 折 + 嵌套)**: reuse on/off 所有非 timing
  列一致到 1.2e-12 相对差以内——该残差是 ERT 多线程 predict 的求和顺序噪声 (同种子
  重跑两次本身就有 ~4e-16 绝对差),单线程下逐位相等已由单测证明;`[reuse]` 行数符合
  预期,该最廉价配对墙钟已省 18.5%。
  数值影响: 默认关 → 逐位不变;开启后确定性路径逐位相等,**timing 列变为生产者测量值**
  (不同开关状态的 timing 不可互比,H7 同理)。
- **[test]** 新增 `tests/test_experiment_params.py` (7 用例)、`tests/test_stacking_parallel.py`
  (8 用例)、`tests/test_fit_reuse.py` (6 用例),全部快速/确定性/无 GPU;ML_\* 操作一律
  经 pytest monkeypatch,无 os.environ 直写。
- **[复审修复]** H6 缓存键 `_canon` 对标量叶子加类型标签 (1/1.0/True Python 等值但对
  sklearn 是不同拟合,曾可静默错误复用——opt-in 路径下的潜在正确性缺陷,已修 + 回归
  测试);H7 worker 钳制与有条件不超订表述 (见上);`runtime.py` env 登记处补全。

数值影响汇总: 三项默认全关/不可达 → E01–E12、stability、learning curve、error
propagation **全部逐位不变**;唯一默认行为变化是 `main.py --test` 的冒烟指标 (非科学
产物,`--tier full` 可复现)。

完整测试套件: **154 通过** (133 旧 + 21 新),slow 回归测试默认 deselected。

