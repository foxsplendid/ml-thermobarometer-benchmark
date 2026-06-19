# V8 工作总结与论文结论可靠性审阅

> 本文件是 V8 分支全部工作的单一汇总文档,供作者、合作者或后续会话快速接手。
> 编写日期 2026-06-12;V8 基线重跑完成于 2026-06-11。
> 详细设计见 `SPEC_V8.md`,逐 commit 改动见 `CHANGES_FROM_V7.md`,本文件聚焦
> **全局图景 + 论文结论可靠性 + 待办**,与那两份文档互补而非重复。

---

## 0. 版本与分支约定(务必先读)

| 名称 | git 实体 | commit | 状态 |
|---|---|---|---|
| **V7 = 论文版本** | `main` 分支 / `V7-paper` 标签 / `V1.1` 标签 | `004e4c7` | **冻结,随论文出版的附件代码,任何时候不得更改** |
| **V8 = 科学性二次验证** | `V8` 分支 | `1a63296` | 个人为确保科学性的进一步验证,**不合并入 main** |

- 运行环境: `D:\anaconda3\envs\cpx\python.exe`(catboost 1.2.7);base Anaconda 无 catboost。
- V7(main)远程未受任何影响,已经密码学级验证(详见 §7)。
- 若论文决定采用 V8 数字,正确做法是为 V8 打 release 标签 + Zenodo 新版本,**绝不动 main**。

---

## 1. V8 相对 V7 做了什么(改动总览)

V8 分为两类改动: **科学修正**(可能改变论文数字)与**工程优化**(默认不改数字)。

### 1.1 科学修正(S 系列)

| ID | 名称 | 影响论文数字? | 要点 |
|----|----|----|----|
| **S1** | 校正器嵌套评估 | 是(E10–E12 CV) | V7 用全 OOF 拟合校正器再作用回同一批预测(in-sample,系统性高估校正收益);V8 改为外层每折之上做内层 5 折 OOF 拟合专属校正器。实测影响仅 ±2%。 |
| **S2** | 统一稀疏 bin 处理 | 是(全实验,轻微) | V7 主基准对单样本 P-T bin 不合并(sklearn 告警、分层退化),子工具却合并。V8 统一在切分前合并(222→39 bin),折成员变化致全实验 0–2% 漂移。 |
| **S3** | EPMA 扰动非负截断 | 否 | 乘性高斯噪声理论上可产生负氧化物 wt%;加 `clip_negative=True`。默认 3–8% 误差下截断概率 <4e-36,逐位等同 V7。 |
| **S4** | 区分缺测与真零 | 否 | V7 用 `np.nan_to_num` 静默填零混淆"未测量"与"真零"。实测 input.csv 特征列 0 个 NaN,改为含 NaN 时 loud-fail;字面 0 按"低于检出限"约定保留。 |
| **S5** | 校正器边段外推 | 是(E10–E12 尾部 ~0.1%) | V7 对界外预测完全不校正且边界跳变;V8 `edge_mode='offset'` 斜率 1 + 边界偏移延拓。 |
| **S6** | 配置真源 + 复现性元数据 | 否 | 新增代码指纹(sha256)+ git 状态,写入 `config_used.yaml`。V8 基线首次携带 `code_sha=a3611e4f...`。 |

**影响论文数字的关键只有三处**: S1(嵌套评估)、S2(折成员)、以及下面工程项里的 **H3(CatBoost 设备切换)**。

### 1.2 工程优化(H 系列)

| ID | 名称 | 影响论文数字? | 要点 |
|----|----|----|----|
| **H1** | 硬件探测层 `runtime.py` | 否 | 统一物理核/GPU/内存探测,收口散落的 `os.cpu_count()`。 |
| **H2** | 并行预算分配器 | 否 | `suggest_n_jobs(context)` + 新环境变量 `ML_RESERVE_CORES`/`ML_OUTER_PROCS`,兼容旧 `ML_N_JOBS`。 |
| **H3** | CatBoost 自动设备选择 | **是(重大)** | 设备决策从 init 推迟到 fit,按训练集大小自动选 CPU/GPU(阈值 `ML_CATBOOST_GPU_MIN_SAMPLES=5000`)。**详见 §3 与 §4——这是本阶段最重要的科学发现**。 |
| **H4** | 模型快速档 | 否(仅 `--test`) | `--tier fast` 让冒烟测试拟合预算 ~31s→~3s。科学路径代码层面不可达。 |
| **H5** | 增广内存/权重模拟 | 否(弃置) | 实测增广矩阵仅 3.59 MB、生成占拟合成本 0.5%,三方案均不成立,弃置零代码。 |
| **H6** | 缓存层(外层 CV 拟合复用) | 否(默认关) | `--reuse-fits`:E10–E12 复用 E07–E09 逐位相同的外层拟合(占 V7 折拟合时间 41.5%)。 |
| **H7** | Stacking 内层可选并行 | 否(默认关) | `ML_STACKING_PARALLEL`:任务级并发。实测收益温和,详见 §2。 |

---

## 2. M3.2 性能项实施详情(H4/H5/H6/H7)

> 设计经 4 个独立勘察代理 + 横向评审裁决,实施后再经 4 个对抗性复审代理 + 逐发现核实
> (8 项确认问题全部修复)。开关习语边界: **ML_\* 环境变量只留给核数/线程/设备,
> 行为开关一律走 CLI flag**。

- **H4 快速档**: `experiment_params.FAST_TIER_OVERRIDES` + `build_model_params(tier=)`;
  `main.py --test` 默认 fast,`--test --tier full` 复现旧行为。实测端到端冒烟 4 秒。
- **H5**: 弃置,证据表写入 `SPEC_V8.md §16`。
- **H6 拟合复用**: `precomputed_folds`/`reuse_identical_fits`/`--reuse-fits`(与 `--test` 互斥)。
  **真实数据验收**: reuse on/off 所有非 timing 列一致到 1.2e-12 相对差(纯线程求和噪声,
  单线程下逐位相等);ERT 对省 18.5% 墙钟,含 CatBoost 的配对省得更多。
- **H7 stacking 并行**(空闲机实测,**部分推翻设计期 1.2–1.5× 预测**):

  | 数据规模 | 顺序 | W=2 | W=3 | W=4 |
  |---|---|---|---|---|
  | raw 1730×18(E03/E06) | 7.5 s | **5.8 s (1.29×)** | 6.0 s | — |
  | augmented 27.7k×18(E09/E12) | 45.4 s | **41.4 s (1.10×)** | 49.2 s (0.92×) | 55.2 s (0.82×) |

  结论: 收益集中在 CatBoost 占比高的小矩阵;augmented 大矩阵至多 1.10×,**W≥3 反而变慢**。
  **建议 `ML_STACKING_PARALLEL=2`**(原设计建议的 3 已修正)。

复审修复的关键问题: H6 缓存键 `_canon` 对标量叶子加类型标签(`1==1.0==True` 在 Python
等值但对 sklearn 是不同拟合,曾可静默错误复用——opt-in 路径下的潜在正确性缺陷);
H7 worker 钳制到预算 + CatBoost 并行模式钉 CPU + 有条件不超订表述。

---

## 3. V8 基线重跑:V7 ↔ V8 数字对比

运行事实: 24/24 实验,墙钟 ~7.5 h,cpx 环境,`ML_RESERVE_CORES=2`。`results/` 已由 V8
覆盖;V7 黄金数字保留于 main 分支与 `tests/golden_v7_metrics_summary.csv` 镜像。

### 逐实验 CV RMSE 变化(归因)

| 组 | T_rmse 变化 | P_rmse 变化 | 归因 |
|---|---|---|---|
| ERT/Stacking,raw+balanced(E01/03/04/06) | −0.01% ~ −4.3% | −0.4% ~ −4.8% | S2 折成员(+stacking 含 CatBoost 基模型的设备效应) |
| **CatBoost raw+balanced(E02/E05)** | **−6.8% ~ −15.1%** | **−6.1% ~ −13.8%** | **H3 设备切换**: V7 auto→GPU,V8 小数据(1557<5000)走 CPU |
| 含 CB,augmented(E07–E09) | −0.2% ~ +1.9% | −1.4% ~ +1.7% | augmented 折 ≈24.9k≥5000 → V8 仍 GPU;变化=折成员+GPU 非确定性 |
| E10–E12(segmented,nested) | −2.0% ~ +2.1% | −2.3% ~ +1.8% | nested+折成员;未系统性变差 |

### 三个最重要的结论

1. **修正 M3 旧预期**: "CatBoost CPU/GPU 结果数千分位一致"**在小数据上不成立**——
   CPU 显著更好(E02_liq CV T 36.31→31.19,−14%;E05_liq −15%)。**V7 论文中 CatBoost 在
   raw/balanced 组的数字被 GPU 模式系统性拖低**,相对排序因此改变。
2. **嵌套校正评估(S1)实测影响很小**: E10–E12 CV ±2%,V7 的 in-sample 偏差实际不大,
   分段校正收益在诚实评估下保持。
3. **留出测试集稳定**: E10–E12 测试集 RMSE |Δ|≤0.42%,验证 back-compat 承诺。

性能: 外层折拟合总时长 25,756 s → 8,914 s(**−65%**,主因 CatBoost 小数据 CPU 化)。

---

## 4. 论文结论可靠性审阅(核心)

审计方法: 3 个独立代理脚本化对比 V7 黄金 vs V8 基线(CV+测试集双轨)、子实验暴露面、
Fig.1–8 数据源映射。

### 4.1 被推翻/削弱的结论(须修改论文,根因几乎全是 H3 设备伪象)

1. **"CatBoost 不适合小样本 raw 数据(比 ERT 差 ~13%)"——翻转**。effect_table E02 liq T
   +13.0%(V7,比 ERT-raw 差)→ −2.5%(V8,反而更好);E05、P 目标同向翻转。V7 的"CatBoost 弱"
   是 GPU 测量伪象,非建模发现。
2. **"增广对 CatBoost 收益最大(~13%)"——翻转(符号反转)**。增广 vs raw 的 CatBoost
   收益:V7 −13.4% → V8 +2.7%(liq T,CV)。V7 巨大"收益"实为弱 GPU raw 基线被反衬的假象;
   **受益排序从 catboost>ert>stacking 变为 ert 一枝独秀**。
3. **raw-noliq 测试集最佳 T 模型**: stacking → catboost(唯一测试集排序翻转)。
4. **CV 组内排序多处变动**(约 15/48 处)——但见 4.2 第 6 条。
5. **校正对 CatBoost/stacking 的 CV 小幅收益**: V7 的 −0.1~−0.3% 在嵌套评估下变为 +0.1%。
   幅度 <0.45pp,叙事影响小但符号不再可靠。

### 4.2 经受住检验的稳健结论(两版下均成立,可保留)

1. **Liquid 特征大幅改善 T/P 预测**——全部 12 实验 × 2 目标 × 2 指标**零符号翻转**
   (T 测试集 −30~−40%,如 E01 −38.8% 两版逐字节一致)。**全研究最稳健的结论**。
2. **ERT 从增广明确获益**——test T noliq −6.14% 两版**逐字节一致**(ERT 路径完全不受
   任何 V8 改动影响)。
3. **类别平衡无实质效果**——所有 balanced vs raw 差异 |Δ|<1.5%,两版一致。
4. **分段校正在留出测试集上的诚实收益保持**——ERT-liq P-test −4.9%(两版一致),
   E10–E12 测试集最大漂移 0.42%。"校正只在有系统偏差时有用"的论点完好。
5. **增广组最佳 T 模型是 ERT**——两版不变。
6. **最重要的元结论**: **所有组内模型排序在两版下 95% CI 均重叠**。任何"模型 A 优于
   模型 B"的组内断言**从来没有统计支持**——论文如有此类表述,最安全的改法是改为
   "各模型表现统计上不可区分,差异由数据策略和特征集主导"。这正是论文 §3.4 的核心论点,
   V8 下反而更好论证。

### 4.3 子实验暴露面

| 工件 | 状态 | 说明 |
|---|---|---|
| `results/stability/`(1000 重复) | ✅ 无需重跑 | ERT-only 无 CatBoost,bin 合并语义早已一致,fit 路径验证逐字节相同 |
| `results/error_propagation/` | ✅ 无需重跑 | 内嵌基线 RMSE(σT=6.83/σP=0.45 对应)与 V8 完全一致到全浮点精度 |
| `results/learning_curve/` | ⚠️ 两个问题 | (a) fraction 0.2 的 stacking 点受 H3 影响(内层 CatBoost ~4.7k<5000→CPU);(b) **盘上只有 8 个 repeats**,论文若声称 30 需注意(与 V8 无关的完整性问题) |

### 4.4 图件状态(Fig.1–8)

| 图 | 状态 | 行动 |
|---|---|---|
| Fig.1(P-T 分布)、Fig.3(折覆盖) | ✅ 零风险 | 只依赖 input.csv,不动 |
| Fig.2(手绘流程图) | ⚠️ 仓库外 | 人工核对其中方法学描述(折协议/设备)是否与 V8 一致 |
| Fig.4(parity) | 🔄 **必须重出** | 数据源 parquet 已被 V8 覆盖;实测视觉无差(预测相关性≥0.994),标注末位变化 |
| Fig.5(SHAP) | ✅ 已验证逐位不变 | E07 最终模型 V7/V8 预测 bit-identical,重出结果相同 |
| Fig.6(稳定性) | ✅ 安全 | 数据有效 |
| **Fig.7(学习曲线)** | ⚠️ **风险最高** | 现图称"ERT 全程略优于 Stacking";V8 下 stacking(CatBoost CPU 化)已反超 ERT(E03 30.43<E01 31.99),**重跑后该排序声明很可能翻转**。现图与 V8 表格已自相矛盾 |
| Fig.8(校正 delta) | 🔄 **必须重出** | 数据源已覆盖;实测形态一致、校正收益略增强(−5.0%→−5.3%) |

---

## 5. 手稿审阅与修改建议(`Revised Manuscript-clear.docx`)

**总判断**: 手稿三条核心发现(熔体特征主导、物理先验增广收益、算法复杂度边际递减)全部
站得住,且手稿行文谨慎,**未**做"CatBoost 较差"或"增广对 CatBoost 收益最大"的断言——
恰好避开了被 V8 推翻的两条 V7 伪结论。但**手稿数字全部来自 V7,公开仓库已是 V8,投稿前
必须统一到 V8**。

### 5.1 必须修正的正文数字(已逐项复算 V7→V8)

| 位置 | 手稿现值 | V8 值 | 性质 |
|---|---|---|---|
| 摘要+§3.2 | T 降 **40.9%**(54.35→32.13) | **41.1%**(54.30→31.99) | 刷新 |
| 摘要+§3.2 | P 降 **16.1%**(2.43→2.04) | **16.1%**(2.41→2.03) | 百分比不变,绝对值微调 |
| §3.3 Table 4 | 增广 T 32.13→30.98(**−3.6%**) | 31.99→30.75(**−3.9%**) | 刷新 |
| §3.3 Table 4 | 增广 P 2.04→1.93(**−5.4%**) | 2.03→1.91(**−5.7%**) | 刷新;百分比须用未舍入值算 |
| §3.3 | t 检验 Δ=−1.15°C p=0.019;Δ=−0.10 kbar p=0.012 | Δ≈−1.24°C/−0.12 kbar,**p 值须用 V8 折结果重算** | **重算** |
| §3.4 Table 5 | 三算法 T 相差 "**<1°C**"(V7=0.95) | V8 最大差 **1.28°C** | **措辞必须改**,如 "≤1.3°C,仍小于折间标准差,统计上不可区分" |
| §3.4 Table 5 | P 最大差 "0.08 kbar" | 0.080 kbar | ✓ 不变 |
| §3.5 Table 6 | ERT 校正 T **−1.96°C**/P −0.16 kbar | **−2.06°C**/−0.158 kbar | 刷新(略增强) |
| §3.5 | CatBoost MBE "**−0.04°C**" | **+0.05°C** | 刷新(结论不变,数值与符号变) |
| §3.5 | ERT T MBE +1.21°C | +1.26°C | 刷新 |
| §3.4 | σT=6.83°C、σP=0.45 kbar | 不变(工件已验证有效) | ✓ 无需动 |

**Table 2 / S1 / S2(全矩阵)须整体从 V8 `metrics_summary.csv` 重生成**(E02/E05 CV 改善
7–15%、E03/E06 改善 4–5%)。

### 5.2 必须补充的方法学描述(与代码一致性)

1. **§3.1 分层 CV 缺一句稀疏 bin 合并说明**(代码会打印 "Bin merge: 222→39",论文不提即不符)。
2. **CatBoost 设备策略须文档化**(如 "CatBoost trained on CPU; GPU degrades accuracy at
   this dataset size")——这是 V8 数字成立的前提,也堵住"与旧预印本不同"的疑问。
3. **若 S1 表含 E10–E12 的 CV 列**: 说明 `correction_eval_mode=nested`(方法学加分项)。
   Table 6 是测试集指标,不受此影响。
4. **统计卫生**: 百分比统一用未舍入值计算再舍入展示(−5.4% 应为 −5.1%)。

### 5.3 图件: 见 §4.4。最低成本行动是 `python tools/plot_offline_figures.py` 重出全部图
(分钟级),Fig.4/8 必须,Fig.7 须配学习曲线探针。

### 5.4 数据可用性声明的版本问题

声明中 GitHub main 与 Zenodo DOI(10.5281/zenodo.20112388)冻结的都是 V7。若改用 V8 数字:
① 为 V8 打 release 标签(**不合并 main**);② Zenodo 发布新版本更新 DOI;③ 12 个 >50MB
的模型 joblib 建议迁 git-lfs 或精简。

### 5.5 一个与 V8 无关的疑似笔误

§3.1 "approximately **700–1600°C**",数据实际范围 **650–1700°C**(加载日志可证)。

### 5.6 无需担心的部分(审计背书)

熔体特征效应、ERT 增广收益、Balanced 无效、校正测试集结论、误差传播/稳定性分析——
均稳健或逐字节一致;§3.4 "复杂度无收益"在 V8 下因 CI 全重叠**更好论证**。

---

## 6. V8 vs main 文件变动清单

提交链(V8 比 main 多 6 个):
```
778bfe9 V8 初始整合      f029931 M4 S3/S4/S5+核验
8b21edc M3.2 H4/H6/H7     331fa5d V8 基线+冻结黄金镜像
50b3ec8 合并 M3.2         1a63296 快速档刷新 results_test
```

**总计: 205 文件,+4820 / −743 行。**

### 源码(2 新增 + 6 修改 + 3 工具)

| 文件 | 状态 | 对应改动 | 影响数字? |
|---|---|---|---|
| `src/runtime.py` | 新增(199) | H1+H2 | 否 |
| `src/repro.py` | 新增(70) | S6 | 否 |
| `src/protocol.py` | 改(+499) | S1+S2+H6 | **是** |
| `src/model_modules.py` | 改(+247) | H3+H7 | **是** |
| `main.py` | 改(+110) | H4+H6+横幅 | 否 |
| `config.py` | 改(+30) | S6 | 否 |
| `src/experiment_params.py` | 改(+40) | H4 | 否 |
| `src/correction_modules.py` | 改(+31) | S5 | 是(尾部) |
| `src/perturbation.py` | 改(+26) | S3 | 否 |
| `tools/run_{stability,learning_curve,error_propagation}.py` | 改 | S1 修复+礼让 | 否 |

### 测试(11 新增 + 3 修改,105→155)

新增: `test_runtime`、`test_repro`、`test_catboost_device`、`test_load_data`、
`test_perturbation`、`test_nested_correction`、`test_regression_vs_v7`、
`test_experiment_params`、`test_fit_reuse`、`test_stacking_parallel` +
`golden_v7_metrics_summary.csv`(V7 黄金镜像)。修改: `conftest`、
`test_correction_modules`、`test_model_modules`。

### 文档/配置

`SPEC_V8.md`(新,652)、`CHANGES_FROM_V7.md`(新,210)、`pytest.ini`(新)、
`README.md`(+25)、`requirements.txt`(+1,psutil)。

### 数据工件

`results/` 147 个文件被基线重跑覆盖(fold metrics / OOF parquet / 汇总表 / 持久化模型 /
config_used.yaml);`results_test/` 26 个新增(快速档冒烟,V7 未跟踪此目录)。

---

## 7. 远程仓库状态与隔离性(已密码学验证)

| 远程 ref | hash | 说明 |
|---|---|---|
| `main` | `004e4c7` | V7 论文版本,**零改动**(本地==远程,reflog 自论文提交后未移动) |
| `tag V7-paper` | →`004e4c7` | 论文版本固化标记(本会话新增,只是一个指针,不改 main 内容) |
| `tag V1.1` | `004e4c7` | 同指论文版本 |
| `V8` | `1a63296` | 全部 V8 工作,本地==origin/V8 |

隔离证据: `main..V8 = 6`、`V8..main = 0`(main 是 V8 纯祖先);我的所有提交
`git branch --contains` 只列 V8;origin/main 最后两次 push 在 2026-05(我介入前);
main 树里 results/ 仍是 V7 数据(E01 test rmse 仍为 91.44)。

---

## 8. 后续待办(按性价比排序)

1. **重出 V8 表格与图**: `python tools/plot_offline_figures.py`(分钟级)。Fig.4/8 必须,
   其余顺带一致化;Table 2/S1/S2 从 V8 `metrics_summary.csv` 重生成。
2. **学习曲线探针**: `python tools/run_learning_curve.py --models stacking --fractions 0.2
   --repeats 8 --n-splits 5` 与现有 fraction-0.2 stacking 行(T 47.82±1.25)对比;落在
   CI 内则保留旧图 + 方法注记,否则全量重跑(可顺便 `ML_STACKING_PARALLEL=2`)。
3. **论文文本修订**: 按 §5 改正文数字与措辞(CatBoost 相关结论、Fig.7 排序声明、组内
   排序改为 CI 重叠的诚实陈述);补两句方法学描述;重算 t 检验 p 值。
4. **若采用 V8 数字**: 为 V8 打 release 标签 + Zenodo 新版本 + 更新数据可用性 DOI
   (**不合并 main**);大模型 joblib(12 个 >50MB)考虑 git-lfs。
5. **不要重跑**: stability 与 error propagation(已证 V8 下逐位有效/无效用)。

---

## 附: 测试与验证状态

- 快测套件 **154/154** 通过(133 旧 + 21 新 M3.2 用例);slow 回归测试 **1 个通过**
  (默认 deselected,`pytest -m slow tests/test_regression_vs_v7.py`,V7 等价配置在 E01
  Liquid 上对照黄金镜像逐位复现)。合并 M3.2 后主仓库 V8 全套(含 slow)共 **155 通过**。
  (注: 基线提交后、M3.2 合并前的中间快照曾为 134,已被合并后的 155 取代。)
- 运行主基准: `python main.py`(完整 24 实验,~小时级);冒烟: `python main.py --test`(秒级)。
- 复现 V7 数字: `nested_correction=False, merge_sparse_bins=False` 走 V7 等价路径。
