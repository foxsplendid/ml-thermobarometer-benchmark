# Reproducibility Guide

This document describes how to reproduce all results reported in the paper from scratch.

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.9+ is required. A CUDA-capable GPU is optional but will accelerate CatBoost experiments.

## Environment variables

The training pipeline is controlled by three environment variables so you can tune parallelism without changing code:

| Variable | Default | Meaning |
|---|---|---|
| `ML_N_JOBS` | `4` | Threads per sklearn model (ExtraTrees, RF) |
| `ML_FOLD_WORKERS` | `max(1, cpu_count / ML_N_JOBS)` auto-detected | Concurrent folds in `StratifiedCVProtocol` |
| `ML_FOLD_BACKEND` | `loky` | joblib backend when `ML_FOLD_WORKERS > 1` |

### Recommended settings by machine type

| Machine | `ML_N_JOBS` | `ML_FOLD_WORKERS` | Notes |
|---|---|---|---|
| Laptop (4–8 cores) | `2` | `2` | Keep thermal headroom |
| Workstation (16 cores) | `4` | `4` | 4×4 = 16 threads |
| Server (32 cores) | `4` | `8` | 4×8 = 32 threads |
| Any machine | `-1` | `1` | All cores for one fold — only efficient if `n_estimators` is large |

> **Note**: avoid `ML_N_JOBS=-1` on high-core-count machines (≥ 16 cores). sklearn's loky
> worker pool spawns one process per core; for 200 trees on 32 cores each process trains only
> 6 trees, and the inter-process overhead dominates. A moderate `ML_N_JOBS=4` combined with
> `ML_FOLD_WORKERS` is almost always faster and gives higher apparent utilization.

Example — 32-core server:

```bash
set ML_N_JOBS=4
set ML_FOLD_WORKERS=8
python main.py
```

> `ML_FOLD_BACKEND` defaults to `loky` (isolated processes per fold).  Do
> **not** set it to `threading` when CatBoost experiments are present —
> CatBoost uses process-level thread-local state that causes crashes under
> thread-based fold parallelism.

## Step 1 — Validate the installation (≈ 2 min)

```bash
python main.py --test
```

Runs 4 experiments × 2 folds on both feature sets and prints a summary table.
Expected output: non-NaN RMSE and R² values in the summary.

## Step 2 — Full benchmark (≈ 1–2 hours)

```bash
python main.py
```

Runs all 24 experiments (12 module combinations × 2 feature sets) with 10-fold CV.
Outputs land in `results/`:

- `metrics_summary.csv` — per-experiment CV and test-set metrics
- `effect_table.csv` — delta RMSE relative to baseline (ERT + raw + no correction)
- `config_used.yaml` — exact hyperparameters, data shape, git commit, and library versions
- `models/*.joblib` — serialised final pipelines (one per experiment × target)

## Step 3 — Sub-experiments

All sub-experiments write into subdirectories of `results/` (or a custom `--output-dir`).

### Stability analysis (bootstrapped repeat train/test splits)

```bash
python tools/run_stability.py \
    --exp-id E07_ert_augmented_none_liq \
    --model-module ert \
    --data-module augmented \
    --corr-module none \
    --feature-set Liquid \
    --n-repeats 1000
```

For distributed runs use `--repeat-start` / `--repeat-end` to partition the 1 000 repeats across nodes, then merge:

```bash
python tools/run_stability.py --exp-id <id> --merge-dir results/stability --output-dir results
```

### Learning-curve analysis

```bash
python tools/run_learning_curve.py \
    --feature-set Liquid \
    --models ert stacking \
    --repeats 30 \
    --n-splits 10
```

### Monte Carlo error propagation

```bash
python tools/run_error_propagation.py \
    --exp-id E07_ert_augmented_none_liq \
    --model-module ert \
    --data-module augmented \
    --corr-module none \
    --feature-set Liquid \
    --n-mc 1000
```

### Figures

```bash
python tools/plot_offline_figures.py --selected-only
```

Reads existing result files and regenerates all paper figures into `results/figures/`.

## Seeding strategy

- Base seed: `config.py → CVConfig.random_seed = 42`
- T target seed: `base_seed` (42)
- P target seed: `base_seed + P_SEED_OFFSET` (42 + 1000 = 1042)
- Per-fold seed: `target_seed + fold_idx` (0-indexed)
- Stability per-repeat seed: `target_seed_base + repeat_id`
- MC per-repeat seed: passed explicitly to `MCUncertaintyEstimator`

These offsets are large enough to avoid seed overlap across all expected run configurations.

## Verifying a result

After `python main.py`, load `results/metrics_summary.csv` and compare the
`T_rmse_mean` and `P_rmse_mean` columns against Table X in the paper.
Numeric results may differ slightly across OS/library versions due to floating-point
non-determinism in BLAS; differences beyond ±0.5 % warrant investigation.
