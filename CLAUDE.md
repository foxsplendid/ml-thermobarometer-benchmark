# .md

This file provides guidance to  Code (.ai/code) when working with code in this repository.

## Project Overview

ML Thermobarometer Benchmark — a framework for building and evaluating machine-learning thermobarometer models. It predicts temperature (T) and pressure (P) from geochemical composition data (clinopyroxene ± liquid), comparing different data processing strategies, ML models, and post-training correction techniques under a reproducible cross-validation protocol.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Quick validation (~2 min, 4 experiments, 2 folds)
python main.py --test

# Full benchmark (~1-2 hours, 24 experiments, 10-fold CV)
python main.py

# Sub-experiments (tools/)
python tools/run_stability.py --exp-id <id> --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-repeats 1000
python tools/run_learning_curve.py --feature-set Liquid --models ert stacking --repeats 30 --n-splits 10
python tools/run_error_propagation.py --exp-id <id> --model-module ert --data-module augmented --corr-module none --feature-set Liquid --n-mc 1000
python tools/plot_offline_figures.py --selected-only

# Tests
pytest tests/ -v
pytest tests/ --cov=src
pytest tests/ --runslow   # includes slow integration tests
```

## Architecture

The codebase is built around four pluggable module types defined in `src/interfaces.py`:

| Module | Interface | Implementations |
|--------|-----------|-----------------|
| **M1 Data** | `DataModule` | `RawDataModule`, `BalancedDataModule`, `AugmentedDataModule` |
| **M2 Model** | `ModelModule` | `ExtraTreesModel`, `CatBoostModel`, `RandomForestModel`, `SVRModel`, `StrictOOFStacking`, `RidgeModel` |
| **M3 Correction** | `CorrectionModule` | `NoCorrection`, `SegmentedLinearCorrector` |
| **M4 Uncertainty** | `UncertaintyModule` | `MCUncertaintyEstimator` |

### Execution flow

1. **`main.py`** builds an `ExperimentMatrix` (24 experiments = 12 module combinations × 2 feature sets)
2. **`src/protocol/`** — `ExperimentMatrix` drives `StratifiedCVProtocol` which runs `Pipeline` per fold
3. **`Pipeline`** chains DataModule → ModelModule → CorrectionModule per target (T and P separately)
4. **`src/utils.py`** — P-T grid stratification, perturbation, experiment param builders, and logging

### Feature sets

- **NoLiquid** (9 features): Clinopyroxene oxides only
- **Liquid** (18 features): Clinopyroxene + liquid oxides

### Experiment naming

Experiments follow the pattern `E{01-12}_{model}_{data}_{correction}_{feature_set}`. E01–E03 use raw data, E04–E06 balanced, E07–E09 augmented (no correction), E10–E12 augmented + segmented correction.

### Key files

- **`config.py`** — All hyperparameters and paths in dataclass form; `get_config_dict()` is the public accessor
- **`src/utils.py`** — `build_model_params()` / `build_data_params()`, P-T splitters, perturbation, logging
- **`src/runtime.py`** — CPU/GPU detection and parallelism knobs (`ML_N_JOBS`, `ML_FOLD_WORKERS`, `ML_FOLD_BACKEND`)
- **`src/protocol/pipeline.py`** — `Pipeline`, `StratifiedCVProtocol`, fold-level helpers
- **`src/protocol/matrix.py`** — `ExperimentConfig`, `ExperimentMatrix`
- **`src/metrics.py`** — `compute_all_metrics()`, `summarize_folds()`
- **`tools/_common.py`** — Shared argparse args, config builder, and logging init for all tool scripts

### Outputs (`results/`)

`metrics_summary.csv`, `effect_table.csv`, serialized models (`models/*.joblib`), stability/learning-curve/error-propagation subdirectories, and figures.

### Reproducibility

See `REPRODUCIBILITY.md` for the full step-by-step guide, seeding strategy, and environment variable reference.
