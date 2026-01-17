# Repository Guidelines

## Project Structure
- `src/`: Core Python modules (models, preprocessing, metrics, corrections, visualization, runner).
- `notebooks/`: Jupyter entry points, especially `notebooks/run_experiments.ipynb`.
- `outputs/`: Generated experiment artifacts (metrics, predictions, figures).
- `reference_files/`: Reference scripts/papers.
- Root files: `config.py` (experiment configs), `input.csv` (dataset), `requirements.txt`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: Install Python dependencies.
- Open and run `notebooks/run_experiments.ipynb`: Primary workflow for running the full experiment matrix.
- Quick run from a shell (example):
  `python -c "from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner; df=load_data('input.csv', encoding='latin-1'); data=prepare_data(df, feature_mode='cpx_liq'); cfg=ExperimentConfig(exp_name='exp1_baseline', model_type='catboost'); ExperimentRunner(cfg).run_experiment(X=data['X'], y_T=data['y_T'], y_P=data['y_P'], groups=data['groups'], row_ids=data['row_ids'], refs=data['refs'])"`

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep functions small and focused.
- Prefer `snake_case` for functions/variables and `PascalCase` for classes.
- No formatter or linter config is present; keep style consistent with existing `src/` modules.

## Testing Guidelines
- No automated test suite is currently defined.
- Validate changes by running the notebook workflow and checking `outputs/*/metrics.csv` and `outputs/*/preds.parquet`.
- Use small-scope checks when possible (e.g., a single experiment config in `config.py`).

## Commit & Pull Request Guidelines
- Commit subjects are short, descriptive, and may be in Chinese or English; follow that pattern (e.g., “add exp3_corr_only metrics” or “initial”).
- PRs should include: brief summary, how to run/verify (notebook cell or command), and any new output artifacts generated.
- Avoid committing large generated outputs unless they are needed for reproducibility or review.

## Configuration & Data Notes
- Experiment parameters live in `config.py`; keep new configs consistent with existing keys.
- `input.csv` is treated as the canonical dataset; document any changes to its schema or encoding.

## Project Requirements & Implemented Changes
- Normalize metric keys to `T_rmse` / `P_rmse` across runner output, summaries, and plots.
- Fix Stacking cache hashing to include `groups` and model signature to avoid cache reuse across configs.
- Ensure Stacking can run without explicit `base_models` by falling back to the default stacker.
- Keep modules importable after cleaning corrupted comment/docstring characters.
- Stacking defaults to `n_jobs=1` to avoid Windows multiprocessing permission errors.

## Full Test Run (Ref + cpx_liq)
- Data: full `input.csv`; grouping: `Ref`; features: `cpx_liq`.
- CV: outer `GroupKFold` = 5; inner (Stacking) `GroupKFold` = 5.
- Command used:
```bash
python - <<'PY'
import pandas as pd
from src import load_data, prepare_data, ExperimentConfig, ExperimentRunner
from config import EXPERIMENT_CONFIGS

exp_names = ['exp1_baseline', 'exp2_aug_only', 'exp3_corr_only', 'exp4_aug_corr', 'exp5_stacking']
df = load_data('input.csv', encoding='latin-1')
data = prepare_data(df, feature_mode='cpx_liq')

results_list = []
for exp_name in exp_names:
    config = ExperimentConfig(exp_name=exp_name, **EXPERIMENT_CONFIGS[exp_name])
    runner = ExperimentRunner(config)
    results = runner.run_experiment(
        X=data['X'], y_T=data['y_T'], y_P=data['y_P'],
        groups=data['groups'], row_ids=data['row_ids'], refs=data['refs']
    )
    results_list.append(results)

summary_df = pd.DataFrame(results_list)
summary_df.to_csv('outputs/summary_all.csv', index=False)
PY
```
- Results (from `outputs/summary_all.csv`):
```
exp_name          T_rmse_mean  T_mae_mean  T_r2_mean  P_rmse_mean  P_mae_mean  P_r2_mean
exp1_baseline       45.6925     33.4233     0.8810      2.8523      2.0196      0.8399
exp2_aug_only       45.8487     33.6447     0.8800      2.8306      1.9998      0.8417
exp3_corr_only      45.5629     33.3535     0.8817      2.8347      1.9939      0.8414
exp4_aug_corr       45.7625     33.5980     0.8804      2.8165      1.9796      0.8429
exp5_stacking       46.2476     34.2396     0.8778      2.8687      1.9472      0.8363
```
