# -*- coding: utf-8 -*-
"""Runtime detection and parallelism configuration.

Reads environment variables so that the same codebase can run efficiently on
different machines (laptop, workstation, HPC node) without code changes.

Environment variables
---------------------
ML_N_JOBS : int
    Number of threads used *per model* by sklearn parallel estimators
    (ExtraTrees, RandomForest, etc.).  Defaults to ``4``.  Set to ``-1``
    to use all logical cores (not recommended on high-core-count machines
    due to loky worker overhead).
ML_FOLD_WORKERS : int
    Number of folds that may run concurrently (>1 enables fold-level joblib
    parallelism in StratifiedCVProtocol).  When not set, auto-detected as
    ``max(1, cpu_count // ML_N_JOBS)`` so the total thread count stays within
    the available core budget.  Example: 32 cores, ML_N_JOBS=4 → 8 workers.
    Set to ``1`` to force sequential execution.
ML_FOLD_BACKEND : str
    joblib backend used for fold-level parallelism when ML_FOLD_WORKERS > 1.
    Accepted values: ``loky``, ``threading``, ``multiprocessing``.
    Defaults to ``loky`` (isolated processes per fold; required when
    CatBoost experiments are present — CatBoost is not thread-safe across
    parallel instances in the same process).
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_n_jobs() -> int:
    """Return per-model thread count from ``ML_N_JOBS`` (default: ``4``)."""
    env_val = os.environ.get("ML_N_JOBS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            logger.warning("ML_N_JOBS='%s' is not a valid integer; using 4", env_val)
    return 4


def get_fold_workers() -> int:
    """Return fold-level worker count from ``ML_FOLD_WORKERS``.

    When the variable is not set, auto-detects as
    ``max(1, cpu_count // n_jobs)`` where ``n_jobs`` is the per-model thread
    budget from :func:`get_n_jobs`.  This keeps the total thread count within
    the available core count on any machine:

    - 4-core  (n_jobs=4): 1 worker  (sequential)
    - 8-core  (n_jobs=4): 2 workers
    - 16-core (n_jobs=4): 4 workers
    - 32-core (n_jobs=4): 8 workers

    Set ``ML_FOLD_WORKERS=1`` to force sequential execution.
    """
    env_val = os.environ.get("ML_FOLD_WORKERS")
    if env_val is not None:
        try:
            n = int(env_val)
            if n < 1:
                logger.warning("ML_FOLD_WORKERS=%d < 1; using 1 (sequential)", n)
                return 1
            return n
        except ValueError:
            logger.warning(
                "ML_FOLD_WORKERS='%s' is not a valid integer; using auto-detect", env_val
            )
    # Auto-detect: divide available cores by per-model thread budget
    cpu_count = os.cpu_count() or 1
    n_jobs = get_n_jobs()
    n_per_worker = n_jobs if n_jobs > 0 else cpu_count
    return max(1, cpu_count // n_per_worker)


def get_fold_backend() -> str:
    """Return the joblib backend name from ``ML_FOLD_BACKEND`` (default: ``'loky'``).

    ``loky`` (default) runs each fold in an isolated worker process.  This is
    the only safe choice when the experiment contains CatBoost, because
    CatBoost uses process-level thread-local state that is not safe to share
    across threads.  The per-fold loky overhead is negligible compared to the
    minutes of work each fold performs.

    ``threading`` is faster for fold workers that contain *only* sklearn tree
    models, but will segfault or deadlock when CatBoost is present.
    """
    valid = {"loky", "threading", "multiprocessing"}
    env_val = os.environ.get("ML_FOLD_BACKEND", "loky").strip().lower()
    if env_val not in valid:
        logger.warning(
            "ML_FOLD_BACKEND='%s' is not recognised; using 'loky'. "
            "Valid choices: %s",
            env_val, ", ".join(sorted(valid)),
        )
        return "loky"
    return env_val


def log_runtime_info() -> None:
    """Log a one-line summary of the active runtime configuration."""
    import multiprocessing
    n_cpus = multiprocessing.cpu_count()
    n_jobs = get_n_jobs()
    fold_workers = get_fold_workers()
    fold_backend = get_fold_backend()

    n_jobs_display = "all" if n_jobs == -1 else str(n_jobs)
    logger.info(
        "Runtime: %d logical CPUs | ML_N_JOBS=%s | ML_FOLD_WORKERS=%d | ML_FOLD_BACKEND=%s",
        n_cpus, n_jobs_display, fold_workers, fold_backend,
    )

    # Attempt GPU detection via CatBoost (non-fatal if unavailable)
    try:
        from catboost.utils import get_gpu_device_count
        n_gpu = get_gpu_device_count()
        logger.info("GPU devices visible to CatBoost: %d", n_gpu)
    except Exception:
        pass
