# -*- coding: utf-8 -*-
"""Hardware detection and parallel-budget allocation.

Single source of truth for "how many cores / GPUs do we have, and how many
should this caller use right now". All modules that need n_jobs / device
selection should go through this module instead of calling os.cpu_count().

Environment variables (precedence order, highest first):
  ML_N_JOBS         Per-model thread cap (V7 compat). Hard upper bound.
  ML_RESERVE_CORES  Cores to leave free for sibling processes. V8 addition.
  ML_OUTER_PROCS    Number of sibling outer processes (e.g. parallel
                    experiments) sharing this machine. V8 addition.
                    Only the 'cross_proc' context divides by it.
  ML_STACKING_PARALLEL  Concurrent (fold x base-model) fit workers inside
                    StrictOOFStacking (H7, default 1 = sequential). Parsed
                    in model_modules via this module's _env_int. Workers are
                    clamped to suggest_n_jobs('inner_loop'); each worker's
                    thread budget defaults to budget // workers, but params
                    that explicitly carry a thread count are not reduced
                    (config.ModelDefaults.ert pins n_jobs=4), so keep
                    workers <= budget // 4 with the shipped config (a
                    warning is logged otherwise). CatBoost 'auto' device is
                    pinned to CPU under parallel workers.
  ML_CATBOOST_GPU_MIN_SAMPLES  Training-set size below which CatBoost
                    task_type='auto' stays on CPU (H3, default 5000).
                    Parsed in model_modules._get_catboost_task_type.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Runtime probe
# ============================================================

@dataclass(frozen=True)
class Runtime:
    n_physical_cores: int
    n_logical_cores: int
    has_cuda_gpu: bool
    n_gpu_devices: int
    gpu_mem_gb: Optional[float]
    ram_gb: Optional[float]
    platform: str


def _probe_cores() -> tuple[int, int]:
    logical = os.cpu_count() or 1
    physical = logical
    try:
        import psutil  # type: ignore
        p = psutil.cpu_count(logical=False)
        if p:
            physical = int(p)
    except Exception:
        pass
    return physical, logical


def _probe_ram_gb() -> Optional[float]:
    try:
        import psutil  # type: ignore
        return round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        return None


def _probe_gpu() -> tuple[bool, int, Optional[float]]:
    # Try torch first (most reliable when present)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            mem = None
            try:
                mem = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2)
            except Exception:
                pass
            return True, n, mem
    except Exception:
        pass
    # Fall back to catboost's detector
    try:
        from catboost.utils import get_gpu_device_count  # type: ignore
        n = int(get_gpu_device_count())
        if n > 0:
            return True, n, None
    except Exception:
        pass
    return False, 0, None


@lru_cache(maxsize=1)
def get_runtime() -> Runtime:
    physical, logical = _probe_cores()
    has_gpu, n_gpu, gpu_mem = _probe_gpu()
    return Runtime(
        n_physical_cores=physical,
        n_logical_cores=logical,
        has_cuda_gpu=has_gpu,
        n_gpu_devices=n_gpu,
        gpu_mem_gb=gpu_mem,
        ram_gb=_probe_ram_gb(),
        platform=sys.platform,
    )


def runtime_summary_str() -> str:
    r = get_runtime()
    gpu = f"{r.n_gpu_devices}xCUDA ({r.gpu_mem_gb}GB)" if r.has_cuda_gpu else "no-GPU"
    ram = f"{r.ram_gb}GB" if r.ram_gb else "?GB"
    return (
        f"Runtime: {r.platform} | {r.n_physical_cores} phys / "
        f"{r.n_logical_cores} log cores | RAM {ram} | {gpu}"
    )


# ============================================================
# Parallel budget allocator
# ============================================================

Context = Literal["model", "inner_loop", "cross_proc"]


def _env_int(name: str, default: Optional[int] = None) -> Optional[int]:
    val = os.environ.get(name)
    if val is None or val == "":
        return default
    try:
        return int(val)
    except ValueError:
        logger.warning("Ignoring non-integer %s=%r", name, val)
        return default


def suggest_n_jobs(
    context: Context = "model",
    reserve_cores: Optional[int] = None,
) -> int:
    """Return an n_jobs suitable for `context` on this machine.

    Decision order:
      1. If ML_N_JOBS is set, return min(value, computed).  Treat <=0 as "all".
      2. Compute base = physical_cores - reserve.
      3. For 'cross_proc', divide by ML_OUTER_PROCS (default 1).
      4. Clamp to [1, physical_cores].

    Parameters
    ----------
    context:
        - 'model'      : single estimator's main parallelism.
        - 'inner_loop' : nested loop inside an outer-parallel scope.
                         Currently identical to 'model' but kept separate
                         so future tuning can split them.
        - 'cross_proc' : called inside one of N sibling processes; divide
                         the core budget by ML_OUTER_PROCS.
    reserve_cores:
        Explicit override of cores to leave free. If None, read
        ML_RESERVE_CORES (default 0).
    """
    r = get_runtime()
    cores = r.n_physical_cores

    if reserve_cores is None:
        reserve_cores = _env_int("ML_RESERVE_CORES", 0) or 0

    base = max(1, cores - max(0, reserve_cores))

    if context == "cross_proc":
        outer = max(1, _env_int("ML_OUTER_PROCS", 1) or 1)
        base = max(1, base // outer)

    cap = _env_int("ML_N_JOBS", None)
    if cap is not None:
        # V7 convention: -1 means "all available".
        if cap <= 0:
            cap = cores
        base = min(base, cap)

    return max(1, min(base, cores))


# ============================================================
# Standalone smoke test
# ============================================================

if __name__ == "__main__":
    print(runtime_summary_str())
    for ctx in ("model", "inner_loop", "cross_proc"):
        print(f"  suggest_n_jobs({ctx!r}) = {suggest_n_jobs(ctx)}")
