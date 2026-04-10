# -*- coding: utf-8 -*-
"""Shared utilities: EPMA perturbation, P-T splitters, experiment parameter
builders, and logging helpers.

This module consolidates four short utility modules into a single cohesive
file so that every public helper used by the pipeline lives in one place.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# Perturbation (EPMA measurement-error model)
# ============================================================

DEFAULT_OXIDE_REL_ERR = {
    'SiO2.cpx': 0.03,
    'TiO2.cpx': 0.08,
    'Al2O3.cpx': 0.03,
    'Cr2O3.cpx': 0.08,
    'FeO.cpx': 0.03,
    'MgO.cpx': 0.03,
    'MnO.cpx': 0.08,
    'CaO.cpx': 0.03,
    'Na2O.cpx': 0.08,
    'SiO2.liq': 0.03,
    'TiO2.liq': 0.08,
    'Al2O3.liq': 0.03,
    'FeO.liq': 0.03,
    'MgO.liq': 0.03,
    'MnO.liq': 0.08,
    'CaO.liq': 0.03,
    'Na2O.liq': 0.08,
    'K2O.liq': 0.08,
}


def get_rel_err_vector(
    feature_names: List[str],
    oxide_rel_err: Optional[dict] = None,
    default_rel_err: Optional[float] = None,
    strict: bool = True,
) -> np.ndarray:
    """Build a per-feature relative-error vector aligned with ``feature_names``."""
    if oxide_rel_err is None:
        oxide_rel_err = DEFAULT_OXIDE_REL_ERR

    missing = [name for name in feature_names if name not in oxide_rel_err]
    if missing and (strict or default_rel_err is None):
        raise ValueError(f"Missing EPMA error mapping for feature names: {missing}")

    return np.array([
        oxide_rel_err.get(name, default_rel_err)
        for name in feature_names
    ])


def epma_perturb(
    X: np.ndarray,
    rel_err_vec: np.ndarray,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Apply Gaussian relative-error perturbation to EPMA compositional data."""
    scale = rel_err_vec * np.abs(X)
    noise = rng.normal(0.0, scale, size=X.shape)
    return X + noise


def perturbation_with_repeats(
    X: np.ndarray,
    y: np.ndarray,
    rel_err_vec: np.ndarray,
    n_perturbations: int = 15,
    rng: Optional[np.random.RandomState] = None,
    random_seed: int = 42,
    include_original: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand a dataset with ``n_perturbations`` independent EPMA-error realizations."""
    if rng is None:
        rng = np.random.RandomState(random_seed)

    if include_original:
        X_list = [X]
        y_list = [y]
        for _ in range(n_perturbations):
            X_perturbed = epma_perturb(X, rel_err_vec, rng)
            X_list.append(X_perturbed)
            y_list.append(y)
        X_aug = np.vstack(X_list)
        y_aug = np.concatenate(y_list)
    else:
        X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
        y_aug = np.repeat(y, repeats=n_perturbations, axis=0)
        scale = rel_err_vec * np.abs(X_rep)
        X_aug = rng.normal(X_rep, scale)

    return X_aug, y_aug


# ============================================================
# P-T splitters (stratified hold-out sampling)
# ============================================================

@dataclass(frozen=True)
class PTBins:
    """Grid edges for P-T stratification."""

    p_edges: np.ndarray
    t_edges: np.ndarray

    @property
    def n_p_bins(self) -> int:
        return max(len(self.p_edges) - 1, 0)

    @property
    def n_t_bins(self) -> int:
        return max(len(self.t_edges) - 1, 0)


def compute_pt_edges(y_t: np.ndarray, y_p: np.ndarray) -> PTBins:
    """Return a square P-T bin grid sized by ``sqrt(n_samples)``."""
    n_samples = len(y_t)
    if n_samples == 0:
        raise ValueError("Empty input for P-T binning.")

    k = int(np.ceil(np.sqrt(n_samples)))

    p_min = float(np.min(y_p)) - 0.1
    p_max = float(np.max(y_p)) + 0.1
    p_edges = np.linspace(p_min, p_max, k)
    p_edges = np.round(p_edges, 1)

    t_min = float(np.min(y_t)) - 1.0
    t_max = float(np.max(y_t)) + 1.0
    t_edges = np.linspace(t_min, t_max, k)
    t_edges = np.round(t_edges, 0)

    return PTBins(p_edges=p_edges, t_edges=t_edges)


def assign_pt_bins(y_t: np.ndarray, y_p: np.ndarray, bins: PTBins) -> np.ndarray:
    """Assign each sample to its 2D P-T bin id."""
    if bins.n_p_bins <= 0 or bins.n_t_bins <= 0:
        raise ValueError("Invalid P-T bin edges.")

    p_bins = np.digitize(y_p, bins.p_edges[1:-1], right=False)
    t_bins = np.digitize(y_t, bins.t_edges[1:-1], right=False)

    return p_bins * bins.n_t_bins + t_bins


def select_test_indices(
    tp_bins: np.ndarray,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Pick one representative sample per P-T bin to form a fixed hold-out set."""
    rng = np.random.RandomState(random_state)

    bin_to_indices: Dict[int, np.ndarray] = {}
    for bin_id in np.unique(tp_bins):
        idxs = np.where(tp_bins == bin_id)[0]
        if idxs.size > 0:
            bin_to_indices[int(bin_id)] = idxs

    test_idx_list = []
    for bin_id in sorted(bin_to_indices.keys()):
        idxs = bin_to_indices[bin_id]
        picked = rng.choice(idxs)
        test_idx_list.append(picked)

    return np.array(test_idx_list, dtype=int)


# ============================================================
# Experiment parameter builders
# ============================================================

def build_model_params(
    base_config: Dict[str, Any],
    model_module: str,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a model parameter dict from config defaults and a module name.

    Parameters
    ----------
    base_config:
        Top-level config dict (from ``get_config_dict()``).
    model_module:
        Module name key, e.g. ``'ert'``, ``'catboost'``, ``'stacking'``.
    random_seed:
        If provided, injected as ``random_seed`` in the returned dict.
        When ``None`` the key is omitted and ``protocol.apply_seed`` will
        inject the per-target / per-fold seed at runtime.
    """
    model_defaults = base_config["model_defaults"]
    name = model_module.lower()

    if name in {"ert", "extratrees"}:
        params: Dict[str, Any] = dict(model_defaults["ert"])
    elif name in {"rf", "randomforest"}:
        params = dict(model_defaults["rf"])
    elif name in {"catboost", "cb"}:
        params = dict(model_defaults["catboost"])
    elif name == "stacking":
        stacking_params = dict(model_defaults.get("stacking", {}))
        base_params = {
            "ert": dict(model_defaults["ert"]),
            "catboost": dict(model_defaults["catboost"]),
            "rf": dict(model_defaults["rf"]),
        }
        for key, override in model_defaults.get("stacking_base_defaults", {}).items():
            if key in base_params and isinstance(override, dict):
                base_params[key].update(override)
        params = {"base_model_params": base_params}
        if stacking_params:
            params.update({
                "inner_cv": stacking_params.get("inner_cv"),
                "use_meta_scaler": stacking_params.get("use_meta_scaler"),
            })
    else:
        params = {}

    # Drop any n_jobs key that is explicitly None so each model's own
    # runtime default is used (see src.runtime.get_default_n_jobs).
    if "n_jobs" in params and params["n_jobs"] is None:
        params.pop("n_jobs")

    if random_seed is not None:
        params["random_seed"] = random_seed
    return params


def build_data_params(
    base_config: Dict[str, Any],
    data_module: str,
    feature_set: str,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a data-module parameter dict from config defaults."""
    params: Dict[str, Any] = {
        "feature_names": list(base_config["feature_sets"][feature_set]),
    }
    if data_module.lower() == "augmented":
        params["n_aug"] = base_config["augmentation"]["n_aug"]
    if random_seed is not None:
        params["random_seed"] = random_seed
    return params


# ============================================================
# Logging
# ============================================================

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-5s | %(message)s"
CONSOLE_DATE_FORMAT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s | %(levelname)-5s | %(name)s | %(message)s"
FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColorFormatter(logging.Formatter):
    """Console formatter with optional ANSI color level names."""

    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m",
    }

    def __init__(self, fmt: str, datefmt: str, use_color: bool = True):
        super().__init__(fmt, datefmt)
        self.use_color = use_color and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        if sys.platform == "win32":
            try:
                import ctypes

                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
                return True
            except Exception:
                return False
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_color:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            reset = self.COLORS["RESET"]
            record.levelname = f"{color}{record.levelname}{reset}"
        return super().format(record)


_LOG_DIR: Optional[Path] = None
_INITIALIZED: bool = False
_DEFAULT_MAX_BYTES = 10 * 1024 * 1024
_DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    log_dir: Optional[str] = None,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: Optional[str] = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
    force: bool = False,
) -> None:
    """Initialize root logging with colored console and rotating file outputs."""
    global _LOG_DIR, _INITIALIZED

    if _INITIALIZED and not force:
        return

    if log_dir is None:
        from config import CONFIG
        log_dir = CONFIG.output.log_dir

    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"benchmark_{timestamp}.log"

    log_path = _LOG_DIR / log_filename

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ColorFormatter(CONSOLE_FORMAT, CONSOLE_DATE_FORMAT))
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT, FILE_DATE_FORMAT))
    root_logger.addHandler(file_handler)

    _INITIALIZED = True
    root_logger.debug(f"Logging initialized, log file: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger, initializing global logging on first use."""
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
