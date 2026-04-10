# -*- coding: utf-8 -*-
"""Shared helpers for tool scripts (run_stability, run_learning_curve, etc.).

Import this module *after* the sys.path bootstrap that each tool performs.
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Ensure repo root is always on the path when tools are run directly.
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from config import get_config_dict
from src.protocol import ExperimentConfig
from src.utils import build_data_params, build_model_params, get_logger, setup_logging
from src.runtime import log_runtime_info

BASE_CONFIG = get_config_dict()


# ---------------------------------------------------------------------------
# Experiment config builder
# ---------------------------------------------------------------------------

def build_experiment_config(
    exp_id: str,
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
    random_seed: int,
) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from CLI-level string arguments."""
    model_params = build_model_params(BASE_CONFIG, model_module, random_seed)
    data_params = build_data_params(BASE_CONFIG, data_module, feature_set, random_seed)

    return ExperimentConfig(
        exp_id=exp_id,
        data_module_name=data_module,
        model_module_name=model_module,
        corr_module_name=corr_module,
        feature_set=feature_set,
        data_params=data_params,
        model_params=model_params,
    )


# ---------------------------------------------------------------------------
# Common argparse arguments
# ---------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add experiment-identity and I/O arguments shared across all tools.

    Adds: ``--exp-id``, ``--data-module``, ``--model-module``,
    ``--corr-module``, ``--feature-set``, ``--data-path``,
    ``--output-dir``, ``--n-splits``, ``--random-seed``.
    """
    parser.add_argument(
        "--exp-id",
        default="experiment",
        help="Experiment ID used as filename prefix for outputs.",
    )
    parser.add_argument(
        "--data-module",
        default="augmented",
        choices=["raw", "balanced", "augmented"],
        help="Data module (default: augmented).",
    )
    parser.add_argument(
        "--model-module",
        default="ert",
        choices=["ert", "extratrees", "catboost", "rf", "randomforest", "stacking"],
        help="Model module (default: ert).",
    )
    parser.add_argument(
        "--corr-module",
        default="none",
        choices=["none", "segmented"],
        help="Correction module (default: none).",
    )
    parser.add_argument(
        "--feature-set",
        default="Liquid",
        choices=["NoLiquid", "Liquid"],
        help="Feature set (default: Liquid).",
    )
    parser.add_argument(
        "--data-path",
        default=BASE_CONFIG["data_path"],
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default=BASE_CONFIG["output_dir"],
        help="Output directory (default: results/).",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=BASE_CONFIG["n_splits"],
        help="Number of CV folds (default: 10).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=BASE_CONFIG["random_seed"],
        help="Base random seed (default: 42).",
    )
    return parser


# ---------------------------------------------------------------------------
# Logging initialisation
# ---------------------------------------------------------------------------

def init_tool_logging(tool_name: str) -> logging.Logger:
    """Initialise file+console logging for a tool script.

    ``tool_name`` is used as the log filename prefix (e.g. ``'stability'``).
    Also logs a runtime info line after setup.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{tool_name}_{timestamp}_{os.getpid()}.log"
    setup_logging(log_filename=log_filename)
    log_runtime_info()
    return get_logger(tool_name)
