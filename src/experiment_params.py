# -*- coding: utf-8 -*-
"""Shared helpers for building experiment parameter dictionaries."""

from typing import Any, Dict, Optional


def build_model_params(
    base_config: Dict[str, Any],
    model_module: str,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build model parameter dict from config defaults and module name.

    Parameters
    ----------
    base_config:
        Top-level config dict (from ``get_config_dict()``).
    model_module:
        Module name key, e.g. ``'ert'``, ``'catboost'``, ``'stacking'``.
    random_seed:
        If provided, injected as ``random_seed`` in the returned dict.
        When ``None`` the key is omitted and ``protocol._apply_seed`` will
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

    if random_seed is not None:
        params["random_seed"] = random_seed
    return params


def build_data_params(
    base_config: Dict[str, Any],
    data_module: str,
    feature_set: str,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build data-module parameter dict from config defaults.

    Parameters
    ----------
    base_config:
        Top-level config dict (from ``get_config_dict()``).
    data_module:
        Module name key, e.g. ``'raw'``, ``'balanced'``, ``'augmented'``.
    feature_set:
        Key into ``base_config['feature_sets']``, e.g. ``'Liquid'``.
    random_seed:
        If provided, injected as ``random_seed`` in the returned dict.
        When ``None`` the key is omitted and ``protocol._apply_seed`` will
        inject the per-target / per-fold seed at runtime.
    """
    params: Dict[str, Any] = {
        "feature_names": list(base_config["feature_sets"][feature_set]),
    }
    if data_module.lower() == "augmented":
        params["n_aug"] = base_config["augmentation"]["n_aug"]
    if random_seed is not None:
        params["random_seed"] = random_seed
    return params
