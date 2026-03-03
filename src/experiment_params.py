# -*- coding: utf-8 -*-
"""Shared helpers for building experiment parameter dictionaries."""

from typing import Any, Dict


def build_model_params(base_config: Dict[str, Any], model_module: str, random_seed: int) -> Dict[str, Any]:
    """Build model parameter dict from config defaults and module name."""
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

    params["random_seed"] = random_seed
    return params


def build_data_params(
    base_config: Dict[str, Any],
    data_module: str,
    feature_set: str,
    random_seed: int
) -> Dict[str, Any]:
    """Build data-module parameter dict from config defaults."""
    params: Dict[str, Any] = {
        "random_seed": random_seed,
        "feature_names": list(base_config["feature_sets"][feature_set]),
    }
    if data_module.lower() == "augmented":
        params["n_aug"] = base_config["augmentation"]["n_aug"]
    return params
