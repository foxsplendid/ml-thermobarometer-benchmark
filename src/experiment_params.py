# -*- coding: utf-8 -*-
"""Shared helpers for building experiment parameter dictionaries."""

from typing import Any, Dict, Optional


def build_exp_id(
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
) -> str:
    """Derive a canonical experiment ID from the four config axes.

    Follows the E01-E12 numbering scheme used in the benchmark paper:
      E01-E03: raw  + ert/catboost/stacking + none
      E04-E06: balanced + ert/catboost/stacking + none
      E07-E09: augmented + ert/catboost/stacking + none
      E10-E12: augmented + ert/catboost/stacking + segmented
    """
    _model_offset = {
        "ert": 0, "extratrees": 0,
        "rf": 0, "randomforest": 0,
        "catboost": 1, "cb": 1,
        "stacking": 2,
    }
    _data_base = {
        ("raw",       "none"):      1,
        ("balanced",  "none"):      4,
        ("augmented", "none"):      7,
        ("augmented", "segmented"): 10,
    }

    dm = data_module.lower()
    mm = model_module.lower()
    cm = corr_module.lower()

    base = _data_base.get((dm, cm))
    if base is None:
        # Unrecognised combination — fall back to a descriptive id.
        suffix = "noliq" if feature_set == "NoLiquid" else "liq"
        return f"{dm}_{mm}_{cm}_{suffix}"

    exp_num = base + _model_offset.get(mm, 0)
    suffix = "noliq" if feature_set == "NoLiquid" else "liq"
    return f"E{exp_num:02d}_{mm}_{dm}_{cm}_{suffix}"


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
