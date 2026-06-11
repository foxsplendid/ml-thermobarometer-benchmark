# -*- coding: utf-8 -*-
"""Shared helpers for building experiment parameter dictionaries."""

from typing import Any, Dict, List, Optional

# ============================================================
# Single source of truth for the E01-E12 experiment matrix.
# main.py:get_experiment_configs() iterates this list;
# build_exp_id() derives its lookup index from it.
# Adding or reordering an entry here automatically updates both.
# ============================================================
BASE_CONFIGS: List[Dict[str, str]] = [
    {'data': 'raw',       'model': 'ert',      'corr': 'none'},
    {'data': 'raw',       'model': 'catboost',  'corr': 'none'},
    {'data': 'raw',       'model': 'stacking',  'corr': 'none'},
    {'data': 'balanced',  'model': 'ert',      'corr': 'none'},
    {'data': 'balanced',  'model': 'catboost',  'corr': 'none'},
    {'data': 'balanced',  'model': 'stacking',  'corr': 'none'},
    {'data': 'augmented', 'model': 'ert',      'corr': 'none'},
    {'data': 'augmented', 'model': 'catboost',  'corr': 'none'},
    {'data': 'augmented', 'model': 'stacking',  'corr': 'none'},
    {'data': 'augmented', 'model': 'ert',      'corr': 'segmented'},
    {'data': 'augmented', 'model': 'catboost',  'corr': 'segmented'},
    {'data': 'augmented', 'model': 'stacking',  'corr': 'segmented'},
]

# (data, corr, model) → 1-based experiment number, derived from BASE_CONFIGS
_EXP_INDEX: Dict[tuple, int] = {
    (c['data'], c['corr'], c['model']): i
    for i, c in enumerate(BASE_CONFIGS, 1)
}


def build_exp_id(
    data_module: str,
    model_module: str,
    corr_module: str,
    feature_set: str,
) -> str:
    """Derive a canonical experiment ID from the four config axes.

    Numbering follows BASE_CONFIGS order (E01-E12 for the benchmark paper).
    Unrecognised combinations fall back to a descriptive non-numbered id.
    """
    dm = data_module.lower()
    mm = model_module.lower()
    cm = corr_module.lower()
    suffix = "noliq" if feature_set == "NoLiquid" else "liq"

    exp_num = _EXP_INDEX.get((dm, cm, mm))
    if exp_num is None:
        return f"{dm}_{mm}_{cm}_{suffix}"
    return f"E{exp_num:02d}_{mm}_{dm}_{cm}_{suffix}"


# ============================================================
# H4: smoke/dev-iteration parameter tier. Only reachable via an explicit
# build_model_params(..., tier="fast"); scientific entry points (main.py
# benchmark, tools/run_stability.py, tools/run_learning_curve.py,
# tools/run_error_propagation.py) never pass tier. Values are NOT a quality
# promise — they exist so main.py --test verifies pipeline plumbing fast.
# ============================================================
FAST_TIER_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "ert":      {"n_estimators": 50, "max_depth": 10},
    "rf":       {"n_estimators": 50, "max_depth": 10},
    # learning_rate raised to partially compensate the iteration cut so the
    # smoke-test metrics stay eyeball-readable.
    "catboost": {"iterations": 200, "depth": 4, "learning_rate": 0.1},
    "stacking": {"inner_cv": 3},
}


def build_model_params(
    base_config: Dict[str, Any],
    model_module: str,
    random_seed: Optional[int] = None,
    tier: str = "full",
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
    tier:
        ``'full'`` (default) returns the scientific config defaults
        unchanged. ``'fast'`` overlays :data:`FAST_TIER_OVERRIDES` for
        smoke/dev runs (H4); keys without an override keep their defaults.
        Unknown model modules have no fast overrides: tier='fast' leaves
        their (empty) params untouched, matching the full-tier fallback.
    """
    if tier not in ("full", "fast"):
        raise ValueError(f"Unknown tier: {tier!r}; expected 'full' or 'fast'")
    fast = tier == "fast"

    model_defaults = base_config["model_defaults"]
    name = model_module.lower()

    if name in {"ert", "extratrees"}:
        params: Dict[str, Any] = dict(model_defaults["ert"])
        if fast:
            params.update(FAST_TIER_OVERRIDES["ert"])
    elif name in {"rf", "randomforest"}:
        params = dict(model_defaults["rf"])
        if fast:
            params.update(FAST_TIER_OVERRIDES["rf"])
    elif name in {"catboost", "cb"}:
        params = dict(model_defaults["catboost"])
        if fast:
            params.update(FAST_TIER_OVERRIDES["catboost"])
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
        if fast:
            # Applied after stacking_base_defaults: the fast tier wins.
            for key in base_params:
                base_params[key].update(FAST_TIER_OVERRIDES.get(key, {}))
        params = {"base_model_params": base_params}
        if stacking_params:
            params.update({
                "inner_cv": stacking_params.get("inner_cv"),
                "use_meta_scaler": stacking_params.get("use_meta_scaler"),
            })
        if fast:
            params["inner_cv"] = FAST_TIER_OVERRIDES["stacking"]["inner_cv"]
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
