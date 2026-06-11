# -*- coding: utf-8 -*-
"""Feature perturbation utilities for augmentation and uncertainty analysis."""

import numpy as np
from typing import List, Optional, Tuple

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
    """get_rel_err_vector function."""
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
    clip_negative: bool = True
) -> np.ndarray:
    """epma_perturb function.

    clip_negative (S3, V8): oxide wt% is physically non-negative, so
    perturbed values are clipped at 0. With the default 3-8% relative errors
    the clip probability is Phi(-1/rel_err) < 4e-36 per draw, i.e. it never
    fires and results are bit-identical to V7 (the RNG stream is unchanged).
    It only matters for user-supplied rel errors >~0.2, where it introduces a
    point mass at 0 and a slight upward mean bias — the intended physical
    behaviour. clip_negative=False reproduces V7 exactly.
    """
    scale = rel_err_vec * np.abs(X)

    noise = rng.normal(0.0, scale, size=X.shape)

    X_pert = X + noise
    if clip_negative:
        X_pert = np.maximum(X_pert, 0.0)
    return X_pert


def perturbation_with_repeats(
    X: np.ndarray,
    y: np.ndarray,
    rel_err_vec: np.ndarray,
    n_perturbations: int = 15,
    rng: Optional[np.random.RandomState] = None,
    random_seed: int = 42,
    include_original: bool = True,
    clip_negative: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """perturbation_with_repeats function."""
    if rng is None:
        rng = np.random.RandomState(random_seed)

    n_original = len(X)

    if include_original:
        X_list = [X]
        y_list = [y]

        for _ in range(n_perturbations):
            X_perturbed = epma_perturb(X, rel_err_vec, rng, clip_negative=clip_negative)
            X_list.append(X_perturbed)
            y_list.append(y)

        X_aug = np.vstack(X_list)
        y_aug = np.concatenate(y_list)
    else:
        X_rep = np.repeat(X, repeats=n_perturbations, axis=0)
        y_aug = np.repeat(y, repeats=n_perturbations, axis=0)

        scale = rel_err_vec * np.abs(X_rep)
        X_aug = rng.normal(X_rep, scale)
        if clip_negative:
            X_aug = np.maximum(X_aug, 0.0)

    return X_aug, y_aug

