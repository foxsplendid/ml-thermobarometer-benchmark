# -*- coding: utf-8 -*-
"""Tests for S3: EPMA perturbation non-negative clip."""

import numpy as np

from src.perturbation import (
    DEFAULT_OXIDE_REL_ERR,
    get_rel_err_vector,
    epma_perturb,
    perturbation_with_repeats,
)


LARGE_REL_ERR = np.array([0.5, 0.5])  # large enough that ~2.3% of draws go negative
NEAR_ZERO_X = np.array([[0.05, 0.3]] * 100)  # near-detection-limit wt%


def test_clip_enforces_nonnegative():
    rng = np.random.RandomState(42)
    X_pert = np.vstack([
        epma_perturb(NEAR_ZERO_X, LARGE_REL_ERR, rng) for _ in range(100)
    ])
    assert (X_pert >= 0).all()

    # Sanity: the unclipped path DOES produce negatives, so the clip is real.
    rng = np.random.RandomState(42)
    X_raw = np.vstack([
        epma_perturb(NEAR_ZERO_X, LARGE_REL_ERR, rng, clip_negative=False)
        for _ in range(100)
    ])
    assert (X_raw < 0).any()


def test_clip_is_noop_under_default_errors():
    """Bit-exact back-compat: at 3-8% rel. err. the clip never fires."""
    feature_names = list(DEFAULT_OXIDE_REL_ERR.keys())
    rel_err = get_rel_err_vector(feature_names)
    rng = np.random.RandomState(42)
    X = rng.uniform(0.1, 55.0, size=(200, len(feature_names)))

    out_clip = epma_perturb(X, rel_err, np.random.RandomState(7), clip_negative=True)
    out_raw = epma_perturb(X, rel_err, np.random.RandomState(7), clip_negative=False)
    assert np.array_equal(out_clip, out_raw)


def test_zero_column_stays_zero():
    """Exact zeros (below-detection convention) have scale 0 -> unperturbed."""
    X = np.zeros((50, 3))
    rel_err = np.array([0.03, 0.08, 0.5])
    out = epma_perturb(X, rel_err, np.random.RandomState(42))
    assert np.array_equal(out, X)


def test_perturbation_with_repeats_clips_both_branches():
    y = np.arange(len(NEAR_ZERO_X), dtype=float)

    X_aug, y_aug = perturbation_with_repeats(
        NEAR_ZERO_X, y, LARGE_REL_ERR, n_perturbations=20,
        random_seed=42, include_original=True,
    )
    assert (X_aug >= 0).all()
    assert X_aug.shape == (len(NEAR_ZERO_X) * 21, 2)
    assert len(y_aug) == len(NEAR_ZERO_X) * 21

    X_aug, y_aug = perturbation_with_repeats(
        NEAR_ZERO_X, y, LARGE_REL_ERR, n_perturbations=20,
        random_seed=42, include_original=False,
    )
    assert (X_aug >= 0).all()
    assert X_aug.shape == (len(NEAR_ZERO_X) * 20, 2)


def test_clip_point_mass_fraction():
    """Clipping (not resampling): P(clip) = Phi(-1/rel_err) = Phi(-2) ~ 2.3%."""
    rng = np.random.RandomState(42)
    X = np.full((10_000, 1), 1.0)
    out = epma_perturb(X, np.array([0.5]), rng)
    frac_zero = float((out == 0.0).mean())
    assert abs(frac_zero - 0.0228) < 0.005
