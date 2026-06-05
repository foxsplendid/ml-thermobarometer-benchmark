# -*- coding: utf-8 -*-
"""Tests for S1 (nested correction) and S2 (sparse-bin merge) in StratifiedCVProtocol."""

import numpy as np
import pytest

from src.protocol import StratifiedCVProtocol, Pipeline
from src.data_modules import RawDataModule
from src.model_modules import ExtraTreesModel
from src.correction_modules import NoCorrection, SegmentedLinearCorrector


def _make_pipeline_factory():
    def factory(seed=42):
        return Pipeline(
            data_module=RawDataModule(random_seed=seed),
            model_module=ExtraTreesModel(
                n_estimators=30, max_depth=5, n_jobs=2, random_seed=seed
            ),
            corr_module=NoCorrection(),
        )
    return factory


def _make_biased_dataset(n=400, seed=0):
    """Synthetic regression with deliberate non-linear bias so a segmented
    corrector has something to fit, and so in-sample vs nested evaluation
    diverge measurably."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 10, size=(n, 3))
    y = (X[:, 0] ** 1.5) + 0.3 * X[:, 1] + rng.normal(0, 0.5, size=n)
    bins = np.digitize(y, np.quantile(y, np.linspace(0.1, 0.9, 9)))
    return X, y, bins


class TestSparseBinMerge:
    def test_merge_eliminates_singleton_bins(self):
        rng = np.random.RandomState(0)
        n = 200
        X = rng.normal(size=(n, 4))
        y = rng.normal(size=n)
        # 80% in 5 dense bins, 20% in 40 singleton bins
        bins = np.concatenate([
            rng.randint(0, 5, size=int(0.8 * n)),
            np.arange(100, 100 + int(0.2 * n)),
        ])
        rng.shuffle(bins)

        proto = StratifiedCVProtocol(
            n_splits=5, random_seed=0,
            merge_sparse_bins=True, nested_correction=False,
        )
        # Should not raise — merge keeps StratifiedKFold happy
        res = proto.run(X, y, _make_pipeline_factory(),
                        stratify_labels=bins, verbose=False,
                        corr_module=NoCorrection())
        assert res['summary']['merge_sparse_bins'] is True

    def test_off_switch_preserves_v7_behaviour(self):
        rng = np.random.RandomState(1)
        n = 200
        X = rng.normal(size=(n, 4))
        y = rng.normal(size=n)
        bins = rng.randint(0, 6, size=n)  # well-populated, no merge needed

        proto_off = StratifiedCVProtocol(
            n_splits=5, random_seed=0,
            merge_sparse_bins=False, nested_correction=False,
        )
        proto_on = StratifiedCVProtocol(
            n_splits=5, random_seed=0,
            merge_sparse_bins=True, nested_correction=False,
        )
        r_off = proto_off.run(X, y, _make_pipeline_factory(),
                              stratify_labels=bins, verbose=False)
        r_on = proto_on.run(X, y, _make_pipeline_factory(),
                            stratify_labels=bins, verbose=False)
        # Bins are dense -> merge is no-op -> identical numerics
        assert r_off['summary']['rmse_mean'] == pytest.approx(
            r_on['summary']['rmse_mean'], rel=1e-12
        )


class TestNestedCorrection:
    def test_eval_mode_string(self):
        X, y, bins = _make_biased_dataset()
        proto = StratifiedCVProtocol(
            n_splits=4, random_seed=0,
            nested_correction=True, inner_correction_cv=3,
            merge_sparse_bins=False,
        )
        res = proto.run(X, y, _make_pipeline_factory(),
                        stratify_labels=bins,
                        corr_module=SegmentedLinearCorrector(), verbose=False)
        assert res['summary']['correction_eval_mode'] == "nested"
        assert res['summary']['correction_inner_cv'] == 3

    def test_no_correction_path_unchanged(self):
        X, y, bins = _make_biased_dataset()
        proto = StratifiedCVProtocol(
            n_splits=4, random_seed=0,
            nested_correction=True, inner_correction_cv=3,
            merge_sparse_bins=False,
        )
        res = proto.run(X, y, _make_pipeline_factory(),
                        stratify_labels=bins,
                        corr_module=NoCorrection(), verbose=False)
        # NoCorrection skips the nested branch (zero extra cost)
        assert res['summary']['correction_eval_mode'] == "none"
        assert res['summary']['correction_inner_cv'] == 0

    def test_nested_gives_more_pessimistic_rmse_than_in_sample(self):
        """Nested CV evaluates the corrector on held-out predictions and so
        should report RMSE >= the in-sample (V7) evaluation on the same data,
        in expectation. We allow a tiny tolerance since variance can flip
        the sign on small samples."""
        X, y, bins = _make_biased_dataset(n=600, seed=42)

        proto_nested = StratifiedCVProtocol(
            n_splits=5, random_seed=0,
            nested_correction=True, inner_correction_cv=4,
            merge_sparse_bins=False,
        )
        proto_v7 = StratifiedCVProtocol(
            n_splits=5, random_seed=0,
            nested_correction=False,
            merge_sparse_bins=False,
        )
        r_nested = proto_nested.run(
            X, y, _make_pipeline_factory(),
            stratify_labels=bins,
            corr_module=SegmentedLinearCorrector(), verbose=False,
        )
        r_v7 = proto_v7.run(
            X, y, _make_pipeline_factory(),
            stratify_labels=bins,
            corr_module=SegmentedLinearCorrector(), verbose=False,
        )
        # In-sample should look at least as good as nested (it has seen the
        # very points it's evaluated on). Allow tiny slack for variance.
        assert r_v7['summary']['rmse_mean'] <= r_nested['summary']['rmse_mean'] + 0.05

    def test_back_compat_off_path(self):
        X, y, bins = _make_biased_dataset(n=300, seed=7)
        proto = StratifiedCVProtocol(
            n_splits=4, random_seed=0,
            nested_correction=False,
            merge_sparse_bins=False,
        )
        res = proto.run(X, y, _make_pipeline_factory(),
                        stratify_labels=bins,
                        corr_module=SegmentedLinearCorrector(), verbose=False)
        assert res['summary']['correction_eval_mode'] == "in_sample"

    def test_aggregate_corr_model_still_returned(self):
        """The aggregate corrector (used for held-out test set) is always
        fit on full OOF, regardless of nested flag."""
        X, y, bins = _make_biased_dataset()
        proto = StratifiedCVProtocol(
            n_splits=4, random_seed=0,
            nested_correction=True, inner_correction_cv=3,
            merge_sparse_bins=False,
        )
        res = proto.run(X, y, _make_pipeline_factory(),
                        stratify_labels=bins,
                        corr_module=SegmentedLinearCorrector(), verbose=False)
        assert res['corr_model'] is not None
        assert 'boundaries' in res['corr_model']
