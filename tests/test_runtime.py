# -*- coding: utf-8 -*-
"""Unit tests for src/runtime.py environment-variable knobs."""

import os
import pytest
from unittest.mock import patch

from src.runtime import get_n_jobs, get_fold_workers, get_fold_backend


class TestGetNJobs:
    def test_default_is_four(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_N_JOBS", None)
            assert get_n_jobs() == 4

    def test_env_integer(self):
        with patch.dict(os.environ, {"ML_N_JOBS": "8"}):
            assert get_n_jobs() == 8

    def test_env_negative(self):
        with patch.dict(os.environ, {"ML_N_JOBS": "-1"}):
            assert get_n_jobs() == -1

    def test_env_invalid_falls_back(self):
        with patch.dict(os.environ, {"ML_N_JOBS": "bad"}):
            assert get_n_jobs() == 4


class TestGetFoldWorkers:
    def test_default_auto_detects_large_machine(self):
        """32 cores / 4 n_jobs = 8 workers."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_FOLD_WORKERS", None)
            os.environ.pop("ML_N_JOBS", None)
            with patch("os.cpu_count", return_value=32):
                assert get_fold_workers() == 8

    def test_default_auto_detects_small_machine(self):
        """4 cores / 4 n_jobs = max(1, 1) = 1 (sequential)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_FOLD_WORKERS", None)
            os.environ.pop("ML_N_JOBS", None)
            with patch("os.cpu_count", return_value=4):
                assert get_fold_workers() == 1

    def test_default_auto_detects_medium_machine(self):
        """16 cores / 4 n_jobs = 4 workers."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_FOLD_WORKERS", None)
            os.environ.pop("ML_N_JOBS", None)
            with patch("os.cpu_count", return_value=16):
                assert get_fold_workers() == 4

    def test_env_integer(self):
        with patch.dict(os.environ, {"ML_FOLD_WORKERS": "4"}):
            assert get_fold_workers() == 4

    def test_env_one_forces_sequential(self):
        with patch.dict(os.environ, {"ML_FOLD_WORKERS": "1"}):
            assert get_fold_workers() == 1

    def test_env_zero_clamps_to_one(self):
        with patch.dict(os.environ, {"ML_FOLD_WORKERS": "0"}):
            assert get_fold_workers() == 1

    def test_env_invalid_falls_back_to_auto(self):
        """Invalid value falls back to auto-detect (not a fixed constant)."""
        with patch.dict(os.environ, {"ML_FOLD_WORKERS": "bad"}, clear=False):
            os.environ.pop("ML_N_JOBS", None)
            with patch("os.cpu_count", return_value=32):
                assert get_fold_workers() == 8

    def test_auto_respects_custom_n_jobs(self):
        """Auto-detect uses ML_N_JOBS when set: 32 cores / 8 n_jobs = 4 workers."""
        with patch.dict(os.environ, {"ML_N_JOBS": "8"}, clear=False):
            os.environ.pop("ML_FOLD_WORKERS", None)
            with patch("os.cpu_count", return_value=32):
                assert get_fold_workers() == 4


class TestGetFoldBackend:
    def test_default_is_loky(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ML_FOLD_BACKEND", None)
            assert get_fold_backend() == "loky"

    def test_valid_loky(self):
        with patch.dict(os.environ, {"ML_FOLD_BACKEND": "loky"}):
            assert get_fold_backend() == "loky"

    def test_valid_multiprocessing(self):
        with patch.dict(os.environ, {"ML_FOLD_BACKEND": "multiprocessing"}):
            assert get_fold_backend() == "multiprocessing"

    def test_invalid_falls_back_to_loky(self):
        with patch.dict(os.environ, {"ML_FOLD_BACKEND": "unknown"}):
            assert get_fold_backend() == "loky"

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ML_FOLD_BACKEND": "LOKY"}):
            assert get_fold_backend() == "loky"
