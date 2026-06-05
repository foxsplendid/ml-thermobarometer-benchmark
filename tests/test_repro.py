# -*- coding: utf-8 -*-
"""Tests for src.repro (code fingerprint + git state)."""

import os
import tempfile
from pathlib import Path

import pytest

from src.repro import code_fingerprint, combined_sha, git_state


def test_fingerprint_self():
    root = Path(__file__).resolve().parent.parent
    fp = code_fingerprint(root)
    assert len(fp) > 0
    # Common files we expect
    assert any(p.endswith("src/runtime.py") for p in fp)
    assert any(p.endswith("src/repro.py") for p in fp)
    # No cache files
    assert not any("__pycache__" in p for p in fp)


def test_fingerprint_deterministic():
    root = Path(__file__).resolve().parent.parent
    a = code_fingerprint(root)
    b = code_fingerprint(root)
    assert a == b
    assert combined_sha(a) == combined_sha(b)


def test_combined_sha_length():
    root = Path(__file__).resolve().parent.parent
    fp = code_fingerprint(root)
    assert len(combined_sha(fp)) == 16
    assert len(combined_sha(fp, length=64)) == 64


def test_fingerprint_changes_with_content(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    fp1 = code_fingerprint(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n", encoding="utf-8")
    fp2 = code_fingerprint(tmp_path)
    assert fp1 != fp2
    assert combined_sha(fp1) != combined_sha(fp2)


def test_git_state_returns_dict():
    root = Path(__file__).resolve().parent.parent
    s = git_state(root)
    assert "commit" in s
    assert "dirty" in s
