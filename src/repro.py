# -*- coding: utf-8 -*-
"""Reproducibility helpers: code fingerprint and git state.

Single source of truth for binding a result to a code version.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Dict, Optional


_FINGERPRINT_GLOBS = ("src/**/*.py", "*.py")
_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".git", "results", "results_v8", "results_test"}


def _iter_code_files(root: Path):
    for pattern in _FINGERPRINT_GLOBS:
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            if any(part in _EXCLUDE_DIRS for part in p.parts):
                continue
            yield p


def code_fingerprint(root: Path | str) -> Dict[str, str]:
    """Return {relative_path: sha256} over code-bearing files under root."""
    root = Path(root).resolve()
    out: Dict[str, str] = {}
    for p in sorted(_iter_code_files(root)):
        with p.open("rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        out[str(p.relative_to(root)).replace("\\", "/")] = h
    return out


def combined_sha(fingerprint: Dict[str, str], length: int = 16) -> str:
    """Collapse a fingerprint dict into a single short hash."""
    h = hashlib.sha256()
    for path in sorted(fingerprint):
        h.update(path.encode("utf-8"))
        h.update(b"\x00")
        h.update(fingerprint[path].encode("ascii"))
        h.update(b"\x00")
    return h.hexdigest()[:length]


def git_state(root: Path | str) -> Dict[str, Optional[object]]:
    """Return git HEAD + dirty flag, or unknown if not a git repo."""
    root = str(Path(root).resolve())
    info: Dict[str, Optional[object]] = {"commit": "unknown", "dirty": None}
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=root, timeout=5,
        )
        if r.returncode == 0:
            info["commit"] = r.stdout.strip()
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=root, timeout=5,
        )
        if r.returncode == 0:
            info["dirty"] = bool(r.stdout.strip())
    except Exception:
        pass
    return info
