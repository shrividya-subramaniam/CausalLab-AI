"""Configuration loading and repository paths.

Everything that can change a result (thresholds, sample size, seeds, model IDs)
is read from config.yaml so it is visible, versioned and fixed before a run.
"""
from __future__ import annotations

import hashlib

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_raw(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return a fresh copy of the config so callers can tweak it without side effects."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    return copy.deepcopy(_load_raw(str(path)))


def repo_path(rel: str | Path) -> Path:
    """Resolve a path from the config relative to the repository root."""
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def code_version() -> str:
    """Short fingerprint of the package source (every .py file under src/causallab, line endings
    normalised). Result files record it, so a reader can tell which code produced them:
    `python -m causallab version` prints it for the code in front of you."""
    h = hashlib.sha256()
    for f in sorted((REPO_ROOT / "src" / "causallab").rglob("*.py")):
        h.update(f.relative_to(REPO_ROOT).as_posix().encode())
        h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def stamp(result: dict) -> dict:
    """A result with the code fingerprint added, for writing to a results file."""
    return {**result, "code_version": code_version()}


def display_path(p: str | Path) -> str:
    """A path as written into result files: relative to the repository root, with forward
    slashes, so saved results carry no machine-specific folders."""
    p = Path(p)
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def results_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    p = repo_path(cfg["paths"]["results_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    p = repo_path(cfg["paths"]["processed_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_spec(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """The reference specification committed at the end of Phase 1."""
    cfg = cfg or load_config()
    with open(repo_path(cfg["paths"]["spec_locked"]), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dictionary(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Variable descriptions shown to propose_dag. Descriptions only, no data."""
    cfg = cfg or load_config()
    with open(repo_path(cfg["paths"]["dictionary"]), encoding="utf-8") as f:
        return yaml.safe_load(f)
