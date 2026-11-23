"""Shared fixtures. Tests use small synthetic data so `pytest` runs without the Kaggle file."""
from __future__ import annotations

import copy

import pytest

from causallab.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """Config with results written to a temporary folder and lighter settings for speed."""
    c = copy.deepcopy(load_config())
    c["paths"]["results_dir"] = str(tmp_path / "results")
    c["paths"]["processed_dir"] = str(tmp_path / "processed")
    c["models"]["lgbm"]["n_estimators"] = 200
    c["models"]["cv_folds"] = 3
    c["refuters"]["subset_n"] = 10   # with fewer, the 80% rule needs every half-sample inside
    c["bootstrap"]["benchmark_reps"] = 200
    c["trap_suite"]["rows"] = 6000
    return c
