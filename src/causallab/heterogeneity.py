"""Q2: does the top-3 effect differ by price tier, star rating or trip type?

A causal forest (EconML CausalForestDML) is fitted on a subsample of the observational arm.
Group effects are averages of the forest's individual effects within each group. Each is
then compared with the same group effect computed directly in the randomly ranked arm,
which the benchmark module provides (this module never reads that arm itself).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import load_config, stamp
from .estimators import fold_splits, group_folds, lgbm_params

MODIFIERS = ["price_tercile", "star_band", "trip_type"]


def add_modifiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    star = df["prop_starrating"].to_numpy(dtype=float, na_value=np.nan)  # plain NaN, not pd.NA
    df["star_band"] = np.select([np.isnan(star), star <= 2, star == 3, star == 4, star >= 5],
                                ["unrated", "1-2", "3", "4", "5"], "unrated")
    return df


def _group_queries(df: pd.DataFrame) -> dict[tuple[str, str], str]:
    q = {}
    for m in MODIFIERS:
        for level in sorted(df[m].astype(str).unique()):
            q[(m, level)] = f"{m}.astype('str') == '{level}'"
    return q


def run_q2(df: pd.DataFrame, covariates: list[str], cfg: dict | None = None, synthetic: bool = False,
           out_path: Path | None = None, rows: int | None = None) -> dict[str, Any]:
    from econml.dml import CausalForestDML

    from .benchmark import group_benchmarks

    cfg = cfg or load_config()
    qc = cfg["q2"]
    seed = cfg["seed"]
    df = add_modifiers(df)
    # subsample whole searches
    rows = rows or qc["forest_subsample_rows"]
    if len(df) > rows:
        rng = np.random.default_rng(seed)
        ids = df["srch_id"].unique()
        keep = rng.permutation(ids)[: int(len(ids) * rows / len(df))]
        df = df[df["srch_id"].isin(keep)].reset_index(drop=True)
    X = pd.get_dummies(df[MODIFIERS].astype(str), dtype=float)
    W = df[sorted(c for c in dict.fromkeys(covariates) if c not in ("prop_starrating",))]  # order-invariant
    folds = group_folds(df["srch_id"].to_numpy(), cfg["models"]["cv_folds"], seed)
    cf = CausalForestDML(model_y=lgb.LGBMRegressor(**lgbm_params(cfg, seed)),
                         model_t=lgb.LGBMClassifier(**lgbm_params(cfg, seed)),
                         discrete_treatment=True, cv=fold_splits(folds), n_estimators=qc["forest_n_estimators"],
                         min_samples_leaf=qc["forest_min_samples_leaf"], random_state=seed, allow_missing=True)
    cf.fit(df["click_bool"].to_numpy(float), df["top3"].to_numpy(int), X=X, W=W)

    queries = _group_queries(df)
    exp = group_benchmarks({f"{m}={lv}": q for (m, lv), q in queries.items()}, cfg, synthetic)
    groups = []
    for (m, lv), q in queries.items():
        mask = df.eval(q).to_numpy()
        if mask.sum() < 200:
            continue
        inf = cf.ate_inference(X=X[mask])
        lo, hi = (float(np.ravel(v)[0]) for v in inf.conf_int_mean())
        groups.append({"modifier": m, "level": lv, "n_rows": int(mask.sum()),
                       "estimate": float(np.ravel(inf.mean_point)[0]), "ci_low": lo, "ci_high": hi,
                       "experimental": exp.get(f"{m}={lv}")})
    overall = cf.ate_inference(X=X)
    res = {"n_rows": int(len(df)), "n_searches": int(df["srch_id"].nunique()),
           "overall": {"estimate": float(np.ravel(overall.mean_point)[0])}, "groups": groups,
           "note": "Forest intervals are not clustered by search; experimental intervals are (cluster bootstrap)."}
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stamp(res), indent=2), encoding="utf-8")
    return res
