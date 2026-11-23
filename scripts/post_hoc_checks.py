"""Post-hoc numbers the README quotes that the pipeline does not save itself.

    python scripts/post_hoc_checks.py              # writes results/post_hoc_checks.json
    python scripts/post_hoc_checks.py --synthetic  # writes results/post_hoc_checks_synthetic.json

Run it after `python -m causallab estimate`, which saves the reference outcome models it reads.
It lives outside src/causallab on purpose: it only reads saved models and data, so the
`code_version` of the pipeline results is unchanged. The output records both that code_version
and a hash of this script.

1. The reference outcome models' predicted effect on the ranker's own hotels (the algorithm-
   ranked rows in positions 1-10). The carried-over estimate in population_checks_reference.json
   is the same models' prediction on the randomly ranked hotels, so the difference between the
   two is the part of the gap that comes from the hotel mix, measured like for like.
2. Grouped sensitivity benchmarks (Cinelli & Hazlett, 2020) on the same regression as the
   pipeline's sensitivity check: how far the regression estimate would move with a hidden
   confounder as strong as a whole group of observed columns, instead of one column at a time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from causallab import data, features
from causallab.config import load_config, load_spec, results_dir, stamp
from causallab.estimators import impute_for_linear, score_effect
from causallab.refuters import ovb_bound

GROUPS = {
    "location_score_2": ["prop_location_score2", "loc2_missing", "loc2_rel"],
    "hotel_price_page_query": [
        # hotel
        "prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1",
        "prop_location_score2", "prop_log_historical_price", "star_zero", "review_zero",
        "review_missing", "hist_price_zero", "loc2_missing",
        # price and promotion
        "log_price", "price_rank_pct", "price_rel_median", "promotion_flag",
        # page-relative
        "star_rel", "review_rel", "loc1_rel", "loc2_rel",
        # query
        "srch_query_affinity_score", "affinity_missing", "log_distance", "distance_missing",
    ],
}


def _rss(X: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares of an OLS fit, from the normal equations (memory-light)."""
    xtx, xty = X.T @ X, X.T @ y
    beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]
    r = y - X @ beta
    return float(r @ r)


def grouped_sensitivity(df: pd.DataFrame, covariates: list[str], cfg: dict) -> dict:
    y = df["click_bool"].to_numpy(float)
    t = df["top3"].to_numpy(float)
    Xl = impute_for_linear(df[covariates])
    Xl = Xl.loc[:, Xl.std() > 0]
    cols = list(Xl.columns)
    X = np.c_[np.ones(len(Xl)), Xl.to_numpy(float)]
    n = len(y)
    D = np.c_[t, X]                                   # outcome regression: treatment + covariates
    dof = n - D.shape[1]
    xtx, xty = D.T @ D, D.T @ y
    beta = np.linalg.solve(xtx, xty)
    resid = y - D @ beta
    sigma2 = resid @ resid / dof
    se = float(np.sqrt(sigma2 * np.linalg.inv(xtx)[0, 0]))
    est = float(beta[0])
    rss_y_full = float(resid @ resid)
    rss_t_full = _rss(X, t)
    kd, ky = cfg["refuters"]["sensitivity_kd"], cfg["refuters"]["sensitivity_ky"]
    out = {"ols_estimate": est, "ols_se_classical": se, "dof": dof, "kd": kd, "ky": ky,
           "check_single_covariate_n_results": None, "groups": {}}
    for name, group in GROUPS.items():
        keep = [i for i, c in enumerate(cols) if c not in group]
        used = [c for c in group if c in cols]
        Xr = X[:, [0] + [i + 1 for i in keep]]
        r2d = (_rss(Xr, t) - rss_t_full) / _rss(Xr, t)
        r2y = (_rss(np.c_[t, Xr], y) - rss_y_full) / _rss(np.c_[t, Xr], y)
        b = ovb_bound(est, se, dof, r2d, r2y, kd, ky)
        out["groups"][name] = {"columns": used, "partial_r2_treatment": float(r2d), "partial_r2_outcome": float(r2y), **b}
    # the pipeline's strongest single benchmark, recomputed here: must match refutations_reference.json
    if "n_results" in cols:  # constant (and so dropped) in the synthetic file
        j = cols.index("n_results")
        Xr = np.delete(X, j + 1, axis=1)
        r2d = (_rss(Xr, t) - rss_t_full) / _rss(Xr, t)
        r2y = (_rss(np.c_[t, Xr], y) - rss_y_full) / _rss(np.c_[t, Xr], y)
        out["check_single_covariate_n_results"] = ovb_bound(est, se, dof, r2d, r2y, kd, ky)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config()
    sfx = "_synthetic" if a.synthetic else ""
    model_path = results_dir(cfg) / "models" / f"reference_outcome{sfx}.pkl"
    pc_path = results_dir(cfg) / f"population_checks_reference{sfx}.json"
    est_path = results_dir(cfg) / f"estimates_reference{sfx}.json"
    for p in (model_path, pc_path, est_path):
        if not p.exists():
            print(f"{p} is missing: run `python -m causallab estimate{' --synthetic' if a.synthetic else ''}` first.")
            return 1
    covariates = list(dict.fromkeys(load_spec(cfg)["adjustment_set"]))
    obs = features.q1_rows(data.load_arm("observational", cfg, a.synthetic))

    bundle = joblib.load(model_path)
    on_ranker = float(score_effect(bundle, obs).mean())
    pc = json.loads(pc_path.read_text(encoding="utf-8"))
    carried = pc["transported_estimate"]["estimate"]
    est = json.loads(est_path.read_text(encoding="utf-8"))
    est = est.get("estimates", est)
    primary = est[cfg["models"]["primary_estimator"]]["estimate"]
    bench = pc["plain_benchmark"]["estimate"]
    mix = on_ranker - carried
    res = {
        "post_hoc": True,
        "note": "Computed by scripts/post_hoc_checks.py from saved models and data; not part of the pipeline.",
        "outcome_models_on_ranker_hotels": {
            "estimate": on_ranker,
            "carried_over_to_random_hotels": carried,
            "hotel_mix_share_of_gap": {"difference": mix, "primary_gap": primary - bench,
                                      "share": mix / (primary - bench)},
        },
        "grouped_sensitivity": grouped_sensitivity(obs, covariates, cfg),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12],
    }
    out = results_dir(cfg) / f"post_hoc_checks{sfx}.json"
    out.write_text(json.dumps(stamp(res), indent=2), encoding="utf-8")
    g = res["grouped_sensitivity"]["groups"]
    print(f"Outcome models on the ranker's hotels: {on_ranker * 100:.2f} pp (carried over to random hotels: "
          f"{carried * 100:.2f}); hotel mix {mix * 100:.2f} pp = {mix / (primary - bench):.0%} of the {(primary - bench) * 100:.2f} pp gap")
    for k, v in g.items():
        print(f"Grouped sensitivity, {k} ({len(v['columns'])} columns): regression estimate would fall to "
              f"{v['adjusted_estimate'] * 100:.2f} pp")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
