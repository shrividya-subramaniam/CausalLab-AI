"""The spec's Q1 sensitivity check (Section 3.3, treatment row): use every non-reserved position
from 4 down as the comparison group instead of positions 4-10.

Same reference specification, estimators, trimming rule and gap test as the primary comparison.
The observational estimate reads only the algorithm-ranked arm; the matching benchmark is
computed by the benchmark module, the only code that reads the randomly ranked arm for top3.
A wider window changes both sides (lower hotels are worse, and clicks keep falling with
position), so what is compared is the verdict, not the size of the effect. Hotels far down the
page rarely have any chance of a top-3 slot, so more rows are trimmed; the same overlap rule as
the main run (trimmed share at most max_trimmed_share) says whether the comparison is usable.
"""
from __future__ import annotations

import json
from typing import Any

from . import benchmark, data, estimators, features
from .config import load_config, load_spec, results_dir, stamp


def run_all_positions(cfg: dict | None = None, synthetic: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    spec = load_spec(cfg)
    sfx = "_synthetic" if synthetic else ""
    df = features.all_positions_rows(data.load_arm("observational", cfg, synthetic))
    prop_path = results_dir(cfg) / "models" / f"reference_all_positions_propensity{sfx}.pkl"
    est = estimators.estimate_all(df, spec["treatment"], spec["outcome"], spec["adjustment_set"], spec["group"], cfg,
                                  propensity_path=prop_path)
    estimates = {k: v.to_dict() for k, v in est["estimates"].items()}
    bench = benchmark.run_trimmed(prop_path, results_dir(cfg) / f"benchmark_all_positions{sfx}.json", cfg,
                                  synthetic=synthetic, window="all_positions")
    limit = cfg["overlap"]["max_trimmed_share"]
    overlap = {"trimmed_share": est["trimmed_share"], "rule": f"share of rows outside {est['trim']} <= {limit}",
               "passed": bool(est["trimmed_share"] <= limit)}
    tol = cfg["gap"]["tolerance_share_of_benchmark"]
    gaps = {k: benchmark.gap_test(v["estimate"], v["se"], bench, tol) for k, v in estimates.items()}
    res = {"check": "all_positions_window",
           "note": "Spec sensitivity check: control group = every non-reserved position from 4 down.",
           "n_rows": int(len(df)), "n_searches": int(df[spec["group"]].nunique()),
           "trimmed_share": est["trimmed_share"], "overlap": overlap, "estimates": estimates, "benchmark": bench, "gap": gaps,
           "primary_estimator": cfg["models"]["primary_estimator"], "synthetic": synthetic}
    (results_dir(cfg) / f"sensitivity_all_positions{sfx}.json").write_text(json.dumps(stamp(res), indent=2), encoding="utf-8")
    return res
