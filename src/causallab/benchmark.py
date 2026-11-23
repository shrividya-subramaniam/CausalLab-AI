"""The experimental benchmark (firewalled).

This module is the only code allowed to read the randomly ranked arm for Q1. Its code is
fixed before any observational estimate exists. It saves two numbers:
  * the untrimmed benchmark, computed once in Phase 2 (results/benchmark.json);
  * the trimmed benchmark, recomputed for each specification the agent runs: the saved
    observational propensity model scores the experimental rows, the same trimming rule
    drops rows, and the benchmark is recomputed on what is left.

Inference uses a cluster bootstrap that resamples whole searches. To make 1,000 draws
cheap, each search is first reduced to four numbers (treated rows, treated clicks, control
rows, control clicks), and the draws are vectorised over those.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import data, features
from .config import load_config, results_dir, stamp
from .estimators import score_effect, score_propensity
from .refuters import smd

SEARCH_LEVEL = ["srch_length_of_stay", "srch_booking_window", "srch_adults_count", "srch_children_count",
                "srch_room_count", "srch_saturday_night_bool", "n_results", "visitor_same_country",
                "site_5", "site_14", "site_15", "site_24", "site_32", "search_month", "search_dow"]
BALANCE_COLS = ["prop_starrating", "prop_review_score", "prop_brand_bool", "prop_location_score1",
                "prop_location_score2", "prop_log_historical_price", "log_price", "price_rank_pct",
                "promotion_flag", "srch_query_affinity_score", "log_distance"]


def search_stats(df: pd.DataFrame, outcome: str = "click_bool", treatment: str = "top3",
                 weight: pd.Series | None = None) -> pd.DataFrame:
    """Per search: treated rows, treated outcomes, control rows, control outcomes (optionally weighted)."""
    w = pd.Series(1.0, index=df.index) if weight is None else weight
    t = df[treatment] == 1
    return pd.DataFrame({
        "n1": (w * t).groupby(df["srch_id"]).sum(),
        "y1": (w * t * df[outcome]).groupby(df["srch_id"]).sum(),
        "n0": (w * ~t).groupby(df["srch_id"]).sum(),
        "y0": (w * ~t * df[outcome]).groupby(df["srch_id"]).sum(),
    })


def cluster_bootstrap(stats: pd.DataFrame, reps: int, seed: int, chunk: int = 200) -> dict[str, float]:
    """Difference in means with a bootstrap over searches, vectorised.

    Each draw picks searches with replacement; multinomial counts say how often each
    search was picked, and the pooled means follow from four dot products."""
    s = stats[["n1", "y1", "n0", "y0"]].to_numpy()
    est = s[:, 1].sum() / s[:, 0].sum() - s[:, 3].sum() / s[:, 2].sum()
    rng = np.random.default_rng(seed)
    n = len(s)
    draws = []
    for start in range(0, reps, chunk):
        b = min(chunk, reps - start)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=b)
        agg = counts @ s
        draws.append(agg[:, 1] / agg[:, 0] - agg[:, 3] / agg[:, 2])
    d = np.concatenate(draws)
    return {"estimate": float(est), "se": float(d.std(ddof=1)),
            "ci_low": float(np.percentile(d, 2.5)), "ci_high": float(np.percentile(d, 97.5)),
            "reps": reps, "n_searches": int(n)}


WINDOWS = {"q1": features.q1_rows, "all_positions": features.all_positions_rows}


def _experimental_q1(cfg: dict, synthetic: bool, window: str = "q1") -> pd.DataFrame:
    return WINDOWS[window](data.load_arm("experimental", cfg, synthetic))


def reweight_to_observational(exp: pd.DataFrame, obs: pd.DataFrame, seed: int) -> pd.Series:
    """Weights that make experimental searches look like observational ones on search-level
    covariates (odds of being an observational search). Uses covariates only, no outcomes."""
    se = exp.groupby("srch_id")[SEARCH_LEVEL].first()
    so = obs.groupby("srch_id")[SEARCH_LEVEL].first()
    X = pd.concat([se, so]).fillna(0.0)
    z = np.r_[np.zeros(len(se)), np.ones(len(so))]
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit((X - X.mean()) / X.std().replace(0, 1), z)
    p = clf.predict_proba((se.fillna(0.0) - X.mean()) / X.std().replace(0, 1))[:, 1]
    p = np.clip(p, 0.01, 0.99)  # as in hotel_weights: no infinite odds under near-separation
    w = pd.Series(p / (1 - p), index=se.index)
    w = w / w.mean()
    return exp["srch_id"].map(w)


def run_untrimmed(cfg: dict | None = None, synthetic: bool = False, save: bool = True) -> dict[str, Any]:
    """Phase 2: randomisation check, benchmark, cross-arm comparison, reweighted benchmark."""
    cfg = cfg or load_config()
    reps, seed = cfg["bootstrap"]["benchmark_reps"], cfg["seed"]
    exp = _experimental_q1(cfg, synthetic)
    obs = features.q1_rows(data.load_arm("observational", cfg, synthetic))

    balance = smd(exp, BALANCE_COLS, exp["top3"].to_numpy())
    main = cluster_bootstrap(search_stats(exp), reps, seed)
    cross_arm = {c: {"experimental": float(exp.groupby("srch_id")[c].first().mean()),
                     "observational": float(obs.groupby("srch_id")[c].first().mean())} for c in SEARCH_LEVEL}
    w = reweight_to_observational(exp, obs, seed)
    reweighted = cluster_bootstrap(search_stats(exp, weight=w), reps, seed)

    result = {
        "benchmark": main,
        "benchmark_reweighted_to_observational": reweighted,
        "randomisation_balance_smd": balance,
        "randomisation_max_abs_smd": float(max(abs(v) for v in balance.values())),
        "cross_arm_search_covariates": cross_arm,
        "n_rows": int(len(exp)),
        "synthetic": synthetic,
    }
    out = results_dir(cfg) / ("benchmark_synthetic.json" if synthetic else "benchmark.json")
    if save:
        out.write_text(json.dumps(stamp(result), indent=2), encoding="utf-8")
        click_by_position_figure(cfg, synthetic)
    return result


def run_trimmed(propensity_path: Path, out_path: Path, cfg: dict | None = None,
                synthetic: bool = False, window: str = "q1") -> dict[str, Any]:
    """Score the experimental rows with the observational propensity model, apply the same
    trimming rule, and recompute the benchmark on the rows that survive. `window` picks the
    comparison group: "q1" (positions 4-10) or "all_positions" (the spec's sensitivity check)."""
    cfg = cfg or load_config()
    bundle = joblib.load(propensity_path)
    exp = _experimental_q1(cfg, synthetic, window)
    e = score_propensity(bundle, exp)
    lo, hi = bundle["trim"]
    kept = exp[(e >= lo) & (e <= hi)]
    res = cluster_bootstrap(search_stats(kept), cfg["bootstrap"]["benchmark_reps"], cfg["seed"])
    res.update(trim=[lo, hi], trimmed_share=float(1 - len(kept) / len(exp)), window=window)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stamp(res), indent=2), encoding="utf-8")
    return res


def gap_test(estimate: float, se: float, bench: dict[str, float], tolerance_share: float) -> dict[str, Any]:
    """Gap = observational estimate minus benchmark, with its own 95% interval.

    The arms are independent, so the variances add. 'match' needs the whole interval inside
    +/- tolerance x |benchmark|; an interval that contains zero but is wider is 'inconclusive'."""
    gap = estimate - bench["estimate"]
    se_gap = float(np.sqrt(se ** 2 + bench["se"] ** 2))
    lo, hi = gap - 1.96 * se_gap, gap + 1.96 * se_gap
    tol = tolerance_share * abs(bench["estimate"])
    if -tol <= lo and hi <= tol:
        verdict = "match"
    elif lo <= 0 <= hi:
        verdict = "inconclusive"
    else:
        verdict = "mismatch"
    return {"gap": float(gap), "se": se_gap, "ci_low": float(lo), "ci_high": float(hi),
            "tolerance": float(tol), "verdict": verdict}


def click_by_position_figure(cfg: dict, synthetic: bool = False) -> Path:
    """Click rate by position in both arms, reserved positions marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for arm, style in (("observational", "-o"), ("experimental", "-s")):
        d = data.load_arm(arm, cfg, synthetic)
        r = d.groupby("position")["click_bool"].mean()
        ax.plot(r.index, r.values * 100, style, ms=3, label=f"{arm} arm")
    for p in cfg["data"]["reserved_positions"]:
        ax.axvline(p, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("Position on the page")
    ax.set_ylabel("Click rate (%)")
    ax.set_title("Click rate by position (dotted lines: reserved positions)")
    ax.legend()
    out = results_dir(cfg) / "figures" / ("click_by_position_synthetic.png" if synthetic else "click_by_position.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def group_benchmarks(groups: dict[str, str], cfg: dict | None = None, synthetic: bool = False) -> dict[str, dict]:
    """Q2 validation: top-3 effect within subgroups, computed directly in the randomly ranked
    arm (difference in means, cluster bootstrap). `groups` maps label -> pandas query."""
    from .heterogeneity import add_modifiers

    cfg = cfg or load_config()
    exp = add_modifiers(_experimental_q1(cfg, synthetic))
    out = {}
    for label, query in groups.items():
        sub = exp.query(query)
        if sub["top3"].nunique() < 2 or len(sub) < 200:
            out[label] = None
            continue
        out[label] = cluster_bootstrap(search_stats(sub), cfg["bootstrap"]["benchmark_reps"], cfg["seed"])
    return out


# ---------------------------------------------------------------- post-hoc: like-for-like comparisons

def _with_products(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rebuild interaction columns (a_x_b) if a specification used them."""
    df = df.copy()
    for c in cols:
        if c not in df.columns and "_x_" in c:
            a, b = c.split("_x_", 1)
            df[c] = df[a] * df[b]
    return df


def hotel_weights(obs: pd.DataFrame, exp: pd.DataFrame, cov: list[str], cfg: dict) -> tuple[pd.Series, dict]:
    """Odds that a row with these covariates is an algorithm-ranked Q1 row rather than a randomly
    ranked one (covariates only, cross-fitted by search). Weighting the randomly ranked rows by
    these odds makes them resemble the hotels the ranker puts in the top 10."""
    import lightgbm as lgb
    from .estimators import fold_splits, group_folds, lgbm_params

    X = pd.concat([obs[cov], exp[cov]], ignore_index=True)
    z = np.r_[np.ones(len(obs)), np.zeros(len(exp))]
    groups = np.r_[obs["srch_id"].to_numpy(), exp["srch_id"].to_numpy()]
    p = np.zeros(len(z))
    for tr, te in fold_splits(group_folds(groups, cfg["models"]["cv_folds"], cfg["seed"])):
        m = lgb.LGBMClassifier(**lgbm_params(cfg, cfg["seed"]))
        m.fit(X.iloc[tr], z[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    pe = np.clip(p[len(obs):], 0.01, 0.99)
    w = pd.Series(pe / (1 - pe), index=exp.index)
    w = w / w.mean()
    info = {"max_weight": float(w.max()), "effective_sample_share": float(w.sum() ** 2 / (w ** 2).sum() / len(w))}
    return w, info


def population_matched(outcome_path: Path, out_path: Path, primary: dict, cfg: dict | None = None,
                       synthetic: bool = False) -> dict[str, Any]:
    """POST-HOC (added 20 Nov 2026, after the synthetic check showed the two arms average over
    different hotels): the ranker fills the top 10 with different hotels than random ranking, and
    the effect differs by hotel, so the pre-registered comparison mixes leftover confounding with
    a population difference. Two like-for-like comparisons, both reading covariates only from the
    observational arm:

    1. Hotel-reweighted benchmark: randomly ranked rows weighted to resemble the ranker's top-10
       hotels, compared with the observational estimate. Under-corrects when the ranker also uses
       information no covariate captures (on synthetic data it closed only about 57% of the distance
       to the true effect on the ranker's hotels, so its gap runs too large).
    2. Transported estimate: the observational outcome models' predicted effect averaged over the
       randomly ranked hotels, compared with the plain benchmark on those same hotels. Its interval
       covers sampling of the randomly ranked searches only, not the models' own uncertainty."""
    cfg = cfg or load_config()
    reps, seed = cfg["bootstrap"]["benchmark_reps"], cfg["seed"]
    tol = cfg["gap"]["tolerance_share_of_benchmark"]
    bundle = joblib.load(outcome_path)
    cov = bundle["covariates"]
    exp = _with_products(_experimental_q1(cfg, synthetic), cov)
    obs = _with_products(features.q1_rows(data.load_arm("observational", cfg, synthetic)), cov)

    plain = cluster_bootstrap(search_stats(exp), reps, seed)
    w, winfo = hotel_weights(obs, exp, cov, cfg)
    reweighted = cluster_bootstrap(search_stats(exp, weight=w), reps, seed)
    reweighted.update(winfo)
    gap_reweighted = gap_test(primary["estimate"], primary["se"], reweighted, tol)

    tau = score_effect(bundle, exp)
    st = search_stats(exp)
    tsum = pd.Series(tau, index=exp.index).groupby(exp["srch_id"]).agg(["sum", "count"])
    s = np.c_[tsum["sum"].to_numpy(), tsum["count"].to_numpy(), st[["n1", "y1", "n0", "y0"]].to_numpy()]
    rng = np.random.default_rng(seed)
    n = len(s)
    trans_draws, gap_draws = [], []
    for start in range(0, reps, 200):
        b = min(200, reps - start)
        agg = rng.multinomial(n, np.full(n, 1 / n), size=b) @ s
        tr_ = agg[:, 0] / agg[:, 1]
        trans_draws.append(tr_)
        gap_draws.append(tr_ - (agg[:, 3] / agg[:, 2] - agg[:, 5] / agg[:, 4]))
    trans_draws, gap_draws = np.concatenate(trans_draws), np.concatenate(gap_draws)
    transported = {"estimate": float(tau.mean()), "se": float(trans_draws.std(ddof=1)),
                   "ci_low": float(np.percentile(trans_draws, 2.5)), "ci_high": float(np.percentile(trans_draws, 97.5))}
    gap_est = transported["estimate"] - plain["estimate"]
    g_lo, g_hi = float(np.percentile(gap_draws, 2.5)), float(np.percentile(gap_draws, 97.5))
    t_abs = tol * abs(plain["estimate"])
    verdict = "match" if (-t_abs <= g_lo and g_hi <= t_abs) else ("inconclusive" if g_lo <= 0 <= g_hi else "mismatch")
    gap_transported = {"gap": float(gap_est), "ci_low": g_lo, "ci_high": g_hi, "tolerance": float(t_abs), "verdict": verdict}

    res = {"post_hoc": True,
           "note": "Added 20 Nov 2026 after the synthetic check; not part of the pre-registered comparison.",
           "plain_benchmark": plain, "hotel_reweighted_benchmark": reweighted, "gap_vs_hotel_reweighted": gap_reweighted,
           "transported_estimate": transported, "gap_transported_vs_plain": gap_transported}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stamp(res), indent=2), encoding="utf-8")
    return res
