"""Phase 4 acceptance: each pass/fail check fires on data built to trip it and stays quiet on a
clean one. Placebo and random common cause are tested for staying quiet only."""
import numpy as np
import pandas as pd

from causallab import estimators as E
from causallab import refuters as R
from causallab.synthetic import trap_scenarios


def _scen(name, n=8000, seed=0):
    return next(s for s in trap_scenarios(n=n, seed=seed) if s.name == name)


def _run(sc, cfg, cov=None, trim=None):
    cov = cov or sc.reference_adjustment
    est = E.estimate_all(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg, trim=trim)
    kept = sc.df[est["keep"]].reset_index(drop=True)
    return est, kept, cov


def test_overlap_fires_and_stays_quiet(cfg):
    import pytest
    # A hard placement rule leaves (almost) nothing after trimming: estimation stops with an
    # overlap error, which the agent turns into a failed overlap check.
    with pytest.raises(E.OverlapError):
        _run(_scen("no_overlap"), cfg)
    est = E.estimate_all(_scen("no_overlap").df, "featured_slot", "clicked", ["listing_quality"], "session_id", cfg,
                         which=("naive",), trim=(0.0, 1.0))
    est["trimmed_share"] = 0.5
    assert R.overlap(est, cfg)["passed"] is False
    est, _, _ = _run(_scen("clean_confounding"), cfg)
    assert R.overlap(est, cfg)["passed"] is True


def test_balance_fires_and_stays_quiet(cfg):
    sc = _scen("failing_balance", n=3000)
    est, kept, cov = _run(sc, cfg)
    assert R.balance(kept, cov, kept[sc.treatment].to_numpy(), est["propensity"][est["keep"]], cfg)["passed"] is False
    sc = _scen("clean_confounding")
    est, kept, cov = _run(sc, cfg)
    assert R.balance(kept, cov, kept[sc.treatment].to_numpy(), est["propensity"][est["keep"]], cfg)["passed"] is True


def test_sensitivity_fires_and_stays_quiet(cfg):
    sc = _scen("clean_confounding")
    est, kept, cov = _run(sc, cfg)
    y, t = kept[sc.outcome].to_numpy(float), kept[sc.treatment].to_numpy()
    assert R.sensitivity(y, t, kept[cov], est["estimates"]["aipw"], cfg)["passed"] is True
    # one dominant confounder: a hidden confounder that strong would erase the effect
    rng = np.random.default_rng(0)
    n = 8000
    x = rng.normal(size=n)
    t2 = (rng.random(n) < 1 / (1 + np.exp(-2.0 * x))).astype(int)
    y2 = (rng.random(n) < 1 / (1 + np.exp(-(-1.5 + 1.5 * x + 0.15 * t2)))).astype(float)
    X = pd.DataFrame({"x": x, "noise": rng.normal(size=n)})
    prim = E.EstimateResult("aipw", 0.02, 0.01, n, n, {"mean_control": 0.3})
    assert R.sensitivity(y2, t2, X, prim, cfg)["passed"] is False


def test_subset_stability_fires_and_stays_quiet(cfg):
    sc = _scen("clean_confounding")
    est, kept, cov = _run(sc, cfg)
    main = est["estimates"]["aipw"]
    assert R.subset_stability(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg, main)["passed"] is True
    fake = E.EstimateResult("aipw", main.estimate + 0.2, main.se, main.n_rows, main.n_groups)
    assert R.subset_stability(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg, fake)["passed"] is False


def test_placebo_and_random_common_cause_stay_quiet(cfg):
    sc = _scen("clean_confounding")
    est, kept, cov = _run(sc, cfg)
    assert R.placebo(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg)["passed"] is True
    assert R.random_common_cause(sc.df, sc.treatment, sc.outcome, cov, sc.group, cfg, est["estimates"]["aipw"])["passed"] is True


def test_sensitivity_formula_matches_published_closed_form():
    """Values cross-checked against PySensemakr's own functions during development."""
    b = R.ovb_bound(0.3, 0.04, 2996, 0.17, 0.05)
    assert b["r2dz_x"] > 0 and np.isfinite(b["adjusted_estimate"])
    assert abs(R.robustness_value(7.5, 2996) - 0.1280) < 1e-3


def test_averaged_placebo_uses_single_run_interval(cfg):
    import copy
    sc = _scen("clean_confounding")
    c = copy.deepcopy(cfg); c["refuters"]["placebo_reps"] = 3
    r = R.placebo(sc.df, sc.treatment, sc.outcome, sc.reference_adjustment, sc.group, c)
    assert len(r["estimates"]) == 3 and r["passed"] is True
    half_width = (r["ci"][1] - r["ci"][0]) / 2
    assert abs(half_width - 1.959963984540054 * r["single_run_se"]) < 1e-9


def test_sensitivity_bounds_unchanged_by_clustering_but_interval_wider(cfg):
    """The bound and robustness value are algebraic, so clustering must not change them; only
    the interval around the adjusted estimate widens when rows of a session share a shock."""
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(2000), 5)
    shock = np.repeat(rng.normal(size=2000), 5)
    x = rng.normal(size=g.size)
    t = (rng.random(g.size) < 1 / (1 + np.exp(-(0.8 * x + shock)))).astype(int)
    y = (rng.random(g.size) < 1 / (1 + np.exp(-(-1 + 0.5 * x + 0.3 * t + shock)))).astype(float)
    X = pd.DataFrame({"x": x, "z": rng.normal(size=g.size)})
    prim = E.EstimateResult("aipw", 0.05, 0.01, g.size, 2000, {"mean_control": 0.3})
    classical = R.sensitivity(y, t, X, prim, cfg)
    clustered = R.sensitivity(y, t, X, prim, cfg, groups=g)
    assert clustered["robustness_value_q1"] == classical["robustness_value_q1"]
    assert clustered["bound_at_strongest"] == classical["bound_at_strongest"]
    assert clustered["ols_se_cluster"] > clustered["ols_se_classical"]
    lo, hi = clustered["adjusted_ci_cluster"]
    assert lo < clustered["bound_at_strongest"]["adjusted_estimate"] < hi


def test_held_out_covariate_leaves_out_its_derived_columns():
    from causallab.refuters import held_out_covariate
    rng = np.random.default_rng(0)
    n = 4000
    g = np.repeat(np.arange(n // 10), 10)
    loc2 = rng.normal(size=n)
    df = pd.DataFrame({"prop_location_score2": loc2, "loc2_missing": 0.0,
                       "loc2_rel": loc2 - pd.Series(loc2).groupby(g).transform("mean").to_numpy(),
                       "prop_starrating": rng.normal(size=n)})
    t = (rng.random(n) < 1 / (1 + np.exp(-loc2))).astype(int)
    r = held_out_covariate(df, t, list(df.columns), "prop_location_score2", g)
    assert r["also_left_out"] == ["loc2_missing", "loc2_rel"]
    assert r["estimate"] > 0.5  # treated rows have clearly higher loc2 once loc2_rel is not in the model


def test_e_value_is_undefined_for_a_non_positive_risk_ratio():
    assert abs(R.e_value(2.0) - (2 + np.sqrt(2))) < 1e-12
    assert R.e_value(0.5) == R.e_value(2.0)          # protective effects use 1/RR
    assert R.e_value(0.0) is None and R.e_value(-0.5) is None
    # an estimate below minus the control rate: the E-value is reported as undefined, no crash
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({"x": rng.normal(size=n)})
    t = rng.integers(0, 2, n)
    y = (rng.random(n) < 0.05).astype(float)
    bad = E.EstimateResult("aipw", -0.08, 0.01, n, n, {"mean_control": 0.05})
    from causallab.config import load_config
    r = R.sensitivity(y, t, X, bad, load_config())
    assert r["e_value"] is None and r["e_value_ci"] is None
