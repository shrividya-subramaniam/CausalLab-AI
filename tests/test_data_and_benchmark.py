"""Data layer and benchmark: sampling by search, feature order, bootstrap, gap test."""
import numpy as np
import pandas as pd

from causallab import benchmark, data, features
from causallab.synthetic import expedia_like


def test_sampling_keeps_whole_searches_and_is_deterministic(cfg, tmp_path):
    import duckdb
    raw, _ = expedia_like(n_searches=800, seed=1)
    src = tmp_path / "raw.parquet"
    raw.to_parquet(src, index=False)
    keep = data.search_hash_keep("srch_id", 0.3, cfg["seed"])
    a = duckdb.connect().execute(f"SELECT srch_id, count(*) n FROM '{src.as_posix()}' WHERE {keep} GROUP BY 1").df()
    b = duckdb.connect().execute(f"SELECT srch_id, count(*) n FROM '{src.as_posix()}' WHERE {keep} GROUP BY 1").df()
    full = raw.groupby("srch_id").size()
    assert set(a.srch_id) == set(b.srch_id)
    assert (a.set_index("srch_id").n == full.loc[a.srch_id]).all()
    assert 0.2 < len(a) / raw.srch_id.nunique() < 0.4


def test_within_search_features_use_the_whole_page(cfg, tmp_path):
    raw, _ = expedia_like(n_searches=300, seed=2)
    src = tmp_path / "s.parquet"
    raw.to_parquet(src, index=False)
    out = features.build_features(src, tmp_path / "f.parquet", cfg)
    f = pd.read_parquet(out)
    assert (f.groupby("srch_id").n_results.first() == raw.groupby("srch_id").size()).all()
    q1 = features.q1_rows(f)
    assert q1.position.max() <= 10 and not q1.position.isin(cfg["data"]["reserved_positions"]).any()
    # price rank still spans the full page after the position filter
    assert q1.price_rank_pct.max() > 0.9


def test_vectorised_bootstrap_matches_point_estimate():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"srch_id": np.repeat(np.arange(400), 9), "top3": np.tile([1, 1, 1, 0, 0, 0, 0, 0, 0], 400)})
    d["click_bool"] = (rng.random(len(d)) < np.where(d.top3 == 1, 0.12, 0.06)).astype(int)
    r = benchmark.cluster_bootstrap(benchmark.search_stats(d), 300, 1)
    dim = d[d.top3 == 1].click_bool.mean() - d[d.top3 == 0].click_bool.mean()
    assert abs(r["estimate"] - dim) < 1e-12
    assert r["ci_low"] < dim < r["ci_high"] and r["se"] > 0


def test_gap_verdicts():
    bench = {"estimate": 0.05, "se": 0.002}
    assert benchmark.gap_test(0.0505, 0.001, bench, 0.2)["verdict"] == "match"
    assert benchmark.gap_test(0.065, 0.001, bench, 0.2)["verdict"] == "mismatch"
    assert benchmark.gap_test(0.05, 0.02, bench, 0.2)["verdict"] == "inconclusive"


def test_synthetic_truth_differs_by_population():
    """The ranker fills the top ten with better hotels, which gain more from a top slot."""
    raw, truth = expedia_like(n_searches=800, seed=3)
    t = raw.attrs["truths"]
    assert t["observational_q1_rows"] > t["experimental_q1_rows"] + 0.01


def test_population_matched_moves_toward_the_right_answers(cfg):
    from causallab import estimators as E
    raw, _ = expedia_like(n_searches=3000, seed=4)
    truths = raw.attrs["truths"]
    raw.to_parquet(data.sample_path(cfg, True), index=False)
    features.build_features(data.sample_path(cfg, True), data.features_path(cfg, True), cfg)
    obs = features.q1_rows(data.load_arm("observational", cfg, True))
    cov = ["prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2", "log_price",
           "promotion_flag", "star_rel", "review_rel", "loc1_rel", "loc2_rel", "price_rank_pct"]
    from causallab.config import processed_dir
    op = processed_dir(cfg) / "outcome.pkl"
    est = E.estimate_all(obs, "top3", "click_bool", cov, "srch_id", cfg, which=("aipw",), outcome_path=op)
    pc = benchmark.population_matched(op, processed_dir(cfg) / "pc.json", est["estimates"]["aipw"].to_dict(), cfg, True)
    # reweighting moves the benchmark toward the ranker's hotels; carrying over lands near the random hotels' truth
    assert pc["hotel_reweighted_benchmark"]["estimate"] > pc["plain_benchmark"]["estimate"]
    assert abs(pc["transported_estimate"]["estimate"] - truths["experimental_q1_rows"]) < 0.015


def test_feature_build_is_bitwise_reproducible(tmp_path):
    import pandas as pd
    from causallab import features
    from causallab.config import load_config
    from causallab.synthetic import expedia_like
    df, _ = expedia_like(n_searches=400, seed=3)
    src = tmp_path / "s.parquet"
    df.drop(columns=[c for c in ("quality", "score", "rand") if c in df.columns]).to_parquet(src, index=False)
    a = features.build_features(src, tmp_path / "a.parquet", load_config())
    b = features.build_features(src, tmp_path / "b.parquet", load_config())
    pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True)


def test_all_positions_window_runs_and_respects_firewall(cfg):
    from causallab import data, features
    from causallab.sensitivity_window import run_all_positions
    from causallab.synthetic import expedia_like
    raw, _ = expedia_like(n_searches=1500, seed=5)
    raw.to_parquet(data.sample_path(cfg, True), index=False)
    features.build_features(data.sample_path(cfg, True), data.features_path(cfg, True), cfg)
    obs = data.load_arm("observational", cfg, True)
    wide, narrow = features.all_positions_rows(obs), features.q1_rows(obs)
    assert len(wide) > len(narrow) and set(wide["position"]) > set(narrow["position"])
    assert not wide["position"].isin(cfg["data"]["reserved_positions"]).any()
    data.ACCESS_LOG.clear()
    r = run_all_positions(cfg, synthetic=True)
    assert {arm for arm, _ in data.ACCESS_LOG} == {"observational", "experimental"}
    assert {c for arm, c in data.ACCESS_LOG if arm == "experimental"} == {"causallab.benchmark"}
    assert r["benchmark"]["window"] == "all_positions" and set(r["gap"]) == set(r["estimates"])
    assert r["overlap"]["passed"] in (True, False)


def test_trap_scenarios_are_well_formed():
    from causallab.synthetic import trap_scenarios
    scs = trap_scenarios(n=3000, seed=1)
    assert [s.name for s in scs] == ["clean_confounding", "no_overlap", "collider", "mediator", "failing_balance"]
    for s in scs:
        assert isinstance(s.question, str) and s.question.endswith("?")
        assert isinstance(s.true_effect, float) and 0 < s.true_effect < 1
        assert {s.treatment, s.outcome, s.group} <= set(s.df.columns)
        assert set(s.reference_adjustment) <= set(s.df.columns) and set(s.bad_controls) <= set(s.df.columns)


def test_result_files_store_paths_relative_to_the_repository(tmp_path):
    from causallab.config import REPO_ROOT, display_path
    assert display_path(REPO_ROOT / "results" / "runs" / "r1" / "report.md") == "results/runs/r1/report.md"
    outside = tmp_path / "x.json"
    assert display_path(outside) == outside.as_posix()


def test_results_carry_the_code_version_and_save_false_writes_nothing(cfg):
    import json

    from causallab.config import code_version, results_dir, stamp
    v = code_version()
    assert len(v) == 12 and v == code_version() and stamp({"a": 1}) == {"a": 1, "code_version": v}
    raw, _ = expedia_like(n_searches=600, seed=5)
    raw.to_parquet(data.sample_path(cfg, True), index=False)
    features.build_features(data.sample_path(cfg, True), data.features_path(cfg, True), cfg)
    out = results_dir(cfg) / "benchmark_synthetic.json"
    benchmark.run_untrimmed(cfg, True, save=False)  # the notebooks' call: computes, never writes
    assert not out.exists()
    benchmark.run_untrimmed(cfg, True)
    assert json.loads(out.read_text(encoding="utf-8"))["code_version"] == v
