"""Phase 5 acceptance: offline end-to-end run, pause and resume from the checkpoint, the
firewall, the number guard, the refusal path and the re-specification rules."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from causallab import data, features
from causallab.agent import graph
from causallab.agent.report import Numbers, check_numbers, prose_is_clean
from causallab.agent.state import DagProposal, Edge
from causallab.synthetic import expedia_like


@pytest.fixture
def synthetic_expedia(cfg):
    raw, truth = expedia_like(n_searches=1500, seed=5)
    raw.to_parquet(data.sample_path(cfg, True), index=False)
    features.build_features(data.sample_path(cfg, True), data.features_path(cfg, True), cfg)
    return truth


def test_offline_q1_pauses_resumes_and_respects_firewall(cfg, synthetic_expedia):
    data.ACCESS_LOG.clear()
    r = graph.start("What is the causal effect of a top-3 position on clicks?", offline=True, synthetic=True, cfg=cfg)
    assert r["paused"] and r["review_request"]["treatment"] == "top3"
    for _ in range(3):  # a re-specification goes back through human review
        r = graph.resume(r["run_id"], {"decision": "approve"}, cfg)
        if not r["paused"]:
            break
    st = r["state"]
    assert st["outcome_status"] == "reported", st.get("verdict")
    assert st["benchmark"] is not None and st["gap"] is not None
    # firewall: only the benchmark module ever read the randomly ranked arm
    readers = {caller for arm, caller in data.ACCESS_LOG if arm == "experimental"}
    assert readers <= {"causallab.benchmark"}, readers
    # every number in the report was written by code
    report = Path(st["report_path"]).read_text(encoding="utf-8")
    assert "Experimental benchmark" in report
    trace = [json.loads(l) for l in Path(st["report_path"]).with_name("trace.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(t["event"] == "route" for t in trace)


def test_top3_question_cannot_target_experimental_arm(cfg, synthetic_expedia):
    from causallab.agent import nodes
    from causallab.agent.state import AgentState, QuestionSpec
    rt = nodes.make_runtime("fw", cfg)
    st = AgentState(run_id="fw", question="x", synthetic=True, offline=True,
                    spec=QuestionSpec(treatment="top3", outcome="click_bool", unit="u", population="p",
                                      target_arm="experimental"))
    with pytest.raises(PermissionError):
        nodes.load_target(rt, st)


def test_reject_goes_to_refusal(cfg):
    r = graph.start("q", dataset="trap:clean_confounding", offline=True, cfg=cfg)
    r = graph.resume(r["run_id"], {"decision": "reject"}, cfg)
    assert r["state"]["outcome_status"] == "refused"


def test_no_overlap_is_refused(cfg):
    r = graph.start("q", dataset="trap:no_overlap", offline=True, auto_approve=True, cfg=cfg)
    st = r["state"]
    assert st["outcome_status"] == "refused" and "overlap" in st["verdict"]["failures"]


def test_failing_balance_respecifies_without_dropping_confounders(cfg):
    r = graph.start("q", dataset="trap:failing_balance", offline=True, auto_approve=True, cfg=cfg)
    st = r["state"]
    assert st["attempts"] >= 1
    assert set(st["approved_confounders"]) <= set(st["dag"].adjustment_set)
    assert st["outcome_status"] == "refused"


def test_identify_rejects_mediator_and_collider(cfg):
    from causallab.agent import nodes
    from causallab.agent.state import AgentState, QuestionSpec
    rt = nodes.make_runtime("id", cfg)
    spec = QuestionSpec(treatment="T", outcome="Y", unit="u", population="p", target_arm="observational")
    edges = [Edge(source=a, target=b) for a, b in
             [("X", "T"), ("X", "Y"), ("T", "Y"), ("T", "M"), ("M", "Y"), ("T", "C"), ("Y", "C")]]
    for adj, ok in ((["X"], True), (["X", "M"], False), (["X", "C"], False), ([], False)):
        st = AgentState(run_id="id", question="q", spec=spec,
                        dag=DagProposal(roles=[], edges=edges, adjustment_set=adj, excluded=[]))
        assert nodes.identify(rt, st)["identification"]["valid"] is ok, adj


def test_number_guard():
    nums = Numbers()
    good = f"Effect {nums.fmt(0.0645)} pp in Q1 for top3, 95% interval."
    assert check_numbers(good, nums) == []
    assert check_numbers(good + " Claude says 7.1.", nums) == ["7.1"]
    assert prose_is_clean("The effect is {primary_estimate}.", {"primary_estimate"})
    assert not prose_is_clean("The effect is about 6 points.", {"primary_estimate"})
    assert not prose_is_clean("The effect is {made_up}.", {"primary_estimate"})
    # a placeholder name without braces would be left unfilled
    assert not prose_is_clean("It differs from benchmark_estimate.", {"benchmark_estimate"})


def test_identify_merges_variable_that_defines_the_treatment(cfg):
    """Claude's live proposal drew position -> top3 and position -> click. That is not a
    back-door path: top3 is a summary of position."""
    from causallab.agent import nodes
    from causallab.agent.state import AgentState, QuestionSpec, VariableRole
    rt = nodes.make_runtime("id2", cfg)
    spec = QuestionSpec(treatment="top3", outcome="click_bool", unit="u", population="p", target_arm="observational")
    edges = [Edge(source=a, target=b) for a, b in [("X", "top3"), ("X", "click_bool"), ("top3", "click_bool"),
                                                    ("position", "top3"), ("position", "click_bool")]]
    roles = [VariableRole(name="position", role="design", reason="defines top3")]
    st = AgentState(run_id="id2", question="q", spec=spec,
                    dag=DagProposal(roles=roles, edges=edges, adjustment_set=["X"], excluded=[]))
    out = nodes.identify(rt, st)["identification"]
    assert out["valid"] and out["merged_into_treatment"] == ["position"]
    # without the design role, the same graph has an open back-door path
    st.dag.roles = []
    assert nodes.identify(rt, st)["identification"]["valid"] is False


def test_negative_zero_prints_as_zero():
    from causallab.agent.report import Numbers
    n = Numbers()
    assert n.fmt(-0.00001) == "0.00"
    assert n.fmt(-0.0001, "x", 3) == "0.000"
    assert n.fmt(-0.0123) == "-1.23"


def test_post_hoc_placeholders_reach_prose():
    from causallab.agent import nodes, prompts
    from causallab.agent.report import prose_is_clean
    prose = ("Part of the gap is population: {reweighted_benchmark} gives {reweighted_gap} ({reweighted_verdict}); "
             "{transported_estimate} gives {transported_gap} ({transported_verdict}).")
    allowed = {"gap"} | nodes.POST_HOC_PLACEHOLDERS
    assert prose_is_clean(prose, allowed)
    assert not prose_is_clean("the reweighted_benchmark was close", allowed)
    (prompts.NARRATE + prompts.NARRATE_POST_HOC).format(placeholders="gap")


def test_scorecard_escapes_pipes_in_cells():
    from causallab.agent.report import Numbers, scorecard_table
    t = scorecard_table({"balance": {"passed": True, "max_abs_smd_after": 0.024, "worst_covariate": "x",
                                     "rule": "all |SMD| after weighting < 0.1"}}, Numbers())
    row = t.splitlines()[-1]
    assert row.count("|") - row.count("\\|") == 5  # four cells, five separators


def test_q3_design_and_threat_text_do_not_blame_the_ranker():
    from causallab.agent import nodes
    from causallab.agent.state import AgentState, QuestionSpec
    st = AgentState(run_id="r", question="Does the promotion badge cause more bookings?", dataset="expedia",
                    spec=QuestionSpec(treatment="promotion_flag", outcome="booking_bool", unit="u", population="p",
                                      target_arm="experimental"))
    assert "randomly ranked" in nodes._design_text(st)
    assert "ranker" not in nodes._template_prose(st, "interpretation")
    st.spec.treatment = "top3"
    st.spec.target_arm = "observational"
    assert "ranker" in nodes._template_prose(st, "interpretation")


def test_reviewer_edit_drops_unknown_columns(cfg, synthetic_expedia):
    r = graph.start("What is the causal effect of a top-3 position on clicks?", offline=True, synthetic=True, cfg=cfg)
    keep = r["review_request"]["adjustment_set"][:6]
    r = graph.resume(r["run_id"], {"decision": "edit", "adjustment_set": keep + ["not_a_column", keep[0]]}, cfg)
    while r["paused"]:
        r = graph.resume(r["run_id"], {"decision": "approve"}, cfg)
    st = r["state"]
    # six of the reference confounders cannot block every back door: identification must refuse
    assert st["outcome_status"] == "refused"
    assert st["identification"]["valid"] is False
    assert "back-door path open" in " ".join(st["identification"]["problems"])
    dag = st["dag"]
    adj = dag.adjustment_set if hasattr(dag, "adjustment_set") else dag["adjustment_set"]
    assert "not_a_column" not in adj and "not_a_column" not in st["approved_confounders"]
    assert adj[:len(keep)] == keep and len(set(adj)) == len(adj)


def test_bare_placeholder_names_are_repaired_not_rejected():
    from causallab.agent.report import prose_is_clean, repair_placeholders
    allowed = {"primary_estimate", "primary_ci_low", "primary_ci_high", "gap"}
    txt = "The effect was primary_estimate, from primary_ci_low to {primary_ci_high}; the gap is small."
    fixed = repair_placeholders(txt, allowed)
    assert fixed == "The effect was {primary_estimate}, from {primary_ci_low} to {primary_ci_high}; the gap is small."
    assert prose_is_clean(fixed, allowed)
    assert repair_placeholders("my_primary_estimate_x", allowed) == "my_primary_estimate_x"
    assert not prose_is_clean(repair_placeholders("It was 0.25 points.", allowed), allowed)


def test_cli_edit_without_adjust_file_is_a_clear_error(capsys):
    from causallab import cli
    with pytest.raises(SystemExit) as err:
        cli.main(["resume", "--run-id", "r1", "--decision", "edit"])
    assert err.value.code == 2 and "--adjust-file is required" in capsys.readouterr().err
