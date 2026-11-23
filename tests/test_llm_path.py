"""The Claude code path, with a stand-in model (no API key, no network).

Checks that structured outputs flow through the nodes, that prose with digits is rejected in
favour of the template, and that a proposal adjusting for a bad control is caught by identify."""
from __future__ import annotations

from pathlib import Path

from causallab.agent import graph, llm
from causallab.agent.datasets import trap
from causallab.agent.state import QuestionSpec


def _fake(monkeypatch, cfg, scenario: str, add_bad: bool = False, prose: str = "Placeholder prose."):
    ctx = trap(scenario, cfg=cfg)

    def structured(role, cfg_, schema, system, user):
        if schema is QuestionSpec:
            return ctx.offline_intake("")
        dag = ctx.reference_dag(ctx.offline_intake(""))
        if add_bad:
            dag.adjustment_set += ctx.scenario.bad_controls
        return dag

    monkeypatch.setattr(llm, "structured", structured)
    monkeypatch.setattr(llm, "text", lambda role, cfg_, system, user: prose)


def test_llm_path_reports_with_template_when_prose_has_digits(monkeypatch, cfg):
    _fake(monkeypatch, cfg, "clean_confounding", prose="The effect is 9.8 points.\n\nIt is robust.")
    r = graph.start("Does the featured slot cause clicks?", dataset="trap:clean_confounding", offline=False,
                    auto_approve=True, cfg=cfg)
    st = r["state"]
    assert st["outcome_status"] == "reported"
    report = Path(st["report_path"]).read_text(encoding="utf-8")
    assert "9.8 points" not in report  # rejected prose never reaches the report


def test_llm_proposal_with_mediator_is_refused_by_identify(monkeypatch, cfg):
    _fake(monkeypatch, cfg, "mediator", add_bad=True)
    r = graph.start("Does the featured slot cause clicks?", dataset="trap:mediator", offline=False,
                    auto_approve=True, cfg=cfg)
    st = r["state"]
    assert st["outcome_status"] == "refused"
    assert not st["identification"]["valid"]
