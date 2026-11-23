"""Phase 8: run the agent on the five synthetic trap scenarios and count what it did.

Each run uses a different data seed and auto-approves the agent's own proposal at human
review, so the outcome reflects the agent's judgement. In offline mode the proposal is the
hidden reference answer, so offline counts test the plumbing, not judgement.
"""
from __future__ import annotations

import json
import uuid
from collections import Counter
from typing import Any

from .agent import graph
from .config import load_config, results_dir, stamp
from .synthetic import trap_scenarios

EXPECTED = {
    "clean_confounding": "proceed and recover the effect",
    "no_overlap": "refuse",
    "collider": "exclude the collider and proceed",
    "mediator": "exclude the mediator and proceed",
    "failing_balance": "re-specify, then refuse",
}


def _classify(name: str, st: dict[str, Any], truth: float, bad: list[str], primary_name: str = "aipw") -> dict[str, Any]:
    status = st.get("outcome_status")
    dag = st.get("dag")
    if dag is None:
        adj = []
    elif isinstance(dag, dict):  # state reloaded from summary.json
        adj = list(dag.get("adjustment_set", []))
    else:
        adj = list(dag.adjustment_set)
    respecified = int(st.get("attempts", 0)) > 0
    primary = st.get("estimates", {}).get(primary_name)
    recovered = bool(primary and primary["ci_low"] <= truth <= primary["ci_high"])
    bad_in = [b for b in bad if b in adj]
    if name == "clean_confounding":
        ok = status == "reported" and recovered
    elif name == "no_overlap":
        ok = status == "refused"
    elif name in ("collider", "mediator"):
        ok = status == "reported" and not bad_in
    else:
        ok = status == "refused" and respecified
    return {"status": status, "respecified": respecified, "bad_controls_adjusted": bad_in, "recovered": recovered,
            "estimate": primary["estimate"] if primary else None, "truth": truth, "as_expected": ok}


def run(offline: bool, runs: int | None = None, cfg: dict | None = None, fresh: bool = False) -> dict[str, Any]:
    cfg = cfg or load_config()
    runs = runs or cfg["trap_suite"]["runs_per_scenario"]
    names = [s.name for s in trap_scenarios(n=200)]
    records = []
    # fresh: new run ids, so nothing is reused from earlier runs (and no old checkpoint is resumed)
    batch = uuid.uuid4().hex[:6] if fresh else ""
    by_seed: dict[int, dict] = {}  # each seed's five scenarios are built once, not once per scenario
    for name in names:
        for i in range(runs):
            if i not in by_seed:
                by_seed[i] = {s.name: s for s in trap_scenarios(n=cfg["trap_suite"]["rows"], seed=i)}
            sc = by_seed[i][name]
            run_id = f"trap-{name}-{i}-{'offline' if offline else 'llm'}" + (f"-{batch}" if fresh else "")
            run_dir = results_dir(cfg) / "runs" / run_id
            summary = run_dir / "summary.json"
            try:
                if summary.exists():          # finished earlier: reuse it, no new Claude calls
                    st = json.loads(summary.read_text(encoding="utf-8"))
                    if not st.get("outcome_status"):  # older runs: the status is in the report
                        report = (run_dir / "report.md").read_text(encoding="utf-8")
                        st["outcome_status"] = "refused" if "status: **refused**" in report else "reported"
                elif run_dir.exists():        # interrupted earlier: retry from its last checkpoint
                    st = graph.resume(run_id, None, cfg)["state"]
                else:
                    st = graph.start(sc.question, dataset=f"trap:{name}@{i}", offline=offline, auto_approve=True,
                                     cfg=cfg, run_id=run_id)["state"]
                rec = _classify(name, st, sc.true_effect, sc.bad_controls, cfg["models"]["primary_estimator"])
            except Exception as err:          # record and move on; a rerun retries it
                rec = {"status": "error", "error": f"{type(err).__name__}: {err}"[:300], "as_expected": False,
                       "respecified": False, "bad_controls_adjusted": [], "recovered": False, "estimate": None,
                       "truth": sc.true_effect}
            rec.update(scenario=name, seed=i, run_id=run_id)
            records.append(rec)
            print(f"{name:18s} seed {i}: {rec['status']:9s} expected={rec['as_expected']}")
    summary = {}
    for name in names:
        rs = [r for r in records if r["scenario"] == name]
        summary[name] = {"expected": EXPECTED[name], "runs": len(rs), "as_expected": sum(r["as_expected"] for r in rs),
                         "status_counts": dict(Counter(r["status"] for r in rs)),
                         "bad_control_adjusted": sum(bool(r["bad_controls_adjusted"]) for r in rs),
                         "respecified": sum(r["respecified"] for r in rs)}
    res = {"mode": "offline (reference stubs: plumbing test only)" if offline else "Claude", "summary": summary,
           "records": records}
    tag = "offline" if offline else "llm"
    if runs != cfg["trap_suite"]["runs_per_scenario"]:  # partial runs (e.g. notebook 07) keep their own file
        tag += f"_{runs}runs"
    out = results_dir(cfg) / f"trap_suite_{tag}.json"
    out.write_text(json.dumps(stamp(res), indent=2, default=str), encoding="utf-8")
    lines = [f"# Trap suite ({res['mode']})", "", "| Scenario | Expected | As expected | Runs | Outcomes | Bad control adjusted | Re-specified |",
             "|---|---|---|---|---|---|---|"]
    for k, v in summary.items():
        lines.append(f"| {k} | {v['expected']} | {v['as_expected']} | {v['runs']} | {v['status_counts']} | "
                     f"{v['bad_control_adjusted']} | {v['respecified']} |")
    (results_dir(cfg) / f"trap_suite_{tag}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res
