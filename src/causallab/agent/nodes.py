"""Node functions. Each takes the state and returns a dict of updates.

Claude nodes (intake, propose_dag, judge's rationale, narrate, refuse) call the LLM unless the
run is offline, in which case fixed stubs built from the reference answer are used.
Tool nodes (profile_data, identify, route_design, estimate, refute, heterogeneity) are code.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import networkx as nx
import pandas as pd
from langgraph.types import interrupt

from .. import benchmark, estimators, refuters
from ..config import code_version, display_path, load_config, load_spec, results_dir
from . import prompts
from .datasets import DatasetContext, get as get_dataset
from .report import Numbers, check_numbers, estimates_table, fill, prose_is_clean, repair_placeholders, scorecard_table
from .state import AgentState, DagProposal, QuestionSpec, Review


# ---------------------------------------------------------------- runtime and tracing

@dataclass
class Runtime:
    cfg: dict
    run_dir: Path
    trace_path: Path
    _ctx: dict[str, DatasetContext] = field(default_factory=dict)

    def ctx(self, state: AgentState) -> DatasetContext:
        if state.dataset not in self._ctx:
            name, _, seed = state.dataset.partition("@")
            self._ctx[state.dataset] = get_dataset(name, synthetic=state.synthetic, seed=int(seed or 0), cfg=self.cfg)
        return self._ctx[state.dataset]

    def log(self, run_id: str, node: str, event: str, detail: Any = None) -> None:
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "run_id": run_id, "node": node, "event": event,
                                "detail": detail}, default=str) + "\n")


def make_runtime(run_id: str, cfg: dict | None = None) -> Runtime:
    cfg = cfg or load_config()
    run_dir = results_dir(cfg) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return Runtime(cfg=cfg, run_dir=run_dir, trace_path=run_dir / "trace.jsonl")


def traced(rt: Runtime, name: str, fn):
    def wrapper(state: AgentState) -> dict[str, Any]:
        rt.log(state.run_id, name, "enter")
        t0 = time.time()
        out = fn(rt, state)
        rt.log(state.run_id, name, "exit", {"seconds": round(time.time() - t0, 2), "keys": sorted(out)})
        return out
    wrapper.__name__ = name
    return wrapper


def load_target(rt: Runtime, state: AgentState) -> pd.DataFrame:
    """The firewall. Estimation gets only the loader for the arm held in state, and a top3
    question can never be pointed at the randomly ranked arm."""
    arm = state.spec.target_arm
    if state.spec.treatment == "top3" and arm != "observational":
        raise PermissionError("Firewall: the top3 question may only read the observational arm.")
    ctx = rt.ctx(state)
    return ctx.prepare(ctx.load(arm), state.spec)


# ---------------------------------------------------------------- nodes

def intake(rt: Runtime, state: AgentState) -> dict[str, Any]:
    ctx = rt.ctx(state)
    if state.offline:
        spec = ctx.offline_intake(state.question)
    else:
        from . import llm
        user = (f"Dataset: {ctx.description}\nArms available: {', '.join(ctx.arms)}\nVariables:\n"
                + "\n".join(f"- {k}: {v}" for k, v in ctx.dictionary.items()) + f"\n\nQuestion: {state.question}")
        spec = llm.structured("intake", rt.cfg, QuestionSpec, prompts.INTAKE, user)
    notes = []
    if spec.target_arm not in ctx.arms:
        notes.append(f"intake chose arm {spec.target_arm}, not available; using observational")
        spec.target_arm = "observational"
    if spec.treatment == "top3" and spec.target_arm != "observational":
        notes.append("firewall: top3 questions use the observational arm only")
        spec.target_arm = "observational"
    for col in (spec.treatment, spec.outcome):
        if col not in ctx.dictionary:
            raise ValueError(f"intake returned unknown column {col!r}")
    rt.log(state.run_id, "intake", "spec", spec.model_dump())
    return {"spec": spec, "notes": state.notes + notes}


def profile_data(rt: Runtime, state: AgentState) -> dict[str, Any]:
    df = load_target(rt, state)
    s = state.spec
    g = rt.ctx(state).group
    miss = df.isna().mean()
    return {"profile": {
        "rows": int(len(df)), "groups": int(df[g].nunique()), "arm": s.target_arm,
        "treatment_share": float(df[s.treatment].mean()), "outcome_rate": float(df[s.outcome].mean()),
        "columns": int(df.shape[1]),
        "most_missing": {k: float(v) for k, v in miss.sort_values(ascending=False).head(8).items() if v > 0},
    }}


def _candidate_vars(ctx: DatasetContext, spec: QuestionSpec) -> dict[str, str]:
    return {k: v for k, v in ctx.dictionary.items() if k not in ("session_id", "srch_id")}


def propose_dag(rt: Runtime, state: AgentState) -> dict[str, Any]:
    ctx = rt.ctx(state)
    spec = state.spec
    attempt = state.attempts
    if attempt == 0:
        if state.offline:
            dag = ctx.reference_dag(spec)
        else:
            from . import llm
            user = (f"Dataset: {ctx.description}\nQuestion: {state.question}\nTreatment: {spec.treatment}\n"
                    f"Outcome: {spec.outcome}\nTarget arm: {spec.target_arm}\nVariables (name: description):\n"
                    + "\n".join(f"- {k}: {v}" for k, v in _candidate_vars(ctx, spec).items()))
            dag = llm.structured("propose_dag", rt.cfg, DagProposal, prompts.PROPOSE_DAG, user)
        if state.adjustment_override is not None:
            # Exploratory run: the dropped variables are treated as partly caused by the outcome
            # (e.g. scores computed from click logs), so the graph is redrawn to say so.
            from .state import Edge, Excluded
            dropped = set(dag.adjustment_set) - set(state.adjustment_override)
            dag.adjustment_set = list(state.adjustment_override)
            dag.edges = [e for e in dag.edges if e.source not in dropped] + \
                        [Edge(source=spec.outcome, target=v) for v in sorted(dropped)]
            dag.excluded += [Excluded(name=v, reason="exploratory: treated as partly caused by clicks")
                             for v in sorted(dropped)]
    else:
        prev = state.dag
        failing = state.refutations.get("balance", {}).get("failing", [])
        if state.offline:
            dag = prev.model_copy(deep=True)
            tighter = rt.cfg["loop"]["tighter_trims"]
            dag.trim = list(tighter[min(attempt, len(tighter)) - 1])
        else:
            from . import llm
            user = (f"Previous proposal:\n{prev.model_dump_json()}\n\nCovariates still unbalanced after weighting: "
                    f"{', '.join(failing)}\nApproved confounders that must stay: {', '.join(state.approved_confounders)}\n"
                    f"Variables available:\n" + "\n".join(f"- {k}: {v}" for k, v in _candidate_vars(ctx, spec).items()))
            dag = llm.structured("propose_dag", rt.cfg, DagProposal, prompts.PROPOSE_DAG + "\n\n" + prompts.RESPECIFY, user)
        # Code enforces the re-specification rules, whatever the proposal says.
        missing = [c for c in state.approved_confounders if c not in dag.adjustment_set]
        if missing:
            dag.adjustment_set += missing
            rt.log(state.run_id, "propose_dag", "restored_dropped_confounders", missing)
        old_trim = (prev.trim or rt.cfg["overlap"]["trim"])
        if dag.trim is None or dag.trim[0] < old_trim[0] or dag.trim[1] > old_trim[1]:
            dag.trim = list(old_trim)
    known = set(ctx.dictionary)
    unknown = [c for c in dag.adjustment_set if c not in known]
    if unknown:
        rt.log(state.run_id, "propose_dag", "dropped_unknown_columns", unknown)
        dag.adjustment_set = [c for c in dag.adjustment_set if c in known]
    dag.interactions = [p for p in dag.interactions if len(p) == 2 and all(c in dag.adjustment_set for c in p)]
    return {"dag": dag}


def human_review(rt: Runtime, state: AgentState) -> dict[str, Any]:
    """Pause for a person to approve, edit or reject the assumptions (not the results)."""
    dag = state.dag
    if state.auto_approve:
        decision = {"decision": "approve"}
    else:
        decision = interrupt({
            "run_id": state.run_id, "attempt": state.attempts, "treatment": state.spec.treatment,
            "outcome": state.spec.outcome, "target_arm": state.spec.target_arm,
            "adjustment_set": dag.adjustment_set, "excluded": [e.model_dump() for e in dag.excluded],
            "unobserved_threats": dag.unobserved_threats, "trim": dag.trim, "interactions": dag.interactions,
            "n_edges": len(dag.edges),
        })
    review = Review(**decision)
    updates: dict[str, Any] = {"review": review}
    if review.decision == "edit" and review.adjustment_set is not None:
        known = set(rt.ctx(state).dictionary)
        unknown = [c for c in review.adjustment_set if c not in known]
        if unknown:
            rt.log(state.run_id, "human_review", "dropped_unknown_columns", unknown)
        dag = dag.model_copy(deep=True)
        dag.adjustment_set = list(dict.fromkeys(c for c in review.adjustment_set if c in known))
        dag.interactions = [p for p in dag.interactions if all(c in dag.adjustment_set for c in p)]
        updates["dag"] = dag
    if review.decision in ("approve", "edit"):
        updates["approved_confounders"] = sorted(set(state.approved_confounders) | set(dag.adjustment_set))
    rt.log(state.run_id, "human_review", "decision", review.model_dump())
    return updates


def identify(rt: Runtime, state: AgentState) -> dict[str, Any]:
    """Back-door criterion on the graph built from the explicit edge list.

    Two conditions: (1) no adjustment variable is a descendant of the treatment, and
    (2) the set d-separates treatment and outcome once the treatment's outgoing edges are
    removed (DoWhy). DoWhy's check covers (2) only, so (1) is checked here with networkx."""
    from dowhy.graph import check_valid_backdoor_set

    s, dag = state.spec, state.dag
    G = nx.DiGraph([(e.source, e.target) for e in dag.edges])
    G.add_nodes_from([s.treatment, s.outcome, *dag.adjustment_set])
    # A variable that defines the treatment (role "design", e.g. position -> top3) is not a
    # confounder: top3 is a summary of position, so an edge position -> click is part of the
    # treatment's own effect. Merge such variables into the treatment node before the check.
    merged = []
    for r in dag.roles:
        v = r.name
        if r.role == "design" and v in G and G.has_edge(v, s.treatment) and v not in dag.adjustment_set:
            for child in list(G.successors(v)):
                if child != s.treatment:
                    G.add_edge(s.treatment, child)
            for parent in list(G.predecessors(v)):
                if parent != s.treatment:
                    G.add_edge(parent, s.treatment)
            G.remove_node(v)
            merged.append(v)
    problems = []
    if not nx.is_directed_acyclic_graph(G):
        problems.append("the proposed graph has a cycle")
    if not G.has_edge(s.treatment, s.outcome) and not nx.has_path(G, s.treatment, s.outcome):
        problems.append("the graph has no path from treatment to outcome")
    desc = nx.descendants(G, s.treatment) if s.treatment in G else set()
    bad_desc = sorted(set(dag.adjustment_set) & desc)
    if bad_desc:
        problems.append(f"adjustment set contains descendants of the treatment: {', '.join(bad_desc)}")
    dsep = None
    if not problems:
        dsep = bool(check_valid_backdoor_set(G, [s.treatment], [s.outcome], list(dag.adjustment_set))["is_dseparated"])
        if not dsep:
            problems.append("the adjustment set leaves a back-door path open")
    ident = {"valid": not problems, "problems": problems, "d_separated": dsep, "descendants_in_set": bad_desc,
             "merged_into_treatment": merged,
             "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}
    rt.log(state.run_id, "identify", "result", ident)
    return {"identification": ident}


def route_design(rt: Runtime, state: AgentState) -> dict[str, Any]:
    df = load_target(rt, state)
    t = df[state.spec.treatment]
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    ok = n1 >= 50 and n0 >= 50 and t.nunique() == 2
    reason = None if ok else f"the treatment does not vary enough ({n1} treated, {n0} control rows)"
    return {"design": {"proceed": ok, "reason": reason, "n_treated": n1, "n_control": n0}}


def _with_interactions(df: pd.DataFrame, dag: DagProposal) -> tuple[pd.DataFrame, list[str]]:
    cols = list(dag.adjustment_set)
    for a, b in dag.interactions:
        name = f"{a}_x_{b}"
        df[name] = df[a] * df[b]
        cols.append(name)
    return df, cols


def estimate(rt: Runtime, state: AgentState) -> dict[str, Any]:
    s, dag, ctx = state.spec, state.dag, rt.ctx(state)
    df, cov = _with_interactions(load_target(rt, state).copy(), dag)
    trim = dag.trim or rt.cfg["overlap"]["trim"]
    tag = f"{state.run_id}_a{state.attempts}"
    prop_path = results_dir(rt.cfg) / "models" / f"{tag}_propensity.pkl"
    outcome_path = results_dir(rt.cfg) / "models" / f"{tag}_outcome.pkl"
    try:
        est = estimators.estimate_all(df, s.treatment, s.outcome, cov, ctx.group, rt.cfg, trim=trim,
                                      propensity_path=prop_path, outcome_path=outcome_path)
    except estimators.OverlapError as err:
        rt.log(state.run_id, "estimate", "overlap_error", str(err))
        return {"estimates": {}, "estimation_info": {"trim": list(trim), "overlap_error": str(err)}}
    joblib.dump({"est": {k: v for k, v in est.items() if k not in ("ols_fit", "ols_design")}, "covariates": cov},
                rt.run_dir / f"estimation_cache_a{state.attempts}.pkl")
    estimates = {k: v.to_dict() for k, v in est["estimates"].items()}
    info = {"trim": est["trim"], "trimmed_share": est["trimmed_share"], "n_before_trim": est["n_before_trim"],
            "covariates": cov, "propensity_model": display_path(prop_path), "synthetic": state.synthetic}
    updates: dict[str, Any] = {"estimates": estimates, "estimation_info": info}
    if ctx.has_benchmark(s):
        # The benchmark module, not this path, reads the experimental arm.
        bench = benchmark.run_trimmed(prop_path, rt.run_dir / f"benchmark_trimmed_a{state.attempts}.json",
                                      rt.cfg, synthetic=state.synthetic)
        tol = rt.cfg["gap"]["tolerance_share_of_benchmark"]
        gaps = {k: benchmark.gap_test(v["estimate"], v["se"], bench, tol) for k, v in estimates.items()}
        updates["benchmark"] = bench
        updates["gap"] = {"primary": gaps[rt.cfg["models"]["primary_estimator"]], "by_estimator": gaps}
        # Post-hoc like-for-like comparisons (benchmark module; covariates only from this arm)
        updates["population_checks"] = benchmark.population_matched(
            outcome_path, rt.run_dir / f"population_checks_a{state.attempts}.json",
            estimates[rt.cfg["models"]["primary_estimator"]], rt.cfg, synthetic=state.synthetic)
    (rt.run_dir / f"estimates_a{state.attempts}.json").write_text(json.dumps(
        {"estimates": estimates, "info": info, "benchmark": updates.get("benchmark"), "gap": updates.get("gap"),
         "population_checks": updates.get("population_checks")}, indent=2), encoding="utf-8")
    return updates


def refute(rt: Runtime, state: AgentState) -> dict[str, Any]:
    s, ctx = state.spec, rt.ctx(state)
    if state.estimation_info.get("overlap_error"):
        return {"refutations": {"overlap": {"check": "overlap", "trimmed_share": 1.0, "trim": state.estimation_info["trim"],
                                            "rule": "after trimming, every fold must contain treated and control rows",
                                            "passed": False}}}
    cache = joblib.load(rt.run_dir / f"estimation_cache_a{state.attempts}.pkl")
    df, cov = _with_interactions(load_target(rt, state).copy(), state.dag)
    est = cache["est"]
    est["estimates"] = {k: estimators.EstimateResult(**{f: v[f] for f in ("name", "estimate", "se", "n_rows", "n_groups", "extra")})
                        for k, v in state.estimates.items()}
    held = ctx.held_out if ctx.held_out in cov else None
    res = refuters.run_all(df, est, s.treatment, s.outcome, cov, ctx.group, held, rt.cfg)
    (rt.run_dir / f"refutations_a{state.attempts}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("bounds", "smd_before", "smd_after", "estimates")}
            for k, v in res.items()}
    return {"refutations": json.loads(json.dumps(slim, default=float))}


def judge(rt: Runtime, state: AgentState) -> dict[str, Any]:
    """Code applies the thresholds and picks the route; Claude only explains it."""
    failures = [k for k, v in state.refutations.items() if v.get("passed") is False]
    max_loops = rt.cfg["loop"]["max_respecifications"]
    # Overlap and sensitivity failures cannot be fixed by re-specifying, so they refuse at once.
    # A balance failure gets another specification while attempts remain, even if stability
    # checks failed with it (poor balance can cause the instability).
    if not failures:
        route = "pass"
    elif {"overlap", "sensitivity"} & set(failures):
        route = "refuse"
    elif "balance" in failures and state.attempts < max_loops:
        route = "respecify"
    else:
        route = "refuse"
    entry = {"attempt": state.attempts, "adjustment_set_size": len(state.dag.adjustment_set),
             "trim": state.estimation_info.get("trim"), "interactions": state.dag.interactions,
             "failures": failures, "route": route,
             "primary_estimate": state.estimates.get(rt.cfg["models"]["primary_estimator"], {}).get("estimate")}
    rationale = _judge_rationale(rt, state, route, failures)
    rt.log(state.run_id, "judge", "route", entry)
    upd: dict[str, Any] = {"verdict": {"route": route, "failures": failures, "rationale": rationale},
                           "history": state.history + [entry]}
    if route == "respecify":
        upd["attempts"] = state.attempts + 1
    return upd


def _judge_rationale(rt: Runtime, state: AgentState, route: str, failures: list[str]) -> str:
    default = {"pass": "Every pass-or-fail check passed, so the estimate is reported.",
               "respecify": "Covariate balance after weighting failed, so the specification goes back for revision "
                            "without dropping any approved confounder.",
               "refuse": f"The run was refused because these checks failed: {', '.join(failures)}."}[route]
    if state.offline:
        return default
    from . import llm
    summary = {k: {"passed": v.get("passed"), "rule": v.get("rule")} for k, v in state.refutations.items()}
    txt = llm.text("judge", rt.cfg, prompts.JUDGE.format(route=route), json.dumps(summary))
    return txt if prose_is_clean(txt, set()) else default


def heterogeneity(rt: Runtime, state: AgentState) -> dict[str, Any]:
    from ..heterogeneity import run_q2
    df, cov = _with_interactions(load_target(rt, state).copy(), state.dag)
    res = run_q2(df, cov, rt.cfg, synthetic=state.synthetic, out_path=rt.run_dir / "q2_group_effects.json")
    return {"heterogeneity": res}


# ---------------------------------------------------------------- reports

def _values(rt: Runtime, state: AgentState, nums: Numbers) -> dict[str, str]:
    p = rt.cfg["models"]["primary_estimator"]
    v = {"primary_estimate": "n/a", "primary_ci_low": "n/a", "primary_ci_high": "n/a",
         "naive_estimate": "n/a", "benchmark_estimate": "n/a", "gap": "n/a", "gap_verdict": "n/a"}
    if p in state.estimates:
        r = state.estimates[p]
        v.update(primary_estimate=nums.fmt(r["estimate"]) + " percentage points",
                 primary_ci_low=nums.fmt(r["ci_low"]), primary_ci_high=nums.fmt(r["ci_high"]))
    if "naive" in state.estimates:
        v["naive_estimate"] = nums.fmt(state.estimates["naive"]["estimate"]) + " percentage points"
    if state.benchmark:
        v["benchmark_estimate"] = nums.fmt(state.benchmark["estimate"]) + " percentage points"
    if state.gap:
        v["gap"] = nums.fmt(state.gap["primary"]["gap"]) + " percentage points"
        v["gap_verdict"] = state.gap["primary"]["verdict"]
    if state.population_checks:
        pc = state.population_checks
        v.update(reweighted_benchmark=nums.fmt(pc["hotel_reweighted_benchmark"]["estimate"]) + " percentage points",
                 reweighted_gap=nums.fmt(pc["gap_vs_hotel_reweighted"]["gap"]) + " percentage points",
                 reweighted_verdict=pc["gap_vs_hotel_reweighted"]["verdict"],
                 transported_estimate=nums.fmt(pc["transported_estimate"]["estimate"]) + " percentage points",
                 transported_gap=nums.fmt(pc["gap_transported_vs_plain"]["gap"]) + " percentage points",
                 transported_verdict=pc["gap_transported_vs_plain"]["verdict"])
    return v


POST_HOC_PLACEHOLDERS = {"reweighted_benchmark", "reweighted_gap", "reweighted_verdict",
                         "transported_estimate", "transported_gap", "transported_verdict"}


def _template_prose(state: AgentState, kind: str) -> str:
    expedia = state.dataset.startswith("expedia")
    if kind == "summary":
        base = ("The primary doubly robust estimate of the effect is {primary_estimate}, with a ninety-five percent "
                "interval from {primary_ci_low} to {primary_ci_high}. The naive comparison gave {naive_estimate}.")
        if state.benchmark:
            base += (" The randomly ranked benchmark, trimmed by the same rule, is {benchmark_estimate}; the gap is "
                     "{gap} ({gap_verdict}).")
        return base
    if expedia and state.spec and state.spec.treatment == "top3":
        threat = "information the ranker used and the file does not contain"
    elif expedia:
        threat = "hotel and market factors the file does not contain that drive both the treatment and the outcome"
    else:
        threat = "common causes that are not in the data"
    design = ""
    if expedia and state.spec and state.spec.treatment != "top3":
        design = (" Position is random in these searches, but the treatment is not, and no randomised benchmark "
                  "exists for it, so the result is suggestive, not experimentally verified.")
    post_hoc = ""
    if state.population_checks:
        post_hoc = (" Part of the gap to the benchmark may come from comparing different hotels rather than from bias: "
                    "on the ranker's hotels the reweighted benchmark is {reweighted_benchmark} (gap {reweighted_gap}, "
                    "{reweighted_verdict}), and on the randomly ranked hotels the observational effect carried over is "
                    "{transported_estimate} (gap {transported_gap}, {transported_verdict}). These comparisons were "
                    "added after the pre-registered one.")
    return ("Every reported number comes from saved results. The checks in the scorecard limit what the data can "
            f"reveal about bias, but they cannot detect confounding by {threat}; the sensitivity analysis "
            "describes how strong such a confounder would need to be." + design + post_hoc)


def _write_report(rt: Runtime, state: AgentState, status: str, summary: str, interpretation: str) -> str:
    nums = Numbers()
    ctx = rt.ctx(state)
    vals = _values(rt, state, nums)
    s, dag = state.spec, state.dag
    bench_section = ""
    if state.benchmark:
        b = state.benchmark
        g = state.gap["primary"]
        bench_section = (f"## Comparison with the experimental benchmark\n\nThe benchmark is the difference in click rates "
                         f"in randomly ranked searches, on the rows that survive the same trimming rule "
                         f"({nums.fmt(b['trimmed_share'], 'pct')} of rows trimmed). Primary gap: "
                         f"{nums.fmt(g['gap'])} pp, 95% interval [{nums.fmt(g['ci_low'])}, {nums.fmt(g['ci_high'])}], "
                         f"tolerance ±{nums.fmt(g['tolerance'])} pp: **{g['verdict']}**.")
    if state.population_checks:
        pc = state.population_checks
        rw, gr = pc["hotel_reweighted_benchmark"], pc["gap_vs_hotel_reweighted"]
        tr, gt = pc["transported_estimate"], pc["gap_transported_vs_plain"]
        bench_section += (
            "\n\n### Post-hoc: comparing like with like\n\n"
            "Added after the pre-registered comparison: the ranker fills the top ten with different hotels than "
            "random ranking does, and the effect differs by hotel, so the gap above mixes leftover confounding with "
            "a difference in population.\n\n"
            f"- **Ranker's hotels:** benchmark reweighted to resemble them {nums.fmt(rw['estimate'])} pp "
            f"[{nums.fmt(rw['ci_low'])}, {nums.fmt(rw['ci_high'])}]; gap to the primary estimate {nums.fmt(gr['gap'])} pp "
            f"[{nums.fmt(gr['ci_low'])}, {nums.fmt(gr['ci_high'])}], {gr['verdict']}. The weakest of the comparisons: "
            "the reweighting under-corrects (it did on synthetic data, where the truth is known), and its interval "
            "ignores the uncertainty in the weights.\n"
            f"- **Randomly ranked hotels:** observational effect carried over to them {nums.fmt(tr['estimate'])} pp "
            f"against the benchmark {nums.fmt(pc['plain_benchmark']['estimate'])} pp; gap {nums.fmt(gt['gap'])} pp "
            f"[{nums.fmt(gt['ci_low'])}, {nums.fmt(gt['ci_high'])}], {gt['verdict']} (interval covers the "
            "randomly ranked sample only, not the models' own uncertainty).")
    ref_diff = ""
    if ctx.name.startswith("expedia") and s and s.treatment == "top3" and dag:
        ref = set(load_spec(rt.cfg)["adjustment_set"])
        added, removed = sorted(set(dag.adjustment_set) - ref), sorted(ref - set(dag.adjustment_set))
        ref_diff = (f"\n\nDifference from the reference specification (config/spec_locked.yaml): added "
                    f"{', '.join(added) or 'none'}; removed {', '.join(removed) or 'none'}.")
    hist = "\n".join(f"- Attempt {nums.fmt(h['attempt'], 'int')}: {nums.fmt(h['adjustment_set_size'], 'int')} adjustment "
                     f"variables, failures: {', '.join(h['failures']) or 'none'}, route: {h['route']}" for h in state.history)
    limitations = list(prompts.EXPEDIA_LIMITATIONS) if ctx.name.startswith("expedia") else \
        ["Synthetic scenario: the data-generating process is known and simple."]
    if ctx.name.startswith("expedia") and s and s.treatment == "top3":
        limitations.insert(0, prompts.RANKER_LIMITATION)
    if ctx.name.startswith("expedia") and s and s.outcome == "booking_bool":
        limitations.append(prompts.BOOKING_CAVEAT)
    # Descriptive text (the question, the population, fixed limitations) may contain numbers that
    # are not results. Register them as written by code so the guard only polices results.
    for txt in [state.question, state.run_id, state.dataset, *(limitations), *((s.unit, s.population) if s else ()),
                *(dag.unobserved_threats if dag else ())]:
        nums.text(txt)
    report = prompts.REPORT_TEMPLATE.format(
        question=state.question, run_id=state.run_id, status=status, dataset=ctx.name, group=ctx.group,
        exploratory_label=" · **EXPLORATORY: specified after the benchmark was seen**" if state.exploratory else "",
        treatment=s.treatment if s else "n/a", outcome=s.outcome if s else "n/a",
        target_arm=s.target_arm if s else "n/a", unit=s.unit if s else "n/a", population=s.population if s else "n/a",
        summary_prose=fill(summary, vals),
        estimates_table=estimates_table(state.estimates, nums, state.benchmark,
                                        state.gap["by_estimator"] if state.gap else None) if state.estimates else "No estimates were produced.",
        benchmark_section=bench_section,
        scorecard_table=scorecard_table(state.refutations, nums) if state.refutations else "No checks were run.",
        interpretation_prose=fill(interpretation, vals),
        n_adjust=nums.fmt(len(dag.adjustment_set), "int") if dag else "0",
        adjustment_list=", ".join(f"`{c}`" for c in dag.adjustment_set) if dag else "none",
        threats="; ".join(dag.unobserved_threats) if dag and dag.unobserved_threats else "none listed",
        limitations="\n".join(f"- {x}" for x in limitations),
        history=(hist or "- Single attempt.") + ref_diff,
    )
    if status == "reported" and ctx.name.startswith("expedia") and s and not ctx.has_benchmark(s):
        # No randomised truth exists for this treatment (Q3), so the verdict is capped.
        report = report.replace("## Question", "## Verdict\n\n**Suggestive, not experimentally verified.** "
                                "The treatment was not randomised in either arm; see the sensitivity row of the "
                                "scorecard.\n\n## Question", 1)
    if state.heterogeneity:
        report += ("\n## Heterogeneous effects (Q2)\n\n" + _q2_table(state.heterogeneity, nums) +
                   "\n\nForest intervals are not clustered by search; experimental intervals are (cluster bootstrap).\n")
    stray = check_numbers(report, nums)
    if stray:
        raise AssertionError(f"Report contains numbers not written by code: {stray}")
    path = rt.run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    dump = state.model_dump()
    dump.update(outcome_status=status, report_path=display_path(path), code_version=code_version())
    (rt.run_dir / "summary.json").write_text(json.dumps(dump, indent=2, default=str), encoding="utf-8")
    return str(path)


def _q2_table(h: dict, nums: Numbers) -> str:
    rows = ["| Group | Forest estimate (pp) | 95% interval | Experimental (pp) | 95% interval |", "|---|---|---|---|---|"]
    for r in h["groups"]:
        e = r.get("experimental") or {}
        rows.append(f"| {r['modifier']} = {r['level']} | {nums.fmt(r['estimate'])} | [{nums.fmt(r['ci_low'])}, "
                    f"{nums.fmt(r['ci_high'])}] | {nums.fmt(e.get('estimate'))} | [{nums.fmt(e.get('ci_low'))}, {nums.fmt(e.get('ci_high'))}] |")
    return "\n".join(rows)


def _design_text(state: AgentState) -> str:
    """How the estimate was obtained, stated by code so Claude cannot misdescribe the design."""
    s = state.spec
    if state.dataset.startswith("expedia") and s and s.treatment == "top3":
        return ("Observational estimate from algorithm-ranked searches, where the ranker chose the positions. "
                "It is compared with randomly ranked searches, where position was random.")
    if state.dataset.startswith("expedia") and s:
        return ("Estimated inside randomly ranked searches, so position is not a confounder. The treatment itself "
                "was not randomised (hotels and Expedia decide it), so no experimental benchmark exists and the "
                "verdict is capped at suggestive, not experimentally verified.")
    return "Observational estimate with adjustment for the approved covariates; no experimental benchmark."


def narrate(rt: Runtime, state: AgentState) -> dict[str, Any]:
    summary, interp = _template_prose(state, "summary"), _template_prose(state, "interpretation")
    if not state.offline:
        from . import llm
        allowed = {"primary_estimate", "primary_ci_low", "primary_ci_high", "naive_estimate",
                   "benchmark_estimate", "gap", "gap_verdict"}
        status = {True: "passed", False: "failed", None: "run and reported (not a pass or fail check)"}
        facts = {"question": state.question, "design": _design_text(state),
                 "unobserved_threats": state.dag.unobserved_threats if state.dag else [],
                 "checks": {k: status[v.get("passed")] for k, v in state.refutations.items()},
                 "has_benchmark": bool(state.benchmark), "gap_verdict": (state.gap or {}).get("primary", {}).get("verdict")}
        if state.population_checks:
            pc = state.population_checks
            allowed |= POST_HOC_PLACEHOLDERS
            facts["post_hoc_population_checks"] = {
                "why": ("The ranker puts different hotels in the top ten than random ranking does, and the effect "
                        "differs by hotel, so the plain gap mixes leftover confounding with a population difference. "
                        "Two comparisons added after the pre-registered one put both estimates on the same hotels."),
                "ranker_hotels_benchmark_reweighted_verdict": pc["gap_vs_hotel_reweighted"]["verdict"],
                "random_hotels_observational_effect_carried_over_verdict": pc["gap_transported_vs_plain"]["verdict"],
                "caveat": ("Reweighting cannot correct for information only the ranker had; the carried-over "
                           "estimate's interval ignores the outcome models' own uncertainty.")}
        prompt = prompts.NARRATE + (prompts.NARRATE_POST_HOC if state.population_checks else "")
        txt = llm.text("narrate", rt.cfg, prompt.format(placeholders=", ".join(sorted(allowed))), json.dumps(facts))
        txt = repair_placeholders(txt, allowed)
        parts = [p.strip() for p in txt.split("\n\n") if p.strip()]
        if len(parts) >= 2 and all(prose_is_clean(p, allowed) for p in parts[:2]):
            summary, interp = parts[0], parts[1]
        else:
            rt.log(state.run_id, "narrate", "llm_prose_rejected", txt[:500])
    path = _write_report(rt, state, "reported", summary, interp)
    return {"report_path": path, "outcome_status": "reported"}


def refuse(rt: Runtime, state: AgentState) -> dict[str, Any]:
    reasons = []
    if state.review and state.review.decision == "reject":
        reasons.append("the reviewer rejected the proposed assumptions")
    if state.identification and not state.identification["valid"]:
        reasons += state.identification["problems"]
    if state.design and not state.design["proceed"]:
        reasons.append(state.design["reason"])
    if state.verdict and state.verdict["route"] == "refuse":
        reasons.append("failed checks: " + ", ".join(state.verdict["failures"]))
    default = ("The effect cannot be credibly estimated from these data because "
               + "; ".join(r.replace("_", " ") for r in reasons) + ". A randomized experiment, better overlap between "
               "treated and untreated units, or measurement of the missing common causes would fix this. "
               "Any estimates below are shown for transparency and are not credible estimates.")
    default = default if prose_is_clean(default, set()) else ("The effect cannot be credibly estimated from these data; "
                                                               "see the failed checks below.")
    prose = default
    if not state.offline:
        from . import llm
        txt = llm.text("refuse", rt.cfg, prompts.REFUSE, json.dumps({"question": state.question, "reasons": reasons}))
        if prose_is_clean(txt, set()):
            prose = txt + " Any estimates below are shown for transparency and are not credible estimates."
    path = _write_report(rt, state, "refused", prose, (state.verdict or {}).get("rationale", "") or
                         "No estimate is reported as credible.")
    return {"report_path": path, "outcome_status": "refused", "notes": state.notes + reasons}
