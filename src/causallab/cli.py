"""Command line: python -m causallab <command>

Phase 1  convert, prepare, eda
Phase 2  benchmark
Phase 3  estimate          (reference specification)
Phase 4  refute            (reference specification)
Phase 5  diagram, run --no-llm, resume
Phase 6  run, resume, table
Phase 7  q2, q3
Phase 8  traps
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config, load_spec, results_dir, stamp


def _pp(x: float) -> str:
    """Percentage points with two decimals; a value that rounds to zero prints as 0.00."""
    s = f"{x * 100:.2f}"
    return s[1:] if s in ("-0.00",) else s


def _print_review(req: dict) -> None:
    print("\n=== Human review: approve the assumptions before any estimate is computed ===")
    print(f"run id      : {req['run_id']}  (attempt {req['attempt']})")
    print(f"question    : effect of {req['treatment']} on {req['outcome']}, {req['target_arm']} arm")
    print(f"adjust for  : {len(req['adjustment_set'])} variables")
    for c in req["adjustment_set"]:
        print(f"    - {c}")
    print("excluded    :")
    for e in req["excluded"]:
        print(f"    - {e['name']}: {e['reason']}")
    print(f"threats     : {'; '.join(req['unobserved_threats']) or 'none listed'}")
    print(f"trim        : {req['trim'] or 'default'}   interactions: {req['interactions'] or 'none'}")
    print("\nResume with one of:")
    print(f"  python -m causallab resume --run-id {req['run_id']} --decision approve")
    print(f"  python -m causallab resume --run-id {req['run_id']} --decision reject")
    print(f"  python -m causallab resume --run-id {req['run_id']} --decision edit --adjust-file my_set.txt")


def _finish(r: dict) -> None:
    if r["paused"]:
        _print_review(r["review_request"])
        return
    st = r["state"]
    print(f"\nRun {r['run_id']}: {st.get('outcome_status')}. Report: {st.get('report_path')}")
    if st.get("verdict"):
        print(f"Checks failed: {st['verdict']['failures'] or 'none'}")
    if st.get("gap"):
        g = st["gap"]["primary"]
        print(f"Gap to benchmark (primary): {g['gap'] * 100:.2f} pp [{g['ci_low'] * 100:.2f}, {g['ci_high'] * 100:.2f}] -> {g['verdict']}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m causallab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("convert", help="stream train.csv to Parquet")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("prepare", help="sample searches and build features")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("eda", help="Phase 1 checks")
    s.add_argument("--synthetic", action="store_true")
    for name in ("benchmark", "estimate", "refute", "q2"):
        s = sub.add_parser(name)
        s.add_argument("--synthetic", action="store_true")
        if name == "q2":
            s.add_argument("--rows", type=int, default=None)
        if name == "estimate":
            s.add_argument("--all-positions", action="store_true",
                           help="spec sensitivity check: control group = every non-reserved position from 4 down")
    s = sub.add_parser("run", help="run the agent")
    s.add_argument("--question", default="What is the causal effect of a top-3 position on the probability a listing is clicked?")
    s.add_argument("--dataset", default="expedia")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--auto-approve", action="store_true")
    s.add_argument("--q2", action="store_true", help="also run the causal forest if the checks pass")
    s.add_argument("--exploratory-drop", default="", help="comma-separated columns to drop from the adjustment set; "
                   "the run is labelled exploratory")
    s = sub.add_parser("resume")
    s.add_argument("--run-id", required=True)
    s.add_argument("--decision", choices=["approve", "reject", "edit"], required=True)
    s.add_argument("--adjust-file", help="one column name per line, for --decision edit")
    s = sub.add_parser("q3", help="promotion badge: click and booking runs")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--synthetic", action="store_true")
    s.add_argument("--auto-approve", action="store_true")
    s = sub.add_parser("traps", help="Phase 8 trap suite")
    s.add_argument("--no-llm", action="store_true")
    s.add_argument("--runs", type=int, default=None)
    s.add_argument("--fresh", action="store_true", help="new run ids: rerun everything instead of reusing finished runs")
    sub.add_parser("diagram", help="export the agent graph (Mermaid)")
    sub.add_parser("version", help="print the code fingerprint that result files record")
    s = sub.add_parser("table", help="Section 8 results table from a run's files")
    s.add_argument("--run-id", required=True)

    a = p.parse_args(argv)
    cfg = load_config()
    t0 = time.time()

    if a.cmd == "convert":
        from .data import convert_csv_to_parquet
        print(convert_csv_to_parquet(force=a.force, cfg=cfg))
    elif a.cmd == "prepare":
        from . import data, features
        if a.synthetic:
            from .synthetic import expedia_like
            raw, truth = expedia_like(seed=cfg["seed"] % 1000)
            truths = dict(raw.attrs.get("truths", {"all_hotels": truth}))
            raw.to_parquet(data.sample_path(cfg, True), index=False)
            (results_dir(cfg) / "synthetic_truth.json").write_text(json.dumps(truths, indent=2), encoding="utf-8")
            print("Synthetic data written. True top-3 effect: " + ", ".join(
                f"{k.replace('_', ' ')} {v * 100:.2f} pp" for k, v in truths.items()))
            sample = data.sample_path(cfg, True)
        else:
            sample = data.build_sample(force=a.force, cfg=cfg)
        out = features.build_features(sample, data.features_path(cfg, a.synthetic), cfg)
        print(f"Features: {out}")
    elif a.cmd == "eda":
        from .eda import run_eda
        for line in run_eda(cfg, a.synthetic)["findings"]:
            print("-", line)
    elif a.cmd == "benchmark":
        from .benchmark import run_untrimmed
        r = run_untrimmed(cfg, a.synthetic)
        b, w = r["benchmark"], r["benchmark_reweighted_to_observational"]
        print(f"Randomisation check, max |SMD| top3 vs rest: {r['randomisation_max_abs_smd']:.3f}")
        for k, v in sorted(r["randomisation_balance_smd"].items(), key=lambda kv: -abs(kv[1]))[:5]:
            print(f"    {k:28s} {v:+.3f}")
        print(f"Benchmark: {b['estimate'] * 100:.2f} pp [{b['ci_low'] * 100:.2f}, {b['ci_high'] * 100:.2f}]")
        print(f"Reweighted to the observational searches: {w['estimate'] * 100:.2f} pp [{w['ci_low'] * 100:.2f}, {w['ci_high'] * 100:.2f}]")
    elif a.cmd == "estimate" and a.all_positions:
        from .sensitivity_window import run_all_positions
        r = run_all_positions(cfg, a.synthetic)
        b = r["benchmark"]
        pp = _pp
        o = r["overlap"]
        print(f"Control group: every non-reserved position from 4 down ({r['n_rows']:,} rows)")
        print(f"overlap: {'pass' if o['passed'] else 'FAIL'} ({o['trimmed_share']:.1%} trimmed; rule {o['rule']})")
        print(f"benchmark (trimmed) {pp(b['estimate']):>6s} pp  [{pp(b['ci_low'])}, {pp(b['ci_high'])}]")
        for k, v in r["estimates"].items():
            g = r["gap"].get(k)
            gs = f"  gap {pp(g['gap'])} [{pp(g['ci_low'])}, {pp(g['ci_high'])}] {g['verdict']}" if g else ""
            print(f"{k:16s} {pp(v['estimate']):>6s} pp  [{pp(v['ci_low'])}, {pp(v['ci_high'])}]{gs}")
    elif a.cmd in ("estimate", "refute"):
        from . import data, estimators, features, refuters
        spec = load_spec(cfg)
        sfx = "_synthetic" if a.synthetic else ""  # synthetic checks never overwrite the real reference files
        df = features.q1_rows(data.load_arm("observational", cfg, a.synthetic))
        outcome_path = results_dir(cfg) / "models" / f"reference_outcome{sfx}.pkl"
        est = estimators.estimate_all(df, spec["treatment"], spec["outcome"], spec["adjustment_set"], spec["group"], cfg,
                                      propensity_path=results_dir(cfg) / "models" / f"reference_propensity{sfx}.pkl",
                                      outcome_path=outcome_path)
        out = {k: v.to_dict() for k, v in est["estimates"].items()}
        (results_dir(cfg) / f"estimates_reference{sfx}.json").write_text(json.dumps(
            stamp({"estimates": out, "trimmed_share": est["trimmed_share"]}), indent=2), encoding="utf-8")
        for k, v in out.items():
            print(f"{k:16s} {v['estimate'] * 100:6.2f} pp  [{v['ci_low'] * 100:.2f}, {v['ci_high'] * 100:.2f}]")
        if a.cmd == "estimate":
            from .benchmark import population_matched
            pc = population_matched(outcome_path, results_dir(cfg) / f"population_checks_reference{sfx}.json",
                                    out[cfg["models"]["primary_estimator"]], cfg, a.synthetic)
            pp = _pp
            rw, gr, tr, gt = (pc["hotel_reweighted_benchmark"], pc["gap_vs_hotel_reweighted"],
                              pc["transported_estimate"], pc["gap_transported_vs_plain"])
            print("\nPost-hoc, like-for-like comparisons:")
            print(f"  ranker's hotels : benchmark reweighted {pp(rw['estimate'])} pp [{pp(rw['ci_low'])}, {pp(rw['ci_high'])}]"
                  f" -> gap {pp(gr['gap'])} [{pp(gr['ci_low'])}, {pp(gr['ci_high'])}] {gr['verdict']}"
                  f"  (max weight {rw['max_weight']:.1f}, effective sample {rw['effective_sample_share']:.0%})")
            print(f"  random hotels   : observational effect carried over {pp(tr['estimate'])} pp vs benchmark "
                  f"{pp(pc['plain_benchmark']['estimate'])} -> gap {pp(gt['gap'])} [{pp(gt['ci_low'])}, {pp(gt['ci_high'])}] {gt['verdict']}")
        if a.cmd == "refute":
            res = refuters.run_all(df, est, spec["treatment"], spec["outcome"], spec["adjustment_set"], spec["group"],
                                   spec.get("held_out_covariate"), cfg)
            (results_dir(cfg) / f"refutations_reference{sfx}.json").write_text(json.dumps(stamp(res), indent=2, default=float), encoding="utf-8")
            for k, v in res.items():
                print(f"{k:22s} {({True: 'pass', False: 'FAIL', None: 'reported'})[v['passed']]}")
    elif a.cmd == "q2":
        from . import data, features
        from .heterogeneity import run_q2
        spec = load_spec(cfg)
        df = features.q1_rows(data.load_arm("observational", cfg, a.synthetic))
        r = run_q2(df, spec["adjustment_set"], cfg, a.synthetic, results_dir(cfg) / f"q2_group_effects{'_synthetic' if a.synthetic else ''}.json", rows=a.rows)
        for g in r["groups"]:
            e = g["experimental"] or {}
            ee = f"{e['estimate'] * 100:6.2f}" if e else "   n/a"
            print(f"{g['modifier']:14s} {g['level']:15s} forest {g['estimate'] * 100:6.2f} pp   experimental {ee} pp")
    elif a.cmd == "run":
        from .agent import graph
        drop = [c for c in a.exploratory_drop.split(",") if c]
        override = [c for c in load_spec(cfg)["adjustment_set"] if c not in drop] if drop else None
        _finish(graph.start(a.question, dataset=a.dataset, offline=a.no_llm, synthetic=a.synthetic,
                            auto_approve=a.auto_approve, run_heterogeneity=a.q2, exploratory=bool(drop),
                            adjustment_override=override, cfg=cfg))
    elif a.cmd == "resume":
        from .agent import graph
        decision = {"decision": a.decision}
        if a.decision == "edit":
            if not a.adjust_file:
                p.error("--adjust-file is required when --decision is edit")
            decision["adjustment_set"] = [l.strip() for l in Path(a.adjust_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        _finish(graph.resume(a.run_id, decision, cfg))
    elif a.cmd == "q3":
        from .agent import graph
        for outcome in ("clicks", "bookings"):
            q = f"Does the promotion badge cause more {outcome}?"
            print(f"\n### {q}")
            _finish(graph.start(q, offline=a.no_llm, synthetic=a.synthetic, auto_approve=a.auto_approve, cfg=cfg))
    elif a.cmd == "traps":
        from .trap_suite import run
        r = run(offline=a.no_llm, runs=a.runs, cfg=cfg, fresh=a.fresh)
        for k, v in r["summary"].items():
            print(f"{k:18s} {v['as_expected']}/{v['runs']} as expected ({v['expected']})")
    elif a.cmd == "diagram":
        from .agent.graph import export_diagram
        print(export_diagram(results_dir(cfg) / "agent_graph.mmd", cfg))
    elif a.cmd == "version":
        from .config import code_version
        print(code_version())
    elif a.cmd == "table":
        from .results_table import write_table
        print(write_table(a.run_id, cfg))
    print(f"\n[{a.cmd} finished in {time.time() - t0:.0f} s]", file=sys.stderr)


if __name__ == "__main__":
    main()
