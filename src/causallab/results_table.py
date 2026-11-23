"""Section 8 results table, filled only from a run's saved files."""
from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, results_dir

LABELS = [("naive", "Naive, observational arm"), ("regression", "Regression adjustment"),
          ("ipw", "Inverse propensity weighting"), ("aipw", "Doubly robust (AIPW)"), ("dml", "Double machine learning")]


def write_table(run_id: str, cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    run_dir = results_dir(cfg) / "runs" / run_id
    files = sorted(run_dir.glob("estimates_a*.json"))
    if not files:
        raise FileNotFoundError(f"No estimates in {run_dir}")
    d = json.loads(files[-1].read_text(encoding="utf-8"))
    est, bench, gap = d["estimates"], d["benchmark"], (d["gap"] or {}).get("by_estimator", {})
    def pp(x: float) -> str:
        s = f"{x * 100:.2f}"
        return s[1:] if s == "-0.00" else s
    rows = ["| Estimate of top-3 effect on click | Effect (pp) | 95% interval | Gap to benchmark |", "|---|---|---|---|"]
    if bench:
        rows.append(f"| Experimental benchmark (trimmed) | {pp(bench['estimate'])} | [{pp(bench['ci_low'])}, {pp(bench['ci_high'])}] | n/a |")
    for k, label in LABELS:
        if k in est:
            e = est[k]
            g = gap.get(k)
            gs = f"{pp(g['gap'])} [{pp(g['ci_low'])}, {pp(g['ci_high'])}] {g['verdict']}" if g else "n/a"
            rows.append(f"| {label} | {pp(e['estimate'])} | [{pp(e['ci_low'])}, {pp(e['ci_high'])}] | {gs} |")
    naive, primary = est.get("naive"), est.get(cfg["models"]["primary_estimator"])
    extra = []
    if bench and naive and primary and naive["estimate"] != bench["estimate"]:  # no naive gap, no share
        removed = (naive["estimate"] - primary["estimate"]) / (naive["estimate"] - bench["estimate"])
        extra.append(f"\nShare of the naive gap removed by adjustment (primary estimator): {removed:.0%}")
    pc = d.get("population_checks")
    if pc:
        rw, gr, tr, gt = (pc["hotel_reweighted_benchmark"], pc["gap_vs_hotel_reweighted"],
                          pc["transported_estimate"], pc["gap_transported_vs_plain"])
        extra.append("\nPost-hoc, like-for-like (added after the pre-registered comparison):\n")
        extra.append("| Comparison | Effect (pp) | Gap (pp) |\n|---|---|---|")
        extra.append(f"| Ranker's hotels: benchmark reweighted vs primary estimate | {pp(rw['estimate'])} [{pp(rw['ci_low'])}, {pp(rw['ci_high'])}] | "
                     f"{pp(gr['gap'])} [{pp(gr['ci_low'])}, {pp(gr['ci_high'])}] {gr['verdict']} |")
        extra.append(f"| Random hotels: observational effect carried over vs plain benchmark | {pp(tr['estimate'])} | "
                     f"{pp(gt['gap'])} [{pp(gt['ci_low'])}, {pp(gt['ci_high'])}] {gt['verdict']} |")
    sens_path = results_dir(cfg) / ("sensitivity_all_positions_synthetic.json" if d["info"].get("synthetic")
                                    else "sensitivity_all_positions.json")
    if bench and sens_path.exists():
        sw = json.loads(sens_path.read_text(encoding="utf-8"))
        sb, o = sw["benchmark"], sw["overlap"]
        extra.append("\nSpec sensitivity check: control group = every non-reserved position from 4 down "
                     f"(reference specification; overlap {'pass' if o['passed'] else 'FAIL'}, {o['trimmed_share']:.0%} trimmed):\n")
        extra.append("| Estimate | Effect (pp) | 95% interval | Gap to benchmark |\n|---|---|---|---|")
        extra.append(f"| Experimental benchmark (trimmed) | {pp(sb['estimate'])} | [{pp(sb['ci_low'])}, {pp(sb['ci_high'])}] | n/a |")
        for k, label in LABELS:
            if k in sw["estimates"]:
                e, g = sw["estimates"][k], sw["gap"][k]
                extra.append(f"| {label} | {pp(e['estimate'])} | [{pp(e['ci_low'])}, {pp(e['ci_high'])}] | "
                             f"{pp(g['gap'])} [{pp(g['ci_low'])}, {pp(g['ci_high'])}] {g['verdict']} |")
    out = run_dir / "results_table.md"
    out.write_text(f"# Results table for run {run_id}\n\nSource: {files[-1].name}\n\n" + "\n".join(rows) + "\n" + "\n".join(extra) + "\n", encoding="utf-8")
    return out
