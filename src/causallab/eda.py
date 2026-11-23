"""Phase 1 EDA: check every 'verify in EDA' item of the spec against the full Parquet file.

Firewall note: this prints click and booking rates per ARM (the spec's Phase 1 check), and row
counts by position, but never click rates by position, which would reveal the benchmark.
"""
from __future__ import annotations

import json
from typing import Any

import duckdb

from .config import display_path, load_config, processed_dir, results_dir, stamp


def run_eda(cfg: dict | None = None, synthetic: bool = False, save: bool = True) -> dict[str, Any]:
    cfg = cfg or load_config()
    from .data import sample_path
    src = (sample_path(cfg, synthetic) if synthetic else processed_dir(cfg) / "train.parquet").as_posix()
    con = duckdb.connect()

    def q(sql: str):
        return con.execute(sql.replace("FROM T", f"FROM '{src}'")).df()

    shape = q("SELECT count(*) AS n_rows, count(DISTINCT srch_id) AS n_searches, min(date_time) AS first, "
              "max(date_time) AS last FROM T").iloc[0]
    arms = q("SELECT random_bool, count(*) AS n_rows, count(DISTINCT srch_id) AS n_searches, avg(click_bool) AS click_rate, "
             "avg(booking_bool) AS booking_rate FROM T GROUP BY 1 ORDER BY 1")
    searches = q("WITH s AS (SELECT srch_id, any_value(random_bool) rb, sum(click_bool) c, sum(booking_bool) b FROM T GROUP BY 1) "
                 "SELECT rb AS random_bool, avg((c = 0)::INT) AS share_zero_click, avg((b > 0)::INT) AS share_with_booking FROM s GROUP BY 1 ORDER BY 1")
    pos = q("SELECT position, sum((random_bool = 0)::INT) AS observational_rows, sum((random_bool = 1)::INT) AS experimental_rows "
            "FROM T GROUP BY 1 ORDER BY 1")
    zeros = q("SELECT avg(coalesce(prop_starrating = 0, false)::INT) AS star_zero, avg(coalesce(prop_review_score = 0, false)::INT) AS review_zero, "
              "avg((prop_review_score IS NULL)::INT) AS review_null, avg(coalesce(prop_log_historical_price = 0, false)::INT) AS hist_price_zero, "
              "max(price_usd) AS max_price, quantile_cont(price_usd, 0.999) AS p999_price FROM T").iloc[0]
    cross = q("SELECT random_bool, avg(srch_length_of_stay) AS length_of_stay, avg(srch_booking_window) AS booking_window, "
              "avg(promotion_flag) AS promotion_share FROM T GROUP BY 1 ORDER BY 1")

    # reserved positions: far fewer rows than the median of positions 1-25, in BOTH arms
    p10 = pos[pos.position <= 25].set_index("position")
    med_o, med_e = p10.observational_rows.median(), p10.experimental_rows.median()
    reserved = [int(p) for p, r in p10.iterrows() if r.observational_rows < 0.25 * med_o and r.experimental_rows < 0.25 * med_e]
    random_share = float(arms.set_index("random_bool").n_searches.get(1, 0) / shape.n_searches)

    findings = [
        f"Rows: {int(shape.n_rows):,}; searches: {int(shape.n_searches):,} (spec: about 9.9 million rows).",
        f"Randomly ranked share of searches: {random_share:.1%} (spec: roughly one third).",
        "Share of searches with zero clicks: " + ", ".join(f"arm {int(r.random_bool)} = {r.share_zero_click:.1%}" for r in searches.itertuples())
        + " (spec: only searches with a click are in the file).",
        "Share of searches with a booking: " + ", ".join(f"arm {int(r.random_bool)} = {r.share_with_booking:.1%}" for r in searches.itertuples())
        + " (spec: bookings sampled differently across arms).",
        f"Reserved positions (under a quarter of the usual rows in both arms): {reserved} "
        f"(config: {cfg['data']['reserved_positions']}).",
        f"Zero-coded states: star {zeros.star_zero:.1%}, review {zeros.review_zero:.1%} (plus {zeros.review_null:.2%} null), "
        f"historical price {zeros.hist_price_zero:.1%}.",
        f"Price: 99.9th percentile {zeros.p999_price:,.0f}, maximum {zeros.max_price:,.0f} (winsorise).",
        "Arms differ on search covariates: " + "; ".join(
            f"arm {int(r.random_bool)}: stay {r.length_of_stay:.2f} nights, window {r.booking_window:.1f} days" for r in cross.itertuples()),
    ]
    shape_out = {"n_rows": int(shape.n_rows), "n_searches": int(shape.n_searches),
                 "first": str(shape["first"]), "last": str(shape["last"])}
    res = {"source": display_path(src), "shape": shape_out, "random_share": random_share,
           "arms": arms.to_dict("records"), "searches": searches.to_dict("records"),
           "rows_by_position": pos.to_dict("records"), "zero_coded": {k: float(v) for k, v in zeros.items()},
           "cross_arm": cross.to_dict("records"), "reserved_positions_found": reserved, "findings": findings}
    if save:  # the notebook passes save=False so it never rewrites the published file
        out = results_dir(cfg) / ("eda_synthetic.json" if synthetic else "eda.json")
        out.write_text(json.dumps(stamp(res), indent=2, default=str), encoding="utf-8")
    return res
