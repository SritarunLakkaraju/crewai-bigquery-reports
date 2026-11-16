"""FDE-3 evaluation (python -m fde3_crewai_bigquery.evaluate). Writes results/metrics.json.

* SQL accuracy on the 50-question test set (result sets compared against independently written gold SQL)
* guardrail checks: forbidden statements and an over-budget scan are rejected
* % of report numbers verified, and whether an injected invented number is caught and rewritten
* cost per report: bytes scanned x BigQuery on-demand price, LLM tokens (0 offline)
"""
from __future__ import annotations

import json
import os
import tempfile
import time

from .crew import OfflineCrewLLM, run_weekly_report
from .guard import NUM, find_unverified
from .nl2sql import TEST_SET, OfflineNL2SQL, gold_sql
from .warehouse import DuckDBWarehouse, QueryRejected

HERE = os.path.dirname(os.path.abspath(__file__))
BQ_USD_PER_TIB = 6.25  # on-demand price; check the current GCP price list for your region


def _norm(rows, ordered):
    r = [tuple(str(x) if not isinstance(x, (int, float)) else round(float(x), 2) for x in row) for row in rows]
    return r if ordered else sorted(r)


def sql_accuracy(wh=None) -> dict:
    wh = wh or DuckDBWarehouse(max_bytes=10**9)
    parser, ok, fails = OfflineNL2SQL(), 0, []
    for q, spec in TEST_SET:
        try:
            ordered = bool(spec.get("k"))
            good = _norm(wh.run(parser(q)).rows, ordered) == _norm(wh.run(gold_sql(spec)).rows, ordered)
            if not good:
                fails.append(f"{q}: wrong result")
        except Exception as e:  # a parse failure counts as wrong
            good = False
            fails.append(f"{q}: {e}")
        ok += good
    return {"correct": ok, "total": len(TEST_SET), "accuracy": round(ok / len(TEST_SET), 4), "failures": fails}


def guardrail_checks() -> dict:
    wh = DuckDBWarehouse(max_bytes=200_000)
    attacks = ["DROP TABLE shop.orders", "SELECT 1; DELETE FROM shop.users", "UPDATE shop.orders SET status='x'",
               "COPY shop.users TO 'leak.csv'", "ATTACH 'other.db'", "SELECT * FROM shop.order_items"]
    rejected = []
    for sql in attacks:
        try:
            wh.run(sql)
            rejected.append(False)
        except QueryRejected:
            rejected.append(True)
    big = DuckDBWarehouse(max_bytes=10**9)
    est_star = big.dry_run("SELECT * FROM shop.order_items")
    est_one = big.dry_run("SELECT sum(quantity) FROM shop.order_items")
    return {"attacks_rejected": f"{sum(rejected)}/{len(attacks)}",
            "dry_run_bytes_select_star": est_star, "dry_run_bytes_one_column": est_one,
            "row_limit": big.max_rows}


def evaluate(verbose=True) -> dict:
    out = {"sql": sql_accuracy(), "guardrails": guardrail_checks()}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.perf_counter()
        r = run_weekly_report(out_dir=tmp)
        secs = time.perf_counter() - t0
        nums = len(NUM.findall(r["report"]))
        out["report"] = {"numbers_in_report": nums, "unverified": r["unverified_numbers"],
                         "pct_verified": round(100 * (nums - len(r["unverified_numbers"])) / nums, 1),
                         "queries": r["queries"], "bytes_scanned": r["bytes_scanned"],
                         "bigquery_cost_usd": round(r["bytes_scanned"] / 2**40 * BQ_USD_PER_TIB, 8),
                         "llm_tokens": 0, "runtime_s": round(secs, 2), "charts": r["charts"],
                         "notes_cited": r["notes_cited"]}
        llm = OfflineCrewLLM(model="offline-crew", hallucinate_once=True)
        r2 = run_weekly_report(out_dir=tmp, llm=llm)
        out["guard_catches_invented_number"] = {"writer_drafts": llm.writer_drafts,
                                                "final_report_clean": not r2["unverified_numbers"],
                                                "invented_number_in_final": "37.9%" in r2["report"]}
    # sanity: the guard flags a number that is not in the data
    out["guard_unit_check"] = find_unverified("Revenue was $1,234.56", {1234.56}) == [] and \
        find_unverified("Revenue was $999.00", {1234.56}) == ["$999.00"]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    if verbose:
        print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    evaluate()
