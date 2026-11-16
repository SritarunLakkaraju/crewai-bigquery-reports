import os

import pytest

from fde3_crewai_bigquery.guard import allowed_numbers, find_unverified, pct_change
from fde3_crewai_bigquery.nl2sql import OfflineNL2SQL, week_range
from fde3_crewai_bigquery.warehouse import DuckDBWarehouse, QueryRejected, check_read_only


@pytest.fixture(scope="module")
def wh():
    return DuckDBWarehouse(max_bytes=10**9)


@pytest.mark.parametrize("sql", ["DROP TABLE shop.orders", "SELECT 1; SELECT 2", "delete from shop.users",
                                 "WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x", "PRAGMA database_list",
                                 "COPY shop.users TO 'x.csv'", "  -- comment\n UPDATE shop.orders SET status='x'"])
def test_read_only_guard_rejects(sql):
    with pytest.raises(QueryRejected):
        check_read_only(sql)


def test_keywords_inside_strings_are_fine():
    assert check_read_only("SELECT * FROM shop.products WHERE name = 'Drop Shoulder Tee'")


def test_dry_run_counts_only_referenced_columns(wh):
    assert wh.dry_run("SELECT sum(quantity) FROM shop.order_items") < wh.dry_run("SELECT * FROM shop.order_items")


def test_byte_budget_and_row_limit():
    small = DuckDBWarehouse(max_bytes=100_000, max_rows=7)
    with pytest.raises(QueryRejected, match="byte budget"):
        small.run("SELECT * FROM shop.order_items")
    assert len(small.run("SELECT user_id FROM shop.users").rows) == 7


def test_invalid_sql_is_rejected_cleanly(wh):
    with pytest.raises(QueryRejected, match="invalid SQL"):
        wh.run("SELECT nope FROM shop.orders")


def test_parser_slots():
    p = OfflineNL2SQL()
    s = p.parse("Top 3 categories by revenue in week 11")
    assert (s.metric, s.group_by, s.top_k, s.start) == ("revenue", "category", 3, week_range(11)[0])
    s = p.parse("Orders from Social channel users last week")
    assert s.group_by is None and s.filters == {"channel": "Social"}
    with pytest.raises(ValueError):
        p.parse("what is the weather")


def test_sql_test_set_accuracy():
    from fde3_crewai_bigquery.evaluate import sql_accuracy

    assert sql_accuracy()["accuracy"] >= 0.95


def test_numeral_guard():
    allowed = allowed_numbers({"results": {"q": {"rows": [[1234.5, "x"], [10, "y"]]}}, "comparisons": [(110, 100)]},
                              [{"text": "a 30% off sale"}])
    assert pct_change(110, 100) == 10.0
    assert find_unverified("Revenue $1,234.50, up 10.0%, 10 orders, a 30% sale, on 2026-10-19", allowed) == []
    assert find_unverified("Revenue grew 12.5%", allowed) == ["12.5%"]


def test_crew_end_to_end(tmp_path):
    from fde3_crewai_bigquery.crew import run_weekly_report

    r = run_weekly_report(out_dir=str(tmp_path))
    assert r["unverified_numbers"] == [] and r["queries"] == 12
    assert "# Weekly Business Report" in r["report"] and "[note-" in r["report"]
    assert all(os.path.exists(tmp_path / c) for c in r["charts"]) and len(r["charts"]) == 2


def test_guardrail_forces_rewrite_of_invented_number(tmp_path):
    from fde3_crewai_bigquery.crew import OfflineCrewLLM, run_weekly_report

    llm = OfflineCrewLLM(model="offline-crew", hallucinate_once=True)
    r = run_weekly_report(out_dir=str(tmp_path), llm=llm)
    assert llm.writer_drafts == 2 and "37.9%" not in r["report"] and r["unverified_numbers"] == []


def test_delivery_writes_email(tmp_path):
    from fde3_crewai_bigquery.deliver import deliver

    r = deliver(out_dir=str(tmp_path))
    assert os.path.exists(r["eml"]) and r["delivered_to"] == ["outbox"]
