# FDE-3 · CrewAI Weekly Business Report on a BigQuery-style warehouse

An analyst agent writes guarded SQL. Every query is checked read-only, dry-run against a byte budget, and
row-limited before it runs. A researcher agent pulls market notes, and a writer agent produces a Markdown report
with charts. The writer task's CrewAI guardrail is a numeral guard: any number not found in the query results
sends the report back for a rewrite. DuckDB stands in for BigQuery; `BigQueryWarehouse` is the real adapter.

```bash
python -m fde3_crewai_bigquery.data_gen    # data/warehouse.duckdb (6k users, 9.3k orders, 18.5k items)
python -m fde3_crewai_bigquery.evaluate    # SQL accuracy, guardrails, % numbers verified
python -m fde3_crewai_bigquery.deliver     # outbox/<week>/report.md + charts + report.eml
```
Cloud Scheduler / Cloud Run Job commands are in `deliver.py`.
