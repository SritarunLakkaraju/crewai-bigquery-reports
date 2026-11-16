"""Guarded SQL access for the analyst agent. The agent never gets a raw database connection.

Every query passes four guardrails before it runs:
  1. read-only check  - one statement, must start with SELECT/WITH, no DDL/DML/admin keywords
  2. dry run          - estimate bytes scanned *before* running (BigQuery bills by bytes of referenced
                        columns; DuckDB has no such API, so we estimate it the same way: rows x column widths)
  3. byte budget      - reject queries over max_bytes (BigQuery: maximum_bytes_billed does the same server-side)
  4. row limit        - wrap the query so at most max_rows rows come back
Plus defence in depth: the DuckDB connection itself is opened read_only.

Backends: DuckDBWarehouse (local, default) and BigQueryWarehouse (FDE3_BACKEND=bigquery + GOOGLE_CLOUD_PROJECT).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import duckdb

from .data_gen import DB_PATH, generate

FORBIDDEN = re.compile(r"\b(insert|update|delete|merge|drop|create|alter|truncate|grant|revoke|attach|detach|copy|"
                       r"export|import|install|load|pragma|call|set|vacuum|checkpoint)\b", re.I)
TYPE_BYTES = {"INTEGER": 4, "BIGINT": 8, "DOUBLE": 8, "DATE": 4, "BOOLEAN": 1}


class QueryRejected(Exception):
    pass


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    bytes_estimated: int

    def as_dicts(self) -> list[dict]:
        return [dict(zip(self.columns, r)) for r in self.rows]


def check_read_only(sql: str) -> str:
    s = re.sub(r"--.*?$|/\*.*?\*/", " ", sql, flags=re.S | re.M).strip().rstrip(";").strip()
    if ";" in s:
        raise QueryRejected("only one statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise QueryRejected("only SELECT / WITH queries are allowed")
    # strip string literals before scanning keywords, so WHERE name = 'Drop Shoulder Tee' is fine
    if m := FORBIDDEN.search(re.sub(r"'(?:[^']|'')*'", "''", s)):
        raise QueryRejected(f"forbidden keyword: {m.group(0).upper()}")
    return s


class DuckDBWarehouse:
    name = "duckdb"

    def __init__(self, path: str = DB_PATH, max_bytes: int = 5_000_000, max_rows: int = 500):
        if not os.path.exists(path):
            generate(path)
        self.con = duckdb.connect(path, read_only=True)
        self.max_bytes, self.max_rows = max_bytes, max_rows
        self.tables: dict[str, dict] = {}
        for (t,) in self.con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='shop'").fetchall():
            cols = self.con.execute(f"SELECT column_name, data_type FROM information_schema.columns "
                                    f"WHERE table_schema='shop' AND table_name='{t}'").fetchall()
            n = self.con.execute(f"SELECT count(*) FROM shop.{t}").fetchone()[0]
            widths = {}
            for c, typ in cols:
                if typ == "VARCHAR":
                    widths[c] = int(self.con.execute(f"SELECT avg(strlen({c})) + 2 FROM shop.{t}").fetchone()[0] or 2)
                else:
                    widths[c] = TYPE_BYTES.get(typ, 8)
            self.tables[t] = {"rows": n, "widths": widths}

    def schema_text(self) -> str:
        return "\n".join(f"shop.{t}({', '.join(v['widths'])}) -- {v['rows']} rows" for t, v in self.tables.items())

    def dry_run(self, sql: str) -> int:
        """Estimated bytes processed: for each referenced table, rows x width of the referenced columns
        (SELECT * counts every column), which is how BigQuery's on-demand pricing works."""
        s = check_read_only(sql)
        low = s.lower()
        try:
            self.con.cursor().execute(f"EXPLAIN {s}")  # cursor(): own handle, safe when tools run in threads  # also validates syntax and names without running the query
        except duckdb.Error as e:
            raise QueryRejected(f"invalid SQL: {str(e).splitlines()[0]}") from e
        total = 0
        idents = set(re.findall(r"[a-z_][a-z0-9_]*", low))
        for t, meta in self.tables.items():
            if not re.search(rf"\bshop\.{t}\b", low):
                continue
            cols = list(meta["widths"]) if re.search(r"select\s+\*|\.\*", low) else [c for c in meta["widths"] if c in idents]
            total += meta["rows"] * sum(meta["widths"][c] for c in (cols or list(meta["widths"])[:1]))
        return total

    def run(self, sql: str) -> QueryResult:
        s = check_read_only(sql)
        est = self.dry_run(s)
        if est > self.max_bytes:
            raise QueryRejected(f"query would scan ~{est:,} bytes, over the {self.max_bytes:,} byte budget")
        # CrewAI may execute several tool calls in parallel threads; a DuckDB connection is not
        # thread-safe, so every query gets its own cursor (a child connection to the same database).
        cur = self.con.cursor().execute(f"SELECT * FROM ({s}) AS q LIMIT {self.max_rows}")
        cols = [d[0] for d in cur.description]
        rows = [tuple(round(v, 4) if isinstance(v, float) else v for v in r) for r in cur.fetchall()]
        return QueryResult(s, cols, rows, est)


class BigQueryWarehouse:
    """Same interface on BigQuery: real dry runs and maximum_bytes_billed. Not exercised by offline tests."""

    name = "bigquery"

    def __init__(self, project: str | None = None, dataset: str = "shop", max_bytes: int = 1_000_000_000,
                 max_rows: int = 500):
        from google.cloud import bigquery

        self.bq = bigquery
        self.client = bigquery.Client(project=project or os.environ["GOOGLE_CLOUD_PROJECT"])
        self.default_dataset = f"{self.client.project}.{dataset}"
        self.max_bytes, self.max_rows = max_bytes, max_rows

    def schema_text(self) -> str:
        out = []
        for t in self.client.list_tables(self.default_dataset):
            tbl = self.client.get_table(t)
            out.append(f"shop.{t.table_id}({', '.join(f.name for f in tbl.schema)}) -- {tbl.num_rows} rows")
        return "\n".join(out)

    def dry_run(self, sql: str) -> int:
        s = check_read_only(sql)
        cfg = self.bq.QueryJobConfig(dry_run=True, use_query_cache=False, default_dataset=self.default_dataset)
        return self.client.query(s, job_config=cfg).total_bytes_processed

    def run(self, sql: str) -> QueryResult:
        s = check_read_only(sql)
        est = self.dry_run(s)
        if est > self.max_bytes:
            raise QueryRejected(f"query would scan ~{est:,} bytes, over the {self.max_bytes:,} byte budget")
        cfg = self.bq.QueryJobConfig(maximum_bytes_billed=self.max_bytes, default_dataset=self.default_dataset)
        res = self.client.query(f"SELECT * FROM ({s}) LIMIT {self.max_rows}", job_config=cfg).result()
        cols = [f.name for f in res.schema]
        return QueryResult(s, cols, [tuple(r.values()) for r in res], est)


def get_warehouse(**kw):
    if os.environ.get("FDE3_BACKEND") == "bigquery":
        return BigQueryWarehouse(**kw)
    return DuckDBWarehouse(**kw)
