"""Natural-language question -> SQL for the analyst agent.

OfflineNL2SQL is a small semantic parser: it pulls out slots (metric, group-by dimension, filters, time window,
top-k) with keyword rules, then builds SQL from those slots. It only understands the business vocabulary
below, and that limit is deliberate: the metric definitions are agreed with the customer ("revenue = Complete +
Shipped orders") and the SQL always follows them. GeminiNL2SQL writes SQL with an LLM from the same schema and
metric definitions. Its output still goes through the same guardrails in warehouse.py.

Metric definitions (the semantic layer):
  revenue      SUM(quantity * sale_price) over orders with status Complete or Shipped
  orders       COUNT(DISTINCT order_id) over Complete / Shipped orders
  aov          revenue / orders
  units        SUM(quantity) over Complete / Shipped orders
  return_rate  % of all orders with status Returned
  new_users    number of users whose signup_date falls in the window
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from .data_gen import BRANDS, CATEGORIES, CHANNELS, COUNTRIES, START, WEEKS

MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}
VALID = "o.status IN ('Complete', 'Shipped')"
DIM_COL = {"category": "p.category", "country": "u.country", "channel": "u.channel", "brand": "p.brand",
           "week": "date_trunc('week', {date})", "status": "o.status"}


@dataclass
class Spec:
    metric: str
    group_by: str | None = None
    filters: dict = field(default_factory=dict)
    start: dt.date | None = None
    end: dt.date | None = None  # inclusive
    top_k: int | None = None
    ascending: bool = False


def week_range(n: int) -> tuple[dt.date, dt.date]:
    s = START + dt.timedelta(days=7 * (n - 1))
    return s, s + dt.timedelta(days=6)


class OfflineNL2SQL:
    name = "offline-parser"

    def parse(self, q: str) -> Spec:
        low = q.lower()
        if re.search(r"return rate|returned|returns", low):
            metric = "return_rate"
        elif re.search(r"new users|sign-?ups|signed up|new customers", low):
            metric = "new_users"
        elif re.search(r"average order value|\baov\b|average basket", low):
            metric = "aov"
        elif re.search(r"\bunits\b|items sold|quantity", low):
            metric = "units"
        elif re.search(r"revenue|sales|gmv|turnover|earn", low):
            metric = "revenue"
        elif re.search(r"orders|purchases", low):
            metric = "orders"
        else:
            raise ValueError(f"no known metric in: {q!r}")
        spec = Spec(metric)
        for dim, pat in (("category", r"categor"), ("country", r"countr|market|region"), ("channel", r"channel|source"),
                         ("brand", r"brand"), ("week", r"per week|by week|each week|weekly|week by week"),
                         ("status", r"by status|per status")):
            rank = r"(?:top|bottom|best|worst|lowest|highest) \d+"
            if re.search(rf"\b(?:by|per|each|across|which|{rank}|for every)\b.*(?:{pat})|(?:{pat}).*\b(?:breakdown|split)\b",
                         low) \
                    or (dim == "week" and re.search(pat, low)):
                spec.group_by = dim
                break
        for c in COUNTRIES:
            if re.search(rf"\b{c.lower()}\b", low) or (c == "United States" and re.search(r"\b(us|usa)\b", low)) \
                    or (c == "United Kingdom" and re.search(r"\b(uk|britain)\b", low)):
                spec.filters["country"] = c
        for c in CATEGORIES:
            if re.search(rf"\b{re.escape(c.lower())}\b", low):
                spec.filters["category"] = c
        for c in CHANNELS:
            if re.search(rf"\b{c.lower()}\b(?! breakdown)", low) and spec.group_by != "channel":
                spec.filters["channel"] = c
        for b in BRANDS:
            if re.search(rf"\b{b.lower()}\b", low):
                spec.filters["brand"] = b
        if m := re.search(r"\bweek (\d{1,2})\b", low):
            spec.start, spec.end = week_range(int(m.group(1)))
        elif re.search(r"last week|this week|latest week|most recent week", low):
            spec.start, spec.end = week_range(WEEKS)
        elif m := re.search(r"(\d{4}-\d{2}-\d{2}) (?:and|to) (\d{4}-\d{2}-\d{2})", low):
            spec.start, spec.end = dt.date.fromisoformat(m.group(1)), dt.date.fromisoformat(m.group(2))
        else:
            for name, num in MONTHS.items():
                if re.search(rf"\b{name}\b", low):
                    spec.start = dt.date(2026, num, 1)
                    spec.end = (dt.date(2026, num + 1, 1) if num < 12 else dt.date(2027, 1, 1)) - dt.timedelta(days=1)
        if m := re.search(r"\b(top|bottom|best|worst|lowest|highest) (\d+)\b", low):
            spec.top_k = int(m.group(2))
            spec.ascending = m.group(1) in ("bottom", "worst", "lowest")
        elif re.search(r"\bwhich\b.*\b(most|highest|best|top)\b|\b(best|top)-selling\b", low) and spec.group_by:
            spec.top_k = 1
        elif re.search(r"\bwhich\b.*\b(least|lowest|worst)\b", low) and spec.group_by:
            spec.top_k, spec.ascending = 1, True
        return spec

    def to_sql(self, s: Spec) -> str:
        if s.metric == "new_users":
            date, src, where = "u.signup_date", "shop.users u", []
        else:
            date = "o.created_at"
            needs_items = s.metric in ("revenue", "aov", "units") or "category" in s.filters or "brand" in s.filters \
                or s.group_by in ("category", "brand")
            needs_users = "country" in s.filters or "channel" in s.filters or s.group_by in ("country", "channel")
            src = "shop.orders o"
            if needs_items:
                src += " JOIN shop.order_items i ON i.order_id = o.order_id JOIN shop.products p ON p.product_id = i.product_id"
            if needs_users:
                src += " JOIN shop.users u ON u.user_id = o.user_id"
            where = [] if s.metric == "return_rate" else [VALID]
        expr = {"revenue": "ROUND(SUM(i.quantity * i.sale_price), 2)",
                "orders": "COUNT(DISTINCT o.order_id)",
                "aov": "ROUND(SUM(i.quantity * i.sale_price) / COUNT(DISTINCT o.order_id), 2)",
                "units": "SUM(i.quantity)",
                "return_rate": "ROUND(100.0 * COUNT(DISTINCT CASE WHEN o.status = 'Returned' THEN o.order_id END) "
                               "/ COUNT(DISTINCT o.order_id), 2)",
                "new_users": "COUNT(*)"}[s.metric]
        cols = {"category": "p.category", "brand": "p.brand", "country": "u.country", "channel": "u.channel"}
        for k, v in s.filters.items():
            where.append(f"{cols[k]} = '{v}'")
        if s.start:
            where.append(f"{date} BETWEEN DATE '{s.start}' AND DATE '{s.end}'")
        sql = "SELECT "
        if s.group_by:
            g = DIM_COL[s.group_by].format(date=date)
            sql += f"{g} AS {s.group_by}, "
        sql += f"{expr} AS {s.metric} FROM {src}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if s.group_by:
            sql += f" GROUP BY 1 ORDER BY {'1' if s.group_by == 'week' and not s.top_k else '2'}" \
                   f"{' DESC' if s.top_k and not s.ascending else ''}"
            if s.top_k:
                sql += f" LIMIT {s.top_k}"
        return sql

    def __call__(self, question: str) -> str:
        return self.to_sql(self.parse(question))


class GeminiNL2SQL:
    name = "gemini"

    def __init__(self, schema_text: str):
        from common.llm import GeminiLLM

        self.llm, self.schema = GeminiLLM(), schema_text

    def __call__(self, question: str) -> str:
        prompt = (f"Write one DuckDB/BigQuery-compatible SELECT for the question. Tables:\n{self.schema}\n"
                  f"Metric definitions:\n{__doc__.split('Metric definitions (the semantic layer):')[1]}\n"
                  f"Use aliases o (orders), i (order_items), p (products), u (users). Return only SQL.\n"
                  f"Question: {question}")
        return self.llm.generate(prompt).strip().removeprefix("```sql").removesuffix("```").strip()


# ---------------------------------------------------------------------------------------------------------
# 50-question test set. Gold answers are written as hand-made specs and compiled by gold_sql(), which uses a
# deliberately different SQL shape (CTEs, explicit date maths) from the parser's builder, so the two
# cross-check each other.
# ---------------------------------------------------------------------------------------------------------
def _w(n):
    return week_range(n)


TEST_SET = [
    ("What was total revenue last week?", dict(metric="revenue", w=12)),
    ("How many orders did we get in week 3?", dict(metric="orders", w=3)),
    ("What is the average order value in week 12?", dict(metric="aov", w=12)),
    ("Total units sold in September?", dict(metric="units", m=9)),
    ("What is the return rate in October?", dict(metric="return_rate", m=10)),
    ("How many new users signed up last week?", dict(metric="new_users", w=12)),
    ("Revenue by category last week", dict(metric="revenue", g="category", w=12)),
    ("Top 3 categories by revenue in week 11", dict(metric="revenue", g="category", w=11, k=3)),
    ("Top 5 countries by revenue last week", dict(metric="revenue", g="country", w=12, k=5)),
    ("Which country had the most orders in week 10?", dict(metric="orders", g="country", w=10, k=1)),
    ("Which category had the lowest revenue in August?", dict(metric="revenue", g="category", m=8, k=1, asc=True)),
    ("New users by channel in September", dict(metric="new_users", g="channel", m=9)),
    ("Revenue per week", dict(metric="revenue", g="week")),
    ("Weekly orders for India", dict(metric="orders", g="week", f={"country": "India"})),
    ("Outerwear revenue each week", dict(metric="revenue", g="week", f={"category": "Outerwear"})),
    ("What was Sweaters revenue in week 12?", dict(metric="revenue", w=12, f={"category": "Sweaters"})),
    ("How much revenue came from Germany last week?", dict(metric="revenue", w=12, f={"country": "Germany"})),
    ("Average order value in the UK in October", dict(metric="aov", m=10, f={"country": "United Kingdom"})),
    ("Orders from Social channel users last week", dict(metric="orders", w=12, f={"channel": "Social"})),
    ("Return rate by country in September", dict(metric="return_rate", g="country", m=9)),
    ("Units sold by brand in week 5", dict(metric="units", g="brand", w=5)),
    ("Top 2 brands by revenue in October", dict(metric="revenue", g="brand", m=10, k=2)),
    ("Bottom 3 categories by units sold last week", dict(metric="units", g="category", w=12, k=3, asc=True)),
    ("Revenue from Northwind products in September", dict(metric="revenue", m=9, f={"brand": "Northwind"})),
    ("How many Shoes units were sold in week 8?", dict(metric="units", w=8, f={"category": "Shoes"})),
    ("What is the AOV for Email channel customers?", dict(metric="aov", f={"channel": "Email"})),
    ("Total GMV between 2026-09-01 and 2026-09-15", dict(metric="revenue", d=("2026-09-01", "2026-09-15"))),
    ("Number of orders between 2026-08-10 and 2026-08-16", dict(metric="orders", d=("2026-08-10", "2026-08-16"))),
    ("Sales in the US last week", dict(metric="revenue", w=12, f={"country": "United States"})),
    ("Revenue by channel in week 9", dict(metric="revenue", g="channel", w=9)),
    ("Which channel brought the most new users in October?", dict(metric="new_users", g="channel", m=10, k=1)),
    ("Average order value by country last week", dict(metric="aov", g="country", w=12)),
    ("How many orders were placed in Japan in August?", dict(metric="orders", m=8, f={"country": "Japan"})),
    ("Activewear units per week", dict(metric="units", g="week", f={"category": "Activewear"})),
    ("Return rate last week", dict(metric="return_rate", w=12)),
    ("Top 4 countries by orders in September", dict(metric="orders", g="country", m=9, k=4)),
    ("Revenue of Dresses in Brazil in October", dict(metric="revenue", m=10, f={"category": "Dresses", "country": "Brazil"})),
    ("Weekly new users", dict(metric="new_users", g="week")),
    ("Total revenue", dict(metric="revenue")),
    ("Total orders", dict(metric="orders")),
    ("Units by category in week 1", dict(metric="units", g="category", w=1)),
    ("Which brand had the highest revenue last week?", dict(metric="revenue", g="brand", w=12, k=1)),
    ("Average order value per week", dict(metric="aov", g="week")),
    ("Orders by channel last week", dict(metric="orders", g="channel", w=12)),
    ("Return rate for Search channel users in September", dict(metric="return_rate", m=9, f={"channel": "Search"})),
    ("Revenue in Canada each week", dict(metric="revenue", g="week", f={"country": "Canada"})),
    ("Top 3 categories by units sold in October", dict(metric="units", g="category", m=10, k=3)),
    ("How many sign-ups came from Display in August?", dict(metric="new_users", m=8, f={"channel": "Display"})),
    ("Accessories revenue in Australia last week", dict(metric="revenue", w=12, f={"category": "Accessories", "country": "Australia"})),
    ("What is the turnover for Kestrel brand in week 7?", dict(metric="revenue", w=7, f={"brand": "Kestrel"})),
]


def gold_sql(spec: dict) -> str:
    m, g, f = spec["metric"], spec.get("g"), spec.get("f", {})
    if "w" in spec:
        a, b = _w(spec["w"])
    elif "m" in spec:
        a = dt.date(2026, spec["m"], 1)
        b = dt.date(2026, spec["m"] + 1, 1) - dt.timedelta(days=1)
    elif "d" in spec:
        a, b = (dt.date.fromisoformat(x) for x in spec["d"])
    else:
        a = b = None
    if m == "new_users":
        conds = [f"{k} = '{v}'" for k, v in f.items()]
        if a:
            conds.append(f"signup_date >= DATE '{a}' AND signup_date < DATE '{b}' + INTERVAL 1 DAY")
        key = {"channel": "channel", "country": "country", "week": "date_trunc('week', signup_date)"}.get(g)
        base = f"WITH base AS (SELECT * FROM shop.users {'WHERE ' + ' AND '.join(conds) if conds else ''})"
        val = "COUNT(*)"
    else:
        conds = []
        if a:
            conds.append(f"created_at >= DATE '{a}' AND created_at < DATE '{b}' + INTERVAL 1 DAY")
        conds += [f"{k} = '{v}'" for k, v in f.items()]
        if m != "return_rate":
            conds.append("status <> 'Returned' AND status <> 'Cancelled'")
        base = ("WITH base AS (SELECT o.order_id, o.status, o.created_at, i.quantity, i.sale_price, p.category, p.brand, "
                "u.country, u.channel FROM shop.orders o LEFT JOIN shop.order_items i USING (order_id) "
                "LEFT JOIN shop.products p USING (product_id) LEFT JOIN shop.users u USING (user_id)"
                f"{' WHERE ' + ' AND '.join(conds) if conds else ''})")
        key = {"week": "date_trunc('week', created_at)"}.get(g, g)
        val = {"revenue": "ROUND(SUM(quantity * sale_price), 2)", "orders": "COUNT(DISTINCT order_id)",
               "units": "SUM(quantity)", "aov": "ROUND(SUM(quantity * sale_price) / COUNT(DISTINCT order_id), 2)",
               "return_rate": "ROUND(100.0 * COUNT(DISTINCT order_id) FILTER (WHERE status = 'Returned') "
                              "/ COUNT(DISTINCT order_id), 2)"}[m]
    if not g:
        return f"{base} SELECT {val} FROM base"
    order = "v ASC" if spec.get("asc") else "v DESC"
    lim = f" LIMIT {spec['k']}" if spec.get("k") else ""
    return f"{base} SELECT {key} AS k, {val} AS v FROM base GROUP BY 1 ORDER BY {order if spec.get('k') else 'k'}{lim}"
