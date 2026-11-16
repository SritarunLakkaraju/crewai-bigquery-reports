"""FDE-3: a CrewAI crew that analyses the warehouse and writes a weekly business report.

Agents (CrewAI Agent / Task / Crew, sequential process):
  analyst    - tool run_sql (guarded: read-only, dry-run byte budget, row limit). Answers a fixed list of
               business questions; returns the results as JSON.
  researcher - tool search_market_notes. Finds external context for the biggest movers.
  writer     - tool render_chart. Writes the Markdown report. Its Task has a guardrail (the numeral guard):
               any number not traceable to query results / cited notes sends the report back for a rewrite.

The LLM behind the agents:
  * GEMINI_API_KEY set -> crewai.LLM("gemini/<model>")
  * otherwise          -> OfflineCrewLLM, a crewai BaseLLM that plays each role deterministically and
                          speaks CrewAI's native tool-calling protocol. CrewAI's own executor runs the
                          tools, passes context between tasks and enforces the guardrail.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import re

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from crewai import Agent, BaseLLM, Crew, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

from common.llm import gemini_available  # noqa: E402
from common.text import tokenize  # noqa: E402

from .data_gen import WEEKS  # noqa: E402
from .guard import allowed_numbers, find_unverified, make_guardrail, pct_change  # noqa: E402
from .nl2sql import OfflineNL2SQL, week_range  # noqa: E402
from .warehouse import QueryRejected, get_warehouse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

QUESTIONS = [
    ("rev_cur", "What was total revenue last week?"),
    ("rev_prev", f"What was total revenue in week {WEEKS - 1}?"),
    ("ord_cur", "How many orders did we get last week?"),
    ("ord_prev", f"How many orders did we get in week {WEEKS - 1}?"),
    ("aov_cur", "What is the average order value last week?"),
    ("aov_prev", f"What is the average order value in week {WEEKS - 1}?"),
    ("cat_cur", "Revenue by category last week"),
    ("cat_prev", f"Revenue by category in week {WEEKS - 1}"),
    ("country_cur", "Top 5 countries by revenue last week"),
    ("ret_cur", "Return rate last week"),
    ("users_cur", "New users by channel last week"),
    ("weekly", "Revenue per week"),
]
COMPARISONS = [("rev_cur", "rev_prev"), ("ord_cur", "ord_prev"), ("aov_cur", "aov_prev"), ("cat_cur", "cat_prev")]

MARKET_NOTES = [
    {"id": "note-1", "date": "2026-10-20", "topic": "weather europe outerwear sweaters",
     "text": "An early cold snap across Northern Europe pushed searches for coats and knitwear up sharply."},
    {"id": "note-2", "date": "2026-10-18", "topic": "competitor activewear discount",
     "text": "A large competitor started a 30% off sale on activewear, likely pulling demand away."},
    {"id": "note-3", "date": "2026-10-21", "topic": "india festival season shopping",
     "text": "The festival season in India has started, which usually lifts apparel and gifting purchases."},
    {"id": "note-4", "date": "2026-10-15", "topic": "united states holiday early deals",
     "text": "US retailers began early holiday promotions, raising price sensitivity among shoppers."},
    {"id": "note-5", "date": "2026-10-12", "topic": "shoes supply chain delays",
     "text": "Port congestion is delaying footwear shipments by up to 2 weeks for some brands."},
    {"id": "note-6", "date": "2026-10-22", "topic": "social media fashion trend sweaters outerwear",
     "text": "Oversized sweaters and quilted jackets are trending on social media this autumn."},
    {"id": "note-7", "date": "2026-10-09", "topic": "uk consumer confidence",
     "text": "UK consumer confidence fell slightly in October, with households cutting discretionary spend."},
    {"id": "note-8", "date": "2026-10-19", "topic": "germany retail outerwear demand",
     "text": "German retailers report strong early demand for outerwear as temperatures dropped."},
]


class ReportContext:
    """Trusted state shared by the tools. The guard reads query results from here, not from LLM text."""

    def __init__(self, warehouse=None, out_dir: str | None = None):
        self.wh = warehouse or get_warehouse()
        self.results: dict[str, dict] = {}  # sql -> {columns, rows, bytes}
        self.notes: dict[str, dict] = {}
        self.charts: list[str] = []
        self.qid_sql: dict[str, str] = {}
        self.out_dir = out_dir or os.path.join(HERE, "outbox", week_range(WEEKS)[0].isoformat())
        os.makedirs(self.out_dir, exist_ok=True)

    def trusted_data(self) -> dict:
        res = {sql: r for sql, r in self.results.items()}
        comps = []
        for a, b in COMPARISONS:
            ra, rb = self.results.get(self.qid_sql.get(a, "")), self.results.get(self.qid_sql.get(b, ""))
            if not ra or not rb:
                continue
            if len(ra["columns"]) == 1:
                comps.append((ra["rows"][0][0], rb["rows"][0][0]))
            else:
                prev = {r[0]: r[1] for r in rb["rows"]}
                comps += [(r[1], prev[r[0]]) for r in ra["rows"] if r[0] in prev]
        return {"results": res, "comparisons": comps}

    def allowed(self) -> set[float]:
        return allowed_numbers(self.trusted_data(), list(self.notes.values()))


def make_tools(ctx: ReportContext):
    @tool("run_sql")
    def run_sql(sql: str) -> str:
        """Run one read-only SQL query on the `shop` dataset (tables shop.orders, shop.order_items, shop.products,
        shop.users). The query is dry-run first and rejected if it scans too many bytes. Returns JSON rows."""
        try:
            r = ctx.wh.run(sql)
        except QueryRejected as e:
            return json.dumps({"error": str(e)})
        rows = [[v.isoformat() if isinstance(v, (dt.date, dt.datetime)) else v for v in row] for row in r.rows]
        ctx.results[r.sql] = {"columns": r.columns, "rows": rows, "bytes": r.bytes_estimated}
        return json.dumps({"sql": r.sql, "columns": r.columns, "rows": rows, "bytes_estimated": r.bytes_estimated})

    @tool("search_market_notes")
    def search_market_notes(query: str) -> str:
        """Search recent market and news notes. Returns up to 2 notes with id, date and text."""
        q = set(tokenize(query))
        scored = sorted(MARKET_NOTES, key=lambda n: -len(q & set(tokenize(n["topic"] + " " + n["text"]))))
        hits = [n for n in scored[:2] if q & set(tokenize(n["topic"] + " " + n["text"]))]
        for n in hits:
            ctx.notes[n["id"]] = n
        return json.dumps(hits)

    @tool("render_chart")
    def render_chart(kind: str) -> str:
        """Render a chart PNG for the report. kind is 'weekly_revenue' or 'category_revenue'. Returns the file name."""
        if kind == "weekly_revenue":
            r = ctx.results.get(ctx.qid_sql.get("weekly", ""))
            if not r:
                return json.dumps({"error": "run the weekly revenue query first"})
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot([str(x[0])[5:10] for x in r["rows"]], [x[1] for x in r["rows"]], marker="o", color="#2563eb")
            ax.set_title("Revenue per week")
            ax.set_ylabel("USD")
            ax.tick_params(axis="x", rotation=45, labelsize=8)
        elif kind == "category_revenue":
            cur, prev = (ctx.results.get(ctx.qid_sql.get(k, "")) for k in ("cat_cur", "cat_prev"))
            if not cur or not prev:
                return json.dumps({"error": "run the category queries first"})
            p = {x[0]: x[1] for x in prev["rows"]}
            cats = [x[0] for x in sorted(cur["rows"], key=lambda x: -x[1])]
            c = {x[0]: x[1] for x in cur["rows"]}
            fig, ax = plt.subplots(figsize=(7, 3.2))
            xs = range(len(cats))
            ax.bar([i - 0.2 for i in xs], [p.get(k, 0) for k in cats], 0.4, label="previous week", color="#94a3b8")
            ax.bar([i + 0.2 for i in xs], [c[k] for k in cats], 0.4, label="last week", color="#2563eb")
            ax.set_xticks(list(xs), cats, rotation=30, ha="right", fontsize=8)
            ax.set_title("Revenue by category")
            ax.legend(fontsize=8)
        else:
            return json.dumps({"error": f"unknown chart kind {kind!r}"})
        fig.tight_layout()
        name = f"{kind}.png"
        fig.savefig(os.path.join(ctx.out_dir, name), dpi=120)
        plt.close(fig)
        ctx.charts.append(name)
        return json.dumps({"file": name})

    return run_sql, search_market_notes, render_chart


# ---------------------------------------------------------------------------------------------------------
# Offline LLM: plays analyst / researcher / writer, using CrewAI's native tool-call protocol
# ---------------------------------------------------------------------------------------------------------
def _call(i, name, args):
    return {"id": f"call_{name}_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def _extract_data(text: str) -> dict:
    m = re.search(r"<<DATA>>(.*?)<<END>>", text, re.S)
    return json.loads(m.group(1)) if m else {}


def _money(x):
    return f"${x:,.2f}"


def _chg(cur, prev):
    v = pct_change(cur, prev)
    return f"{'+' if v >= 0 else ''}{v}%"


class OfflineCrewLLM(BaseLLM):
    hallucinate_once: bool = False  # test hook: the writer's first draft contains an invented number
    writer_drafts: int = 0

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 128_000

    def call(self, messages, tools=None, callbacks=None, available_functions=None, from_task=None,
             from_agent=None, response_model=None):
        msgs = messages if isinstance(messages, list) else [{"role": "user", "content": messages}]
        role = (getattr(from_agent, "role", "") or "").lower()
        user = "\n".join(str(m.get("content")) for m in msgs if m["role"] == "user")
        tool_msgs = [m for m in msgs if m["role"] == "tool"]
        if "analyst" in role:
            return self._analyst(user, tool_msgs, tools)
        if "research" in role:
            return self._researcher(user, tool_msgs, tools)
        return self._writer(user, tool_msgs, tools)

    def _analyst(self, user, tool_msgs, tools):
        qs = re.findall(r"- \[(\w+)\] (.+)", user)
        if not tool_msgs and tools:
            parser = OfflineNL2SQL()
            return [_call(i, "run_sql", {"sql": parser(q)}) for i, (_, q) in enumerate(qs)]
        by_id = {m.get("tool_call_id"): json.loads(m["content"]) for m in tool_msgs}
        answers = {}
        for i, (qid, q) in enumerate(qs):
            r = by_id.get(f"call_run_sql_{i}", {})
            answers[qid] = {"question": q, "sql": r.get("sql"), "columns": r.get("columns"), "rows": r.get("rows"),
                            "error": r.get("error")}
        return "<<DATA>>" + json.dumps({"answers": answers, "comparisons": COMPARISONS}) + "<<END>>"

    def _researcher(self, user, tool_msgs, tools):
        data = _extract_data(user).get("answers", {})
        if not tool_msgs and tools:
            prev = {r[0]: r[1] for r in data.get("cat_prev", {}).get("rows") or []}
            growth = sorted(((pct_change(r[1], prev[r[0]]), r[0]) for r in data.get("cat_cur", {}).get("rows") or []
                             if r[0] in prev), reverse=True)
            top_country = (data.get("country_cur", {}).get("rows") or [["India"]])[0][0]
            queries = [f"{growth[0][1]} demand", f"{growth[-1][1]} demand", f"{top_country} market"] if growth else []
            return [_call(i, "search_market_notes", {"query": q}) for i, q in enumerate(queries)]
        seen, lines = set(), []
        for m in tool_msgs:
            for n in json.loads(m["content"]):
                if n["id"] not in seen:
                    seen.add(n["id"])
                    lines.append(f"- [{n['id']}] ({n['date']}) {n['text']}")
        return "\n".join(lines) or "- No relevant market notes found."

    def _writer(self, user, tool_msgs, tools):
        # Guardrail retry: CrewAI sends the failed draft plus the validation error. Revise the draft by
        # dropping every line that contains a flagged number (what we would instruct a real LLM to do).
        if "### Previous result:" in user:
            flagged = re.search(r"failed validation: .*?: (.+?)\. Use only", user, re.S)
            bad = [b.strip() for b in flagged.group(1).split(",")] if flagged else []
            draft = user.split("### Previous result:", 1)[1].split("Try again, making sure", 1)[0].strip()
            self.writer_drafts += 1
            return "\n".join(line for line in draft.splitlines() if not any(b and b in line for b in bad))
        if not tool_msgs and tools:
            return [_call(0, "render_chart", {"kind": "weekly_revenue"}), _call(1, "render_chart", {"kind": "category_revenue"})]
        a = _extract_data(user).get("answers", {})
        notes = re.findall(r"- \[(note-\d+)\] \((\d{4}-\d{2}-\d{2})\) (.+)", user)
        charts = [json.loads(m["content"]).get("file") for m in tool_msgs]
        one = lambda k: (a.get(k, {}).get("rows") or [[0]])[0][0]  # noqa: E731
        start, end = week_range(WEEKS)
        rev, rev_p, orders, orders_p, aov, aov_p = (one(k) for k in ("rev_cur", "rev_prev", "ord_cur", "ord_prev",
                                                                      "aov_cur", "aov_prev"))
        prev_cat = {r[0]: r[1] for r in a.get("cat_prev", {}).get("rows") or []}
        out = [f"# Weekly Business Report: {start} to {end}", "", "## Headline",
               f"- Revenue: {_money(rev)} ({_chg(rev, rev_p)} vs the previous week)",
               f"- Orders: {orders:,} ({_chg(orders, orders_p)})",
               f"- Average order value: {_money(aov)} ({_chg(aov, aov_p)})",
               f"- Return rate: {one('ret_cur')}% of orders", "",
               "## Revenue by category", "| Category | Revenue | Change vs previous week |", "|---|---|---|"]
        for cat, v in sorted((tuple(r) for r in a.get("cat_cur", {}).get("rows") or []), key=lambda r: -r[1]):
            out.append(f"| {cat} | {_money(v)} | {_chg(v, prev_cat[cat]) if cat in prev_cat else 'new'} |")
        out += ["", "## Top countries by revenue", "| Country | Revenue |", "|---|---|"]
        out += [f"| {c} | {_money(v)} |" for c, v in a.get("country_cur", {}).get("rows") or []]
        out += ["", "## New users by channel"]
        out += [f"- {c}: {n:,}" for c, n in sorted((tuple(r) for r in a.get("users_cur", {}).get("rows") or []),
                                                     key=lambda r: -r[1])]
        out += ["", "## Market context"] + ([f"- {t} [{i}]" for i, _, t in notes] or ["- None found."])
        if self.hallucinate_once and self.writer_drafts == 0:
            out.append("- Analysts expect revenue to grow 37.9% next week.")  # invented number: the guard must catch it
        self.writer_drafts += 1
        out += ["", "## Charts"] + [f"![{c.split('.')[0].replace('_', ' ')}]({c})" for c in charts if c]
        return "\n".join(out)


def build_crew(ctx: ReportContext, llm=None):
    if llm is None:
        if gemini_available():
            from crewai import LLM

            llm = LLM(model=f"gemini/{os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')}", temperature=0)
        else:
            llm = OfflineCrewLLM(model="offline-crew")
    run_sql, search_notes, render_chart = make_tools(ctx)
    parser = OfflineNL2SQL()
    ctx.qid_sql = {qid: parser(q) for qid, q in QUESTIONS}  # the SQL each question maps to (used by trusted_data)

    analyst = Agent(role="Data Analyst", goal="Answer business questions accurately with read-only SQL",
                    backstory="You know the shop warehouse schema and never guess a number.", llm=llm,
                    tools=[run_sql], allow_delegation=False, verbose=False, max_iter=6)
    researcher = Agent(role="Market Researcher", goal="Find external context that explains the week's biggest moves",
                       backstory="You read market notes and cite them by id.", llm=llm, tools=[search_notes],
                       allow_delegation=False, verbose=False, max_iter=4)
    writer = Agent(role="Report Writer", goal="Write a clear weekly business report for the leadership team",
                   backstory="You only use numbers that appear in the data you were given.", llm=llm,
                   tools=[render_chart], allow_delegation=False, verbose=False, max_iter=4)

    qlist = "\n".join(f"- [{qid}] {q}" for qid, q in QUESTIONS)
    t1 = Task(description=f"Answer each question with one run_sql call. Schema:\n{ctx.wh.schema_text()}\n"
                          f"Questions:\n{qlist}",
              expected_output="<<DATA>>{\"answers\": {qid: {question, sql, columns, rows}}, \"comparisons\": [...]}<<END>>",
              agent=analyst)
    t2 = Task(description="Using the analyst's data, search market notes for the fastest-growing category, the "
                          "fastest-falling category and the top country. List each note as '- [id] (date) text'.",
              expected_output="A bullet list of cited notes.", agent=researcher, context=[t1])
    t3 = Task(description="Write the weekly report in Markdown: headline metrics with week-over-week change, a "
                          "category table, top countries, new users by channel, market context with [note-id] "
                          "citations, and the two charts (call render_chart for 'weekly_revenue' and "
                          "'category_revenue'). Use only numbers from the data.",
              expected_output="A Markdown report.", agent=writer, context=[t1, t2],
              guardrail=make_guardrail(ctx.allowed), guardrail_max_retries=2)
    return Crew(agents=[analyst, researcher, writer], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)


def run_weekly_report(out_dir: str | None = None, llm=None) -> dict:
    ctx = ReportContext(out_dir=out_dir)
    crew = build_crew(ctx, llm)
    result = crew.kickoff()
    report = result.raw
    with open(os.path.join(ctx.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report)
    bad = find_unverified(report, ctx.allowed())
    return {"report": report, "out_dir": ctx.out_dir, "charts": ctx.charts, "queries": len(ctx.results),
            "bytes_scanned": sum(r["bytes"] for r in ctx.results.values()), "unverified_numbers": bad,
            "notes_cited": sorted(ctx.notes), "tasks": [t.raw[:80] for t in result.tasks_output]}


if __name__ == "__main__":
    out = run_weekly_report()
    print(out["report"])
    print({k: v for k, v in out.items() if k not in ("report", "tasks")})
