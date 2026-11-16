"""Numeral guard: every number in the report must be traceable to data.

A number in the report is *verified* if, after removing $ , and %, it equals (at the precision it is shown with):
  * a value returned by one of the analyst's SQL queries, or
  * a week-over-week change computed from a declared (current, previous) pair of query values, or
  * a number quoted in a research note the report cites.
Dates (YYYY-MM-DD) are ignored. Anything else is an unverified number and the report is rejected. In CrewAI this
function is the writer Task's `guardrail`, so a rejected report goes back to the writer with the feedback.
"""
from __future__ import annotations

import re

NUM = re.compile(r"(?<![\w.])\$?(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)%?(?![\w])")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def pct_change(cur: float, prev: float) -> float:
    return round(100.0 * (cur - prev) / prev, 1) if prev else 0.0


def allowed_numbers(data: dict, notes: list[dict] | None = None) -> set[float]:
    vals: set[float] = set()
    for res in data.get("results", {}).values():
        for row in res.get("rows", []):
            for v in row:
                if isinstance(v, (int, float)):
                    vals.add(float(v))
    for cur, prev in data.get("comparisons", []):
        vals.add(pct_change(cur, prev))
    for n in notes or []:
        for m in NUM.finditer(DATE.sub(" ", n["text"])):
            vals.add(float(m.group(1).replace(",", "")))
    return vals


def find_unverified(report: str, allowed: set[float]) -> list[str]:
    bad = []
    text = DATE.sub(" ", report)
    text = re.sub(r"\[note-\d+\]|\]\([^)]*\)", " ", text)  # citation ids and link targets (chart file names)
    for m in NUM.finditer(text):
        raw = m.group(1).replace(",", "")
        x = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not any(abs(round(a, decimals) - x) < 1e-9 for a in allowed):
            bad.append(m.group(0))
    return bad


def make_guardrail(get_allowed):
    """Build a CrewAI task guardrail. get_allowed() is called at check time (the data exists only after the
    analyst and researcher tasks have run)."""

    def guardrail(output):
        text = output.raw if hasattr(output, "raw") else str(output)
        bad = find_unverified(text, get_allowed())
        if bad:
            return False, (f"These numbers are not in the query results or cited notes: {', '.join(bad[:10])}. "
                           f"Use only numbers from the data.")
        return True, text

    return guardrail
