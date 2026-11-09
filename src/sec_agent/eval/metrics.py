"""Pass/fail rules for one eval answer. Used by eval/run.py and the DSPy optimizer."""
from __future__ import annotations

import re

from ..agent.tools import says_not_in_sources  # same detector the agent's finalize step uses

_NUM = re.compile(r"\$?\s*(-?\d[\d,]*\.?\d*)\s*(trillion|billion|million|thousand|bn|b|m|k)?\b", re.I)
_SCALE = {"trillion": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}


def numbers_in(text: str) -> list[float]:
    out = []
    for num, scale in _NUM.findall(text.replace("−", "-")):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        out.append(v * _SCALE.get((scale or "").lower(), 1))
    return out


def numeric_match(answer: str, expected: float, rel_tol: float = 0.01) -> float:
    """1.0 if any number in the answer is within rel_tol of expected (any common scale)."""
    for v in numbers_in(answer):
        for cand in (v, v * 1e6, v * 1e9):  # "$416,161 million" style tables
            if expected and abs(cand - expected) / abs(expected) <= rel_tol:
                return 1.0
    return 0.0


def keyword_recall(answer: str, keywords: list[str]) -> float:
    a = answer.lower()
    return sum(k.lower() in a for k in keywords) / len(keywords) if keywords else 0.0


def has_citation(answer: str) -> float:
    return 1.0 if re.search(r"\[[SFG]\d+", answer) else 0.0


def check(row: dict, answer: str, status: str) -> tuple[bool, str]:
    """(passed, reason). status is the agent's: answered | partial | not_found."""
    expect, refused = row["expect"], status == "not_found"
    if expect == "not_found":
        if refused or says_not_in_sources(answer):
            return True, ""
        return False, "made up an answer (should say not found)"
    if expect == "correction":
        if refused or any(k.lower() in answer.lower() for k in row["contains"]):
            return True, ""
        return False, "accepted the false premise (no correction word: " + ", ".join(row["contains"]) + ")"
    # expect == "answer"
    if refused:
        return False, "said not found (should answer)"
    reasons = []
    if "number" in row and not numeric_match(answer, float(row["number"])):
        reasons.append(f"number {row['number']:,.2f} not in answer")
    if row.get("contains"):
        missing = [k for k in row["contains"] if k.lower() not in answer.lower()]
        if missing:
            reasons.append("missing: " + ", ".join(missing))
    return not reasons, "; ".join(reasons)


def source_found(row: dict, chunks: list[dict]) -> bool | None:
    """Did the answer step see a paragraph of row['ticker'] containing a source phrase? None if no source."""
    if not row.get("source"):
        return None
    return any(c["ticker"] == row["ticker"] and p.lower() in c["content"].lower()
               for c in chunks for p in row["source"])
