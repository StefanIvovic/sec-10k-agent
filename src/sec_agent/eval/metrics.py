"""Scoring answers. Shared by the MLflow eval run and the DSPy optimizer."""
from __future__ import annotations

import re

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


ANSWERABLE_KINDS = {"numeric", "keywords", "open", "answerable"}

from ..agent.tools import says_not_in_sources as declines  # same detector finalize uses


def score(example: dict, answer: str, status: str = "answered") -> dict:
    """correct: 0-1 or None (not scored). refused: the user got the not-found message.

    unanswerable  pass = refused, or the answer says the filings don't contain it
    false_premise pass = refused, or the answer corrects the premise (e.g. says revenue increased)
    answerable    a refusal is a failure (false refusal) regardless of the content check
    """
    kind = example["kind"]
    refused = status == "not_found"
    if kind == "unanswerable":
        correct = 1.0 if (refused or declines(answer)) else 0.0
    elif kind == "false_premise":
        correct = 1.0 if (refused or any(k in answer.lower() for k in example["expected"])) else 0.0
    elif refused:
        correct = 0.0 if kind in ("numeric", "keywords") else None
    elif kind == "numeric":
        correct = numeric_match(answer, float(example["expected"]))
    elif kind == "keywords":
        correct = keyword_recall(answer, example["expected"])
    else:
        correct = None  # open / answerable: status-only
    return {"correct": correct, "cited": has_citation(answer), "refused": float(refused),
            "false_refusal": float(refused and kind in ANSWERABLE_KINDS)}
