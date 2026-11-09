"""Eval questions: data/eval/dev.jsonl and data/eval/test.jsonl, one line per question.

dev   tune against it freely (its scores are optimistic)
test  held out: run it only to report a score. If a test failure makes you change code,
      move that question to dev and write a new one.

Line format (see data/eval/README.md):
  question  required
  expect    required: "answer" | "not_found" | "correction"
  ticker    company the question is about (needed for "source")
  number    expect=answer: this value must appear in the answer (within 1%, base units)
  contains  expect=answer: all of these must appear; expect=correction: any one of them (the correction)
  source    phrase(s) from the filing; also checks that the search found a paragraph containing one
  id, why, source_item   notes for you; not used for scoring
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("data/eval")
EXPECTS = {"answer", "not_found", "correction"}


def path(split: str) -> Path:
    return ROOT / f"{split}.jsonl"


def norm(q: str) -> str:
    return " ".join(q.lower().split()).rstrip("?")


def load(split: str) -> list[dict]:
    """Rows of a split, validated. Raises SystemExit listing every problem."""
    p = path(split)
    if not p.exists():
        raise SystemExit(f"{p} not found")
    rows, problems, seen = [], [], {}
    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{p}:{i}"
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append(f"{where} invalid JSON ({e})")
            continue
        problems += [f"{where} {msg}" for msg in _check_row(r)]
        if isinstance(r.get("question"), str):
            k = norm(r["question"])
            if k in seen:
                problems.append(f"{where} same question as line {seen[k]}")
            seen.setdefault(k, i)
        if isinstance(r.get("source"), str):
            r["source"] = [r["source"]]
        if isinstance(r.get("contains"), str):
            r["contains"] = [r["contains"]]
        r["line"] = i
        rows.append(r)
    if split == "test":
        dev = {norm(r["question"]) for r in load("dev")} if path("dev").exists() else set()
        problems += [f"{p}:{r['line']} also in dev.jsonl: {r['question']!r}" for r in rows
                     if isinstance(r.get("question"), str) and norm(r["question"]) in dev]
    if problems:
        raise SystemExit(f"{len(problems)} problem(s) in {p}:\n  " + "\n  ".join(problems))
    if not rows:
        raise SystemExit(f"{p} has no questions (format: data/eval/README.md)")
    return rows


def _check_row(r: dict) -> list[str]:
    out = []
    if not isinstance(r.get("question"), str) or not r["question"].strip():
        out.append("missing 'question'")
    e = r.get("expect")
    if e not in EXPECTS:
        out.append(f"'expect' must be one of {sorted(EXPECTS)}, got {e!r}")
    if "number" in r and (e != "answer" or not isinstance(r["number"], (int, float))):
        out.append("'number' must be a number, and only with expect=answer")
    if "contains" in r and not (isinstance(r["contains"], (str, list)) and r["contains"]):
        out.append("'contains' must be a phrase or a non-empty list of phrases")
    if e == "correction" and not r.get("contains"):
        out.append("expect=correction needs 'contains' (words that show the premise was corrected)")
    if e == "not_found" and ("number" in r or "contains" in r):
        out.append("expect=not_found takes no 'number' or 'contains'")
    if "source" in r:
        if not (isinstance(r["source"], (str, list)) and r["source"]):
            out.append("'source' must be a phrase or a non-empty list of phrases")
        if not r.get("ticker"):
            out.append("'source' needs 'ticker'")
    return out
