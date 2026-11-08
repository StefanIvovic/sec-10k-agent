"""Build the evaluation set.

* numeric  – generated from XBRL facts in Postgres (gold = reported value)
* keywords – curated in data/eval/curated.jsonl (gold = phrases that must appear)
* open     – curated questions without gold, scored by the LLM judge (grounded + complete)

python -m sec_agent.eval.dataset   -> writes data/eval/questions.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

from ..db import postgres as pg

OUT = Path("data/eval/questions.jsonl")
CURATED = Path("data/eval/curated.jsonl")

NUMERIC_TEMPLATES = {
    "revenue": "What was {name}'s total revenue in fiscal year {fy}?",
    "net_income": "How much net income did {name} report for fiscal {fy}?",
    "research_and_development": "What did {name} spend on research and development in fiscal {fy}?",
    "eps_diluted": "What were {name}'s diluted earnings per share for fiscal {fy}?",
    "operating_cash_flow": "What was {name}'s cash flow from operations in fiscal {fy}?",
}


def numeric_examples() -> list[dict]:
    out = []
    with pg.conn() as c:
        companies = c.execute("SELECT ticker, name FROM companies").fetchall()
        for co in companies:
            fy = c.execute("SELECT max(fiscal_year) AS fy FROM filings f JOIN companies c USING (cik) WHERE c.ticker=%s",
                           (co["ticker"],)).fetchone()["fy"]
            for metric, tmpl in NUMERIC_TEMPLATES.items():
                row = c.execute("SELECT value::float AS v FROM financial_facts WHERE ticker=%s AND metric=%s AND fiscal_year=%s",
                                (co["ticker"], metric, fy)).fetchone()
                if row:
                    out.append({"question": tmpl.format(name=co["name"].title() if co["name"].isupper() else co["name"], fy=fy),
                                "kind": "numeric", "expected": row["v"], "ticker": co["ticker"], "metric": metric})
    return out


def build() -> list[dict]:
    rows = numeric_examples()
    if CURATED.exists():
        rows += [json.loads(l) for l in CURATED.read_text(encoding="utf-8").splitlines() if l.strip()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return rows


def load() -> list[dict]:
    return [json.loads(l) for l in OUT.read_text(encoding="utf-8").splitlines() if l.strip()]


if __name__ == "__main__":
    rows = build()
    print(f"wrote {len(rows)} examples to {OUT}: "
          f"{sum(r['kind'] == 'numeric' for r in rows)} numeric, "
          f"{sum(r['kind'] == 'keywords' for r in rows)} keywords, {sum(r['kind'] == 'open' for r in rows)} open")
