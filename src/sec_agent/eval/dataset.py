"""Print number questions from the XBRL facts in Postgres, as eval lines you can copy into dev/test.jsonl.

python -m sec_agent.eval.dataset                 # latest fiscal year per company
python -m sec_agent.eval.dataset --year 2025     # a given fiscal year

Lines already in dev.jsonl or test.jsonl are skipped. Nothing is written: copy the lines you want.
"""
from __future__ import annotations

import argparse
import json

from ..db import postgres as pg
from . import cases

TEMPLATES = {
    "revenue": "What was {name}'s total revenue in fiscal year {fy}?",
    "net_income": "How much net income did {name} report for fiscal {fy}?",
    "research_and_development": "What did {name} spend on research and development in fiscal {fy}?",
    "eps_diluted": "What were {name}'s diluted earnings per share for fiscal {fy}?",
    "operating_cash_flow": "What was {name}'s cash flow from operations in fiscal {fy}?",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int)
    a = ap.parse_args()
    existing = {cases.norm(json.loads(l)["question"]) for split in ("dev", "test") if cases.path(split).exists()
                for l in cases.path(split).read_text(encoding="utf-8").splitlines() if l.strip()}
    with pg.conn() as c:
        for co in c.execute("SELECT ticker, name FROM companies ORDER BY ticker").fetchall():
            fy = a.year or c.execute("SELECT max(fiscal_year) AS fy FROM filings f JOIN companies c USING (cik) "
                                     "WHERE c.ticker=%s", (co["ticker"],)).fetchone()["fy"]
            name = co["name"].title() if co["name"].isupper() else co["name"]
            for metric, tmpl in TEMPLATES.items():
                row = c.execute("SELECT value::float AS v FROM financial_facts WHERE ticker=%s AND metric=%s "
                                "AND fiscal_year=%s", (co["ticker"], metric, fy)).fetchone()
                q = tmpl.format(name=name, fy=fy)
                if row and cases.norm(q) not in existing:
                    print(json.dumps({"question": q, "ticker": co["ticker"], "expect": "answer", "number": row["v"],
                                      "why": f"XBRL {metric} FY{fy}"}))


if __name__ == "__main__":
    main()
