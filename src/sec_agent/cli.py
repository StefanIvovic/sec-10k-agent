"""Ask from the terminal:  python -m sec_agent.cli "What are Apple's main supply chain risks?" [--json]"""
import argparse
import json
import sys

from .agent.graph import ask
from .llm import ensure_embed_model_loaded
from .tracing import init_mlflow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--ticker", action="append", default=[])
    ap.add_argument("--year", action="append", type=int, default=[])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-trace", action="store_true")
    a = ap.parse_args()
    if not a.no_trace:
        init_mlflow()
    ensure_embed_model_loaded()
    res = ask(a.question, a.ticker, a.year)
    if a.json:
        json.dump(res, sys.stdout, indent=2, default=str)
        return
    print(f"[{res['status']}]")
    print(res["answer"], "\n")
    print("Sources:")
    for c in res["citations"]:
        if c["type"] == "filing_text":
            print(f"  [{c['id']}] {c['ticker']} FY{c['fiscal_year']} Item {c['item']} {c['item_title']}")
        elif c["type"] == "xbrl":
            print(f"  [{c['id']}] {c['ticker']} {c['metric']} FY{c['fiscal_year']} ({c['concept']}, {c['accession']})")
        else:
            print(f"  [{c['id']}] graph {c['row']}")
    print("\nSteps:", " -> ".join(f"{s['node']}({s['seconds']}s)" for s in res["steps"]))


if __name__ == "__main__":
    main()
