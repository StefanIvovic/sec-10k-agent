"""Retrieval eval: does the chunk that holds the answer get retrieved, and how high?

Gold = ticker + the 'source' phrase(s) of the questions in data/eval/<split>.jsonl, not a chunk id,
so the set survives re-ingestion. Any chunk of that ticker containing a gold phrase counts as a hit
(chunk overlap means a phrase can sit in two neighbouring chunks).

Modes
  raw       question as the query, ticker filter only        (retriever alone, no LLM)
  nofilter  question as the query, no filters                (does company context leak?)
  bm25/vector  each half of the hybrid on its own            (which side finds it)
  agent     the planner's queries + filters, exactly as the agent runs them (LLM calls),
            plus whether the gold chunk survives the answer step's prompt budget

python -m sec_agent.eval.retrieval_eval [--agent] [--k 8] [--split dev|test] [--run-name retrieval]
  Debugging tool: compares search methods. The normal eval (eval/run.py) already reports source_found_rate.
"""
from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd

from ..agent import tools
from ..config import get_settings
from ..db import postgres as pg
from ..llm import embed_query, ensure_embed_model_loaded
from ..tracing import init_mlflow
from . import cases

DEPTH = 50  # ranks beyond this count as "not found"


def load(split: str = "dev") -> list[dict]:
    """Questions with a 'source' phrase, as {question, ticker, gold, gold_item}."""
    return [{"question": r["question"], "ticker": r["ticker"], "gold": r["source"], "gold_item": r.get("source_item", "")}
            for r in cases.load(split) if r.get("source") and r["expect"] == "answer"]


def is_gold(ex: dict, ticker: str, content: str) -> bool:
    c = content.lower()
    return ticker == ex["ticker"] and any(g.lower() in c for g in ex["gold"])


def gold_rank(ex: dict, ranked: list[tuple[str, str]]) -> int | None:
    """1-based rank of the first gold chunk in [(ticker, content), ...]."""
    for i, (t, c) in enumerate(ranked, 1):
        if is_gold(ex, t, c):
            return i
    return None


def gold_chunks_exist(ex: dict) -> int:
    with pg.conn() as c:
        return sum(
            c.execute("SELECT count(*) AS n FROM chunks WHERE ticker=%s AND content ILIKE %s",
                      (ex["ticker"], f"%{g}%")).fetchone()["n"] for g in ex["gold"])


def bm25_ranked(q: str, ticker: str | None) -> list[tuple[str, str]]:
    where = "AND ticker = %(t)s" if ticker else ""
    with pg.conn() as c:
        rows = c.execute(f"""SELECT ticker, content FROM chunks WHERE content ||| %(q)s {where}
                             ORDER BY pdb.score(id) DESC LIMIT %(n)s""", {"q": q, "t": ticker, "n": DEPTH}).fetchall()
    return [(r["ticker"], r["content"]) for r in rows]


def vector_ranked(q: str, ticker: str | None) -> list[tuple[str, str]]:
    where = "WHERE ticker = %(t)s" if ticker else ""
    with pg.conn() as c:
        rows = c.execute(f"""SELECT ticker, content FROM chunks {where}
                             ORDER BY embedding <=> %(e)s LIMIT %(n)s""",
                         {"e": np.asarray(embed_query(q), dtype=np.float32), "t": ticker, "n": DEPTH}).fetchall()
    return [(r["ticker"], r["content"]) for r in rows]


def hybrid_ranked(q: str, ticker: str | None) -> list[tuple[str, str]]:
    hits = pg.hybrid_search(q, embed_query(q), k=DEPTH, tickers=[ticker] if ticker else None)
    return [(h.ticker, h.content) for h in hits]


def agent_ranked(ex: dict, program) -> tuple[list[tuple[str, str]], dict, int | None]:
    """Planner -> search_filings exactly as the graph's search_text node; also the prompt cut-off."""
    known = tools.describe_companies()
    p = program.analyze(question=ex["question"], available_companies=known, available_metrics=tools.describe_metrics())
    plan = tools.normalize_plan(p, ex["question"], known)
    hits = tools.search_filings(plan["search_queries"], plan["tickers"], plan["fiscal_years"], plan["items"])
    hits = hits[: get_settings().retrieve_k * 2]          # same cap as the search_text node
    _, cmap = tools.format_sources(hits, [], [])           # what the answer step actually sees
    in_prompt = sum(1 for k in cmap if k.startswith("S"))
    return [(h["ticker"], h["content"]) for h in hits], plan, in_prompt


def summarize(ranks: list[int | None], k: int) -> dict:
    found = [r for r in ranks if r is not None]
    return {
        "hit_at_1": sum(r == 1 for r in found) / len(ranks),
        "hit_at_3": sum(r <= 3 for r in found) / len(ranks),
        f"hit_at_{k}": sum(r <= k for r in found) / len(ranks),
        f"mrr_at_{k}": sum(1 / r for r in found if r <= k) / len(ranks),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=get_settings().retrieve_k)
    ap.add_argument("--agent", action="store_true", help="also run the planner (LLM) mode")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--run-name", default="retrieval")
    a = ap.parse_args()

    gold_file = cases.path(a.split)
    examples = load(a.split)
    ensure_embed_model_loaded()
    init_mlflow()
    missing = [e["question"] for e in examples if gold_chunks_exist(e) == 0]
    if missing:
        raise SystemExit(f"gold phrase not found in any chunk (re-ingested? typo? update {gold_file}):\n  "
                         + "\n  ".join(missing))

    program = None
    if a.agent:
        from ..agent.signatures import load_program
        from ..llm import configure_dspy
        configure_dspy()
        program = load_program()

    rows = []
    for ex in examples:
        q, t = ex["question"], ex["ticker"]
        r = {"question": q, "ticker": t, "gold_item": ex["gold_item"],
             "raw": gold_rank(ex, hybrid_ranked(q, t)),
             "nofilter": gold_rank(ex, hybrid_ranked(q, None)),
             "bm25": gold_rank(ex, bm25_ranked(q, t)),
             "vector": gold_rank(ex, vector_ranked(q, t))}
        if program is not None:
            ranked, plan, in_prompt = agent_ranked(ex, program)
            r["agent"] = gold_rank(ex, ranked)
            r["agent_in_prompt"] = r["agent"] is not None and r["agent"] <= in_prompt
            r["plan_items"] = ",".join(plan["items"])
            r["plan_tickers"] = ",".join(plan["tickers"])
            r["item_filter_excludes_gold"] = bool(plan["items"]) and bool(ex["gold_item"]) and ex["gold_item"] not in plan["items"]
        rows.append(r)
        print(f"{t:5} raw={r['raw']} nofilter={r['nofilter']} bm25={r['bm25']} vector={r['vector']}"
              + (f" agent={r['agent']} in_prompt={r['agent_in_prompt']} items={r['plan_items']}" if program else "")
              + f"  {q[:60]}")

    df = pd.DataFrame(rows)
    modes = ["raw", "nofilter", "bm25", "vector"] + (["agent"] if program else [])
    metrics = {}
    for m in modes:
        for name, v in summarize(df[m].tolist(), a.k).items():
            metrics[f"{m}_{name}"] = v
    if program:
        metrics["agent_in_prompt_rate"] = df["agent_in_prompt"].mean()
        metrics["item_filter_excludes_gold_rate"] = df["item_filter_excludes_gold"].mean()

    with mlflow.start_run(run_name=a.run_name, tags={"split": a.split}):
        mlflow.log_params({"split": a.split, "k": a.k, "n_questions": len(df), "embed_model": get_settings().embed_model,
                           "chunk_size": get_settings().chunk_size, "agent_mode": bool(program)})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_table(df.astype({c: "object" for c in modes}), "retrieval_results.json")

    print()
    table = pd.DataFrame({m: summarize(df[m].tolist(), a.k) for m in modes}).T
    print(table.round(3).to_string())
    if program:
        print(f"\nagent: gold chunk reaches the answer prompt in {metrics['agent_in_prompt_rate']:.0%} of questions; "
              f"planner's item filter excluded the gold item in {metrics['item_filter_excludes_gold_rate']:.0%}")


if __name__ == "__main__":
    main()
