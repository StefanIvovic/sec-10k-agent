"""Retrieval tools used by the agent: hybrid text search, XBRL financials, knowledge graph."""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Sequence

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from ..config import get_settings
from ..db import graph, postgres as pg
from ..ingest.edgar import METRICS
from ..llm import embed_query


# ------------------------------------------------------------------ text: LlamaIndex retriever over Postgres
class HybridPgRetriever(BaseRetriever):
    """LlamaIndex retriever: pg_search BM25 + pgvector, fused with reciprocal rank fusion in SQL."""

    def __init__(self, k: int = 8, tickers=None, years=None, items=None):
        super().__init__()
        self.k, self.tickers, self.years, self.items = k, tickers, years, items

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        q = query_bundle.query_str
        hits = pg.hybrid_search(q, embed_query(q), k=self.k, tickers=self.tickers, years=self.years, items=self.items)
        return [
            NodeWithScore(
                node=TextNode(id_=str(h.id), text=h.content, metadata={
                    "chunk_id": h.id, "ticker": h.ticker, "fiscal_year": h.fiscal_year, "item": h.item,
                    "item_title": h.item_title, "bm25_rank": h.bm25_rank, "vec_rank": h.vec_rank}),
                score=h.score,
            )
            for h in hits
        ]


def search_filings(queries: Sequence[str], tickers=None, years=None, items=None, k: int | None = None) -> list[dict]:
    """Run each query, merge results by chunk id. If an item filter finds nothing, retry without it."""
    k = k or get_settings().retrieve_k
    tickers = tickers or None
    found: dict[int, dict] = {}
    for q in queries:
        nodes = HybridPgRetriever(k, tickers, years or None, items or None).retrieve(q)
        if not nodes and items:
            nodes = HybridPgRetriever(k, tickers, years or None, None).retrieve(q)
        for n in nodes:
            m = n.node.metadata
            cur = found.get(m["chunk_id"])
            if cur is None or n.score > cur["score"]:
                found[m["chunk_id"]] = {**m, "content": n.node.get_content(), "score": n.score, "query": q}
    return sorted(found.values(), key=lambda h: h["score"], reverse=True)


# ------------------------------------------------------------------ structured financials
def get_financials(tickers: Sequence[str], metrics: Sequence[str], years: Sequence[int] | None = None) -> list[dict]:
    metrics = [m for m in metrics if m in METRICS]
    if not tickers or not metrics:
        return []
    rows = pg.get_facts(tickers, metrics, years)
    if not years:  # default: latest 3 fiscal years per ticker/metric
        latest: dict[tuple, list] = {}
        for r in rows:
            latest.setdefault((r["ticker"], r["metric"]), []).append(r)
        rows = [r for v in latest.values() for r in v[-3:]]
    # add year-over-year change
    prev: dict[tuple, dict] = {}
    for r in rows:
        p = prev.get((r["ticker"], r["metric"]))
        if p and p["fiscal_year"] == r["fiscal_year"] - 1 and p["value"]:
            r["yoy_change_pct"] = round((r["value"] - p["value"]) / abs(p["value"]) * 100, 2)
        prev[(r["ticker"], r["metric"])] = r
    return rows


def fmt_value(value: float, unit: str) -> str:
    if unit == "USD" and abs(value) >= 1e6:
        return f"${value / 1e9:,.2f} billion" if abs(value) >= 1e9 else f"${value / 1e6:,.1f} million"
    if unit == "USD/shares":
        return f"${value:,.2f} per share"
    return f"{value:,.0f} {unit}"


# ------------------------------------------------------------------ graph
def query_graph(cypher: str) -> list[dict]:
    return graph.run_read_cypher(cypher)


# ------------------------------------------------------------------ formatting sources for the LLM
def format_sources(hits: list[dict], facts: list[dict], graph_rows: list[dict],
                   text_budget_chars: int = 40000) -> tuple[str, dict]:
    """Returns (prompt text, citation map id -> metadata).

    Chunks are passed WHOLE (most are 2.5-4k chars; truncating hid their second half from the LLM).
    Instead, the highest-scoring chunks are added until text_budget_chars (~10k tokens) is reached,
    which keeps the prompt well inside qwen's ~31k context.
    """
    parts, cmap = [], {}
    used = 0
    kept = []
    for h in hits:  # hits arrive sorted by fused score
        if kept and used + len(h["content"]) > text_budget_chars:
            break
        kept.append(h)
        used += len(h["content"])
    for i, h in enumerate(kept, 1):
        sid = f"S{i}"
        cmap[sid] = {"type": "filing_text", "ticker": h["ticker"], "fiscal_year": h["fiscal_year"],
                     "item": h["item"], "item_title": h["item_title"], "chunk_id": h["chunk_id"],
                     "excerpt": h["content"][:300]}
        parts.append(f"[{sid}] {h['ticker']} 10-K FY{h['fiscal_year']} Item {h['item']} ({h['item_title']}):\n"
                     f"{h['content']}")
    for i, f in enumerate(facts, 1):
        fid = f"F{i}"
        yoy = f", YoY {f['yoy_change_pct']:+.2f}%" if "yoy_change_pct" in f else ""
        cmap[fid] = {"type": "xbrl", "ticker": f["ticker"], "metric": f["metric"], "concept": f["concept"],
                     "fiscal_year": f["fiscal_year"], "value": f["value"], "unit": f["unit"],
                     "accession": f["accession"]}
        parts.append(f"[{fid}] {f['ticker']} {f['metric']} FY{f['fiscal_year']} (period ending {f['period_end']}): "
                     f"{fmt_value(f['value'], f['unit'])} [raw {f['value']:,.2f} {f['unit']}; us-gaap:{f['concept']}]{yoy}")
    for i, g in enumerate(graph_rows, 1):
        gid = f"G{i}"
        cmap[gid] = {"type": "graph", "row": g}
        parts.append(f"[{gid}] {json.dumps(g, default=_json_default)}")
    return "\n\n".join(parts) if parts else "(no sources found)", cmap


def _json_default(o: Any):
    if isinstance(o, date):
        return o.isoformat()
    return str(o)


def normalize_plan(p, question: str, known_companies: str) -> dict:
    """Clean the planner's output: known tickers only, int years, bare item numbers, valid metrics."""
    known = set(re.findall(r"^([A-Z.\-]+):", known_companies, re.M))
    years = []
    for y in p.fiscal_years or []:
        try:
            years.append(int(str(y).upper().replace("FY", "").strip()))
        except ValueError:
            pass
    return {
        "tickers": [t.upper() for t in p.tickers or [] if t.upper() in known],
        "fiscal_years": years,
        "items": [str(i).upper().replace("ITEM", "").strip() for i in p.sections or []],
        "search_queries": (p.search_queries or [question])[:3],
        "metrics": [m for m in p.metrics or [] if m in METRICS],
        "needs_graph": bool(p.needs_graph),
    }


def describe_companies() -> str:
    rows = pg.list_companies()
    return "\n".join(f"{r['ticker']}: {r['name']} (FY {', '.join(map(str, r['years'] or []))})" for r in rows) or "(none)"


def describe_metrics() -> str:
    return ", ".join(METRICS)
