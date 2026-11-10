"""Prefect flows: EDGAR -> parse -> chunk/embed (LlamaIndex) -> Postgres, entities -> Neo4j, XBRL -> Postgres.

Run once:      python -m sec_agent.ingest.flows AAPL MSFT NVDA --fiscal-year 2026   (or --filings 2: latest two)
Serve:         python -m sec_agent.ingest.flows --serve      (deployment + weekly refresh schedule)
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from prefect import flow, task, get_run_logger
from prefect.cache_policies import NONE as NO_CACHE

from ..config import get_settings
from ..db import graph, postgres as pg
from ..db.graph import company_key
from ..llm import configure_dspy, embed_texts, ensure_embed_model_loaded
from . import edgar
from .extract import GraphExtractor
from .parse import Section, parse_10k

RAW_DIR = Path("data/raw")
SKIP_ITEMS = {"6", "16"}  # [Reserved], 10-K Summary


@task(retries=3, retry_delay_seconds=10, cache_policy=NO_CACHE)
def resolve_and_list(ticker: str, n_filings: int, fiscal_years: list[int] | None = None) -> tuple[dict, list[dict]]:
    company = edgar.resolve_company(ticker)
    pg.upsert_company(company["cik"], company["ticker"], company["name"])
    filings = edgar.list_10k_filings(company["cik"], limit=n_filings, fiscal_years=fiscal_years)
    for f in filings:
        pg.upsert_filing(f)
    return company, filings


@task(retries=3, retry_delay_seconds=10, cache_policy=NO_CACHE)
def download_and_parse(company: dict, filing: dict) -> list[Section]:
    path = RAW_DIR / company["ticker"] / f"{filing['accession']}.htm"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(edgar.download(filing["url"]))
    sections = [s for s in parse_10k(path.read_bytes()) if s.item not in SKIP_ITEMS]
    get_run_logger().info("%s FY%s: %d sections %s", company["ticker"], filing["fiscal_year"],
                          len(sections), [s.item for s in sections])
    return sections


@task(cache_policy=NO_CACHE)
def chunk_embed_store(company: dict, filing: dict, sections: list[Section]) -> int:
    s = get_settings()
    splitter = SentenceSplitter(chunk_size=s.chunk_size, chunk_overlap=s.chunk_overlap)
    docs = [
        Document(text=sec.text, metadata={"item": sec.item, "item_title": sec.title},
                 excluded_embed_metadata_keys=["item", "item_title"])
        for sec in sections
    ]
    nodes = splitter.get_nodes_from_documents(docs)
    counters: dict[str, int] = {}
    rows, embed_inputs = [], []
    for n in nodes:
        item = n.metadata["item"]
        counters[item] = counters.get(item, -1) + 1
        content = n.get_content()
        # a short header gives each chunk its company/period context in vector space
        header = f"{company['name']} ({company['ticker']}) 10-K FY{filing['fiscal_year']}, Item {item} {n.metadata['item_title']}"
        embed_inputs.append(f"{header}\n{content}")
        rows.append({"ticker": company["ticker"], "fiscal_year": filing["fiscal_year"], "item": item,
                     "item_title": n.metadata["item_title"], "chunk_index": counters[item], "content": content})
    for r, e in zip(rows, embed_texts(embed_inputs)):
        r["embedding"] = e
    n = pg.replace_chunks(filing["accession"], rows)
    get_run_logger().info("%s FY%s: stored %d chunks", company["ticker"], filing["fiscal_year"], n)
    return n


# retries also cover LM Studio unloading the chat model when another model is JIT-loaded
@task(retries=2, retry_delay_seconds=30, cache_policy=NO_CACHE)
def extract_and_write_graph(company: dict, filing: dict, sections: list[Section]) -> dict:
    configure_dspy()
    entities = GraphExtractor()(company=company["name"], sections=sections)
    graph.write_filing_graph({**company, "key": company_key(company["name"])}, filing, entities)
    summary = {k: len(v) for k, v in entities.items()}
    get_run_logger().info("%s FY%s graph: %s", company["ticker"], filing["fiscal_year"], summary)
    return summary


@task(retries=3, retry_delay_seconds=10, cache_policy=NO_CACHE)
def load_financial_facts(company: dict) -> int:
    rows = edgar.normalize_facts(company["ticker"], edgar.company_facts(company["cik"]))
    n = pg.replace_facts(company["ticker"], rows)
    get_run_logger().info("%s: %d XBRL facts", company["ticker"], n)
    return n


@flow(name="ingest-company", log_prints=True)
def ingest_company(ticker: str, n_filings: int = 2, force: bool = False, with_graph: bool = True,
                   fiscal_years: list[int] | None = None) -> dict:
    graph.ensure_constraints()
    company, filings = resolve_and_list(ticker.upper(), n_filings, fiscal_years)
    load_financial_facts(company)
    report = {"ticker": company["ticker"], "filings": []}
    for f in filings:
        need_chunks = force or not pg.filing_has_chunks(f["accession"])
        need_graph = with_graph and (force or not graph.filing_in_graph(f["accession"]))
        entry: dict = {"fiscal_year": f["fiscal_year"]}
        if not (need_chunks or need_graph):
            entry["status"] = "skipped (already ingested)"
        else:
            sections = download_and_parse(company, f)
            entry["chunks"] = chunk_embed_store(company, f, sections) if need_chunks else "kept"
            if need_graph:
                entry["graph"] = extract_and_write_graph(company, f, sections)
        report["filings"].append(entry)
    return report


@flow(name="ingest-companies", log_prints=True)
def ingest_companies(tickers: list[str], n_filings: int = 2, force: bool = False, with_graph: bool = True,
                     fiscal_years: list[int] | None = None) -> list[dict]:
    get_run_logger().info("embedding model: %s", ensure_embed_model_loaded())
    return [ingest_company(t, n_filings, force, with_graph, fiscal_years) for t in tickers]


DEFAULT_WATCHLIST = ["AAPL", "MSFT", "NVDA"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=DEFAULT_WATCHLIST)
    ap.add_argument("--filings", type=int, default=2, help="latest N 10-Ks per company")
    ap.add_argument("--fiscal-year", type=int, nargs="+", dest="fiscal_years",
                    help="only these fiscal years (overrides --filings), e.g. --fiscal-year 2026")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--serve", action="store_true", help="register deployment (weekly refresh) and serve it")
    a = ap.parse_args()
    if a.serve:
        ingest_companies.serve(
            name="sec-10k-ingest",
            cron="0 6 * * 1",  # Mondays 06:00, picks up newly filed 10-Ks
            parameters={"tickers": a.tickers, "n_filings": a.filings},
        )
    else:
        print(ingest_companies(a.tickers, a.filings, a.force, not a.no_graph, a.fiscal_years))
