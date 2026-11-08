"""PostgreSQL access: filings metadata, chunk storage, hybrid (BM25 + vector) search, XBRL facts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from ..config import get_settings


@lru_cache
def get_pool() -> ConnectionPool:
    return ConnectionPool(
        get_settings().postgres_dsn,
        min_size=1,
        max_size=8,
        kwargs={"row_factory": dict_row, "autocommit": True},
        configure=register_vector,
        open=True,
    )


@contextmanager
def conn():
    with get_pool().connection() as c:
        yield c


# ---------------------------------------------------------------- companies / filings
def upsert_company(cik: str, ticker: str, name: str) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO companies (cik, ticker, name) VALUES (%s, %s, %s)
               ON CONFLICT (cik) DO UPDATE SET ticker = EXCLUDED.ticker, name = EXCLUDED.name""",
            (cik, ticker, name),
        )


def upsert_filing(f: dict) -> None:
    with conn() as c:
        c.execute(
            """INSERT INTO filings (accession, cik, form, fiscal_year, period_of_report, filed_at, url)
               VALUES (%(accession)s, %(cik)s, %(form)s, %(fiscal_year)s, %(period_of_report)s, %(filed_at)s, %(url)s)
               ON CONFLICT (accession) DO UPDATE SET fiscal_year = EXCLUDED.fiscal_year, url = EXCLUDED.url""",
            f,
        )


def filing_has_chunks(accession: str) -> bool:
    with conn() as c:
        return c.execute("SELECT 1 FROM chunks WHERE accession = %s LIMIT 1", (accession,)).fetchone() is not None


def list_companies() -> list[dict]:
    with conn() as c:
        return c.execute(
            """SELECT co.ticker, co.name, array_agg(DISTINCT f.fiscal_year ORDER BY f.fiscal_year) AS years
               FROM companies co LEFT JOIN filings f ON f.cik = co.cik
               GROUP BY co.ticker, co.name ORDER BY co.ticker"""
        ).fetchall()


# ---------------------------------------------------------------- chunks
def replace_chunks(accession: str, rows: Sequence[dict]) -> int:
    """rows: ticker, fiscal_year, item, item_title, chunk_index, content, embedding"""
    with conn() as c, c.transaction():
        c.execute("DELETE FROM chunks WHERE accession = %s", (accession,))
        with c.cursor() as cur:
            cur.executemany(
                """INSERT INTO chunks (accession, ticker, fiscal_year, item, item_title, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    (accession, r["ticker"], r["fiscal_year"], r["item"], r["item_title"],
                     r["chunk_index"], r["content"], np.asarray(r["embedding"], dtype=np.float32))
                    for r in rows
                ],
            )
    return len(rows)


@dataclass
class Hit:
    id: int
    ticker: str
    fiscal_year: int
    item: str
    item_title: str
    content: str
    score: float
    bm25_rank: int | None
    vec_rank: int | None


def _filters(tickers, years, items) -> tuple[str, dict]:
    clauses, params = [], {}
    if tickers:
        clauses.append("ticker = ANY(%(tickers)s)")
        params["tickers"] = list(tickers)
    if years:
        clauses.append("fiscal_year = ANY(%(years)s)")
        params["years"] = list(years)
    if items:
        clauses.append("item = ANY(%(items)s)")
        params["items"] = list(items)
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


def hybrid_search(
    query: str,
    embedding: list[float],
    k: int = 8,
    tickers: Iterable[str] | None = None,
    years: Iterable[int] | None = None,
    items: Iterable[str] | None = None,
    candidates: int = 40,
    rrf_k: int = 60,
) -> list[Hit]:
    """Reciprocal-rank fusion of pg_search BM25 and pgvector cosine search, with metadata filters."""
    where, params = _filters(tickers, years, items)
    params |= {"q": query, "emb": np.asarray(embedding, dtype=np.float32), "n": candidates, "k": k, "rrf": rrf_k}
    sql = f"""
    WITH bm25 AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY pdb.score(id) DESC) AS r
        FROM chunks WHERE content ||| %(q)s {where}
        ORDER BY pdb.score(id) DESC LIMIT %(n)s
    ),
    vec AS (
        SELECT id, ROW_NUMBER() OVER (ORDER BY embedding <=> %(emb)s) AS r
        FROM chunks WHERE TRUE {where}
        ORDER BY embedding <=> %(emb)s LIMIT %(n)s
    ),
    fused AS (
        SELECT COALESCE(b.id, v.id) AS id, b.r AS bm25_rank, v.r AS vec_rank,
               COALESCE(1.0 / (%(rrf)s + b.r), 0) + COALESCE(1.0 / (%(rrf)s + v.r), 0) AS score
        FROM bm25 b FULL OUTER JOIN vec v ON b.id = v.id
    )
    SELECT c.id, c.ticker, c.fiscal_year, c.item, c.item_title, c.content,
           f.score::float AS score, f.bm25_rank, f.vec_rank
    FROM fused f JOIN chunks c ON c.id = f.id
    ORDER BY f.score DESC LIMIT %(k)s
    """
    with conn() as c:
        return [Hit(**r) for r in c.execute(sql, params).fetchall()]


# ---------------------------------------------------------------- financial facts
def replace_facts(ticker: str, rows: Sequence[dict]) -> int:
    with conn() as c, c.transaction():
        c.execute("DELETE FROM financial_facts WHERE ticker = %s", (ticker,))
        with c.cursor() as cur:
            cur.executemany(
                """INSERT INTO financial_facts (ticker, concept, metric, unit, value, fiscal_year,
                                                period_start, period_end, accession, filed_at)
                   VALUES (%(ticker)s, %(concept)s, %(metric)s, %(unit)s, %(value)s, %(fiscal_year)s,
                           %(period_start)s, %(period_end)s, %(accession)s, %(filed_at)s)
                   ON CONFLICT (ticker, metric, period_end) DO NOTHING""",
                rows,
            )
    return len(rows)


def get_facts(tickers: Sequence[str], metrics: Sequence[str], years: Sequence[int] | None = None) -> list[dict]:
    sql = """SELECT ticker, metric, concept, unit, value::float AS value, fiscal_year, period_end, accession
             FROM financial_facts WHERE ticker = ANY(%(t)s) AND metric = ANY(%(m)s)"""
    params: dict = {"t": list(tickers), "m": list(metrics)}
    if years:
        # include the prior year so YoY change can be computed
        sql += " AND fiscal_year = ANY(%(y)s)"
        params["y"] = sorted({y for y in years} | {y - 1 for y in years})
    sql += " ORDER BY ticker, metric, fiscal_year"
    with conn() as c:
        return c.execute(sql, params).fetchall()
