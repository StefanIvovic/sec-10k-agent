"""Neo4j knowledge graph of companies, filings, segments, products, competitors, executives and risks.

Schema:
  (:Company {key, ticker, cik, name})-[:FILED]->(:Filing {accession, fiscal_year})
      # key = normalized lowercase name, e.g. 'microsoft'
  (:Company)-[:HAS_SEGMENT {fiscal_year}]->(:Segment {name})
  (:Company)-[:OFFERS {fiscal_year}]->(:Product {name})
  (:Company)-[:COMPETES_WITH {fiscal_year}]->(:Company {name})      # competitors have no ticker unless also ingested
  (:Person {name})-[:OFFICER_OF {title, fiscal_year}]->(:Company)
  (:Company)-[:FACES_RISK {fiscal_year}]->(:Risk {title, category})
"""
from __future__ import annotations

import re
from functools import lru_cache

from neo4j import Driver, GraphDatabase, unit_of_work

from ..config import get_settings

GRAPH_SCHEMA = __doc__.split("Schema:")[1].strip()

_WRITE_CLAUSES = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+\{|FOREACH)\b", re.I)


@lru_cache
def get_driver() -> Driver:
    s = get_settings()
    return GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))


def ensure_constraints() -> None:
    stmts = [
        "CREATE CONSTRAINT company_key IF NOT EXISTS FOR (c:Company) REQUIRE c.key IS UNIQUE",
        "CREATE CONSTRAINT filing_acc IF NOT EXISTS FOR (f:Filing) REQUIRE f.accession IS UNIQUE",
        "CREATE CONSTRAINT segment_name IF NOT EXISTS FOR (s:Segment) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT person_name IF NOT EXISTS FOR (p:Person) REQUIRE p.name IS UNIQUE",
        "CREATE CONSTRAINT risk_title IF NOT EXISTS FOR (r:Risk) REQUIRE r.title IS UNIQUE",
        "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
    ]
    with get_driver().session() as s:
        for q in stmts:
            s.run(q)


def write_filing_graph(company: dict, filing: dict, entities: dict) -> None:
    """company: ticker, cik, name. entities: segments, products, competitors, executives[{name,title}], risks[{title,category}]"""
    fy = filing["fiscal_year"]
    params = {"c": company, "f": filing, "fy": fy, **entities}
    q = """
    MERGE (co:Company {key: $c.key}) SET co.ticker = $c.ticker, co.cik = $c.cik, co.name = $c.name
    MERGE (f:Filing {accession: $f.accession}) SET f.fiscal_year = $fy, f.form = $f.form, f.url = $f.url
    MERGE (co)-[:FILED]->(f)
    WITH co
    CALL (co) { UNWIND $segments AS s
      MERGE (x:Segment {name: s}) MERGE (co)-[r:HAS_SEGMENT {fiscal_year: $fy}]->(x) }
    CALL (co) { UNWIND $products AS p
      MERGE (x:Product {name: p}) MERGE (co)-[r:OFFERS {fiscal_year: $fy}]->(x) }
    CALL (co) { UNWIND $competitors AS comp
      WITH co, comp WHERE comp.key <> co.key
      MERGE (x:Company {key: comp.key}) ON CREATE SET x.name = comp.name
      MERGE (co)-[r:COMPETES_WITH {fiscal_year: $fy}]->(x) }
    CALL (co) { UNWIND $executives AS e
      MERGE (p:Person {name: e.name}) MERGE (p)-[r:OFFICER_OF {fiscal_year: $fy}]->(co) SET r.title = e.title }
    CALL (co) { UNWIND $risks AS rk
      MERGE (x:Risk {title: rk.title}) SET x.category = rk.category
      MERGE (co)-[r:FACES_RISK {fiscal_year: $fy}]->(x) }
    """
    with get_driver().session() as s:
        s.run(q, params).consume()


_SUFFIX = re.compile(
    r"\([^)]*\)|\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|llc|lp|"
    r"holdings?|group|nv|sa|ag|se|the)\b|[,&']", re.I)


def company_key(name: str) -> str:
    """'MICROSOFT CORP' / 'Microsoft Corporation' -> 'microsoft'"""
    name = re.sub(r"\.(com|net)\b", "", name, flags=re.I).replace(".", "")
    return re.sub(r"\s+", " ", _SUFFIX.sub(" ", name)).strip().lower()


def filing_in_graph(accession: str) -> bool:
    with get_driver().session() as s:
        return s.run("MATCH (f:Filing {accession: $a}) RETURN count(f) AS n", a=accession).single()["n"] > 0


class UnsafeCypher(ValueError):
    pass


def run_read_cypher(cypher: str, params: dict | None = None, limit: int = 50) -> list[dict]:
    """Execute LLM-generated Cypher in a read-only transaction."""
    if _WRITE_CLAUSES.search(cypher):
        raise UnsafeCypher("Only read queries are allowed")

    @unit_of_work(timeout=15)  # driver 6: Query objects are only accepted by session.run
    def _tx(tx):
        res = tx.run(cypher, params or {})
        return [r.data() for r in res][:limit]

    with get_driver().session(default_access_mode="READ") as s:
        return s.execute_read(_tx)
