"""LangGraph agent.

        ┌──────────┐
        │ analyze  │  DSPy: tickers, years, items, search queries, metrics, needs_graph
        └────┬─────┘
   fan-out   │  (parallel, only the branches the plan needs)
 ┌───────────┼───────────────┐
 ▼           ▼               ▼
search_text  financials      graph (DSPy text→Cypher, read-only, 1 retry on error)
 └───────────┼───────────────┘
             ▼
        ┌──────────┐
        │ answer   │  DSPy: cited answer
        └────┬─────┘
             ▼
        ┌──────────┐  DSPy: grounded/complete? ── incomplete & budget left ──► refine_search ─► answer
        │ assess   │
        └────┬─────┘
             ▼
            END
"""
from __future__ import annotations

import operator
import re
import time
from functools import lru_cache
from typing import Annotated, Any, TypedDict

import dspy
from langgraph.graph import END, START, StateGraph

from ..config import get_settings
from ..db.graph import GRAPH_SCHEMA, UnsafeCypher
from ..llm import configure_dspy
from . import tools
from .signatures import SecQAProgram, load_program


def merge_hits(left: list[dict], right: list[dict]) -> list[dict]:
    """Reducer: union of retrieved chunks by chunk_id, keeping the best score."""
    by_id = {h["chunk_id"]: h for h in left}
    for h in right:
        if h["chunk_id"] not in by_id or h["score"] > by_id[h["chunk_id"]]["score"]:
            by_id[h["chunk_id"]] = h
    return sorted(by_id.values(), key=lambda h: h["score"], reverse=True)


class AgentState(TypedDict, total=False):
    question: str
    tickers: list[str]           # optional caller-provided filters
    fiscal_years: list[int]
    plan: dict
    hits: Annotated[list[dict], merge_hits]
    facts: list[dict]
    graph_rows: list[dict]
    cypher: str
    sources_text: str
    citation_map: dict
    answer: str
    assessment: dict
    refinements: int
    steps: Annotated[list[dict], operator.add]   # timing/debug trail


def _step(name: str, t0: float, **info) -> dict:
    return {"steps": [{"node": name, "seconds": round(time.time() - t0, 2), **info}]}


def build_graph(program: SecQAProgram | None = None):
    configure_dspy()
    program = program or load_program()
    settings = get_settings()

    def analyze(state: AgentState) -> dict:
        t0 = time.time()
        known = tools.describe_companies()
        p = program.analyze(question=state["question"], available_companies=known,
                            available_metrics=tools.describe_metrics())
        plan = tools.normalize_plan(p, state["question"], known)
        plan["tickers"] = state.get("tickers") or plan["tickers"]
        plan["fiscal_years"] = state.get("fiscal_years") or plan["fiscal_years"]
        return {"plan": plan, "refinements": 0, **_step("analyze", t0, plan=plan)}

    def route(state: AgentState) -> list[str]:
        plan = state["plan"]
        branches = ["search_text"]
        if plan["metrics"] and plan["tickers"]:
            branches.append("financials")
        if plan["needs_graph"]:
            branches.append("graph")
        return branches

    def search_text(state: AgentState) -> dict:
        t0 = time.time()
        p = state["plan"]
        hits = tools.search_filings(p["search_queries"], p["tickers"], p["fiscal_years"], p["items"])
        return {"hits": hits[: settings.retrieve_k * 2], **_step("search_text", t0, n=len(hits))}

    def financials(state: AgentState) -> dict:
        t0 = time.time()
        p = state["plan"]
        facts = tools.get_financials(p["tickers"], p["metrics"], p["fiscal_years"])
        return {"facts": facts, **_step("financials", t0, n=len(facts))}

    def graph_node(state: AgentState) -> dict:
        t0 = time.time()
        err, cypher, rows = "", "", []
        for _ in range(2):
            cypher = program.cypher(question=state["question"], graph_schema=GRAPH_SCHEMA,
                                    tickers=state["plan"]["tickers"], previous_error=err).cypher
            cypher = re.sub(r"^```\w*|```$", "", cypher.strip()).strip()
            cypher = cypher.split(";")[0].strip()  # Neo4j runs one statement per query
            try:
                rows = tools.query_graph(cypher)
                break
            except UnsafeCypher as e:
                err = str(e)
            except Exception as e:  # syntax errors etc. -> let the LLM fix it once
                err = f"{type(e).__name__}: {e}"[:500]
        return {"cypher": cypher, "graph_rows": rows, **_step("graph", t0, n=len(rows), error=err or None)}

    def answer(state: AgentState) -> dict:
        t0 = time.time()
        text, cmap = tools.format_sources(state.get("hits", []), state.get("facts", []), state.get("graph_rows", []))
        a = program.answer(question=state["question"], sources=text).answer
        return {"answer": a, "sources_text": text, "citation_map": cmap, **_step("answer", t0)}

    def assess(state: AgentState) -> dict:
        t0 = time.time()
        r = program.assess(question=state["question"], sources=state["sources_text"], answer=state["answer"])
        a = {"grounded": bool(r.grounded), "complete": bool(r.complete), "missing": r.missing,
             "follow_up_queries": r.follow_up_queries or []}
        return {"assessment": a, **_step("assess", t0, **a)}

    def after_assess(state: AgentState) -> str:
        a = state["assessment"]
        if (not a["complete"] or not a["grounded"]) and a["follow_up_queries"] \
                and state.get("refinements", 0) < settings.max_refinements:
            return "refine_search"
        return END

    def refine_search(state: AgentState) -> dict:
        t0 = time.time()
        p = state["plan"]
        # broaden: drop the item filter, keep company/year filters
        hits = tools.search_filings(state["assessment"]["follow_up_queries"][:2], p["tickers"], p["fiscal_years"], None)
        return {"hits": hits[: settings.retrieve_k], "refinements": state.get("refinements", 0) + 1,
                **_step("refine_search", t0, n=len(hits))}

    g = StateGraph(AgentState)
    g.add_node("analyze", analyze)
    g.add_node("search_text", search_text)
    g.add_node("financials", financials)
    g.add_node("graph", graph_node)
    g.add_node("answer", answer)
    g.add_node("assess", assess)
    g.add_node("refine_search", refine_search)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route, ["search_text", "financials", "graph"])
    g.add_edge("search_text", "answer")   # parallel branches join: answer runs once per superstep
    g.add_edge("financials", "answer")
    g.add_edge("graph", "answer")
    g.add_edge("answer", "assess")
    g.add_conditional_edges("assess", after_assess, ["refine_search", END])
    g.add_edge("refine_search", "answer")
    return g.compile()


@lru_cache
def get_agent():
    return build_graph()


def cited_sources(answer: str, cmap: dict) -> list[dict]:
    ids = dict.fromkeys(re.findall(r"\[([SFG]\d+)\]", answer))
    # also catch grouped citations like [S1, S3]
    for grp in re.findall(r"\[((?:[SFG]\d+\s*,\s*)+[SFG]\d+)\]", answer):
        for i in re.split(r"\s*,\s*", grp):
            ids.setdefault(i, None)
    return [{"id": i, **cmap[i]} for i in ids if i in cmap]


def ask(question: str, tickers: list[str] | None = None, fiscal_years: list[int] | None = None) -> dict[str, Any]:
    state = get_agent().invoke({"question": question, "tickers": tickers or [], "fiscal_years": fiscal_years or []})
    return {
        "question": question,
        "answer": state.get("answer", ""),
        "citations": cited_sources(state.get("answer", ""), state.get("citation_map", {})),
        "plan": state.get("plan"),
        "cypher": state.get("cypher"),
        "assessment": state.get("assessment"),
        "steps": state.get("steps", []),
    }
