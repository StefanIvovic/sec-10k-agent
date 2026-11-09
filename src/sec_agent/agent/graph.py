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
    status: str                  # answered | partial | not_found (set by finalize)
    draft_answer: str            # LLM answer before finalize; kept for traces, never shown if not_found
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
        a = {"grounded": bool(r.grounded), "complete": bool(r.complete), "answer_found": bool(r.answer_found),
             "missing": r.missing or "", "follow_up_queries": r.follow_up_queries or []}
        return {"assessment": a, **_step("assess", t0, **a)}

    def after_assess(state: AgentState) -> str:
        a = state["assessment"]
        ok = a["grounded"] and a["complete"] and a["answer_found"]
        if not ok and state.get("refinements", 0) < settings.max_refinements:
            return "refine_search"
        return "finalize"

    def refine_search(state: AgentState) -> dict:
        t0 = time.time()
        p, a = state["plan"], state["assessment"]
        # the self-check's follow-up queries, or else what it said is missing (e.g. "Apple sources do not
        # cover cybersecurity risks") - that text is a usable query for BM25 and vector search
        queries = a["follow_up_queries"][:2] or [
            f"{state['question']} {a['missing'][:200]}".strip()]  # missing alone can be a poor query ("X is not present")
        hits = tools.search_filings(queries, p["tickers"], p["fiscal_years"], None)
        return {"hits": hits[: settings.retrieve_k], "refinements": state.get("refinements", 0) + 1,
                **_step("refine_search", t0, n=len(hits), queries=queries)}

    def finalize(state: AgentState) -> dict:
        """Only grounded content reaches the user.

        answered  - self-check: grounded and complete
        partial   - grounded but incomplete: answer + a fixed note (no LLM text added)
        not_found - not grounded or no answer in the sources after the retry, or no sources at all:
                    fixed message, no citations.
                    The rejected draft stays in the state / MLflow trace for debugging only.
        """
        t0 = time.time()
        a = state["assessment"]
        has_sources = bool(state.get("hits") or state.get("facts") or state.get("graph_rows"))
        draft = state.get("answer", "")
        if not has_sources or not a["grounded"] or not a["answer_found"]:
            status, final = "not_found", NOT_FOUND_MESSAGE + f"\n\nIndexed filings: {tools.describe_companies_short()}"
        elif not a["complete"] or tools.says_not_in_sources(draft):
            # also partial when the answer itself says the filings lack something (e.g. "no mention of an
            # office in Serbia"): truthful, but it must show up in the MLflow review of unanswered questions
            status, final = "partial", draft + "\n\n" + PARTIAL_NOTE
        else:
            status, final = "answered", draft
        _tag_trace(state["question"], status, a, state.get("refinements", 0))
        return {"status": status, "answer": final, "draft_answer": draft, **_step("finalize", t0, status=status)}

    g = StateGraph(AgentState)
    g.add_node("analyze", analyze)
    g.add_node("search_text", search_text)
    g.add_node("financials", financials)
    g.add_node("graph", graph_node)
    g.add_node("answer", answer)
    g.add_node("assess", assess)
    g.add_node("refine_search", refine_search)
    g.add_node("finalize", finalize)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route, ["search_text", "financials", "graph"])
    g.add_edge("search_text", "answer")   # parallel branches join: answer runs once per superstep
    g.add_edge("financials", "answer")
    g.add_edge("graph", "answer")
    g.add_edge("answer", "assess")
    g.add_conditional_edges("assess", after_assess, ["refine_search", "finalize"])
    g.add_edge("refine_search", "answer")
    g.add_edge("finalize", END)
    return g.compile()


NOT_FOUND_MESSAGE = ("I couldn't find data in the ingested 10-K filings to answer this question reliably, "
                     "so I'm not giving an answer.")
PARTIAL_NOTE = ("_Note: part of this question could not be answered from the ingested 10-K filings; "
                "the answer above covers only what the filings support._")


def _tag_trace(question: str, status: str, assessment: dict, refinements: int) -> None:
    """Tag the MLflow trace so unanswered questions can be found: tags.answer_status = 'not_found'."""
    try:
        import mlflow

        mlflow.update_current_trace(tags={
            "answer_status": status,
            "grounded": str(assessment.get("grounded")),
            "complete": str(assessment.get("complete")),
            "refinements": str(refinements),
            "question": question[:250],
            "missing": (assessment.get("missing") or "")[:500],
        })
    except Exception:  # tracing off or no active trace: never break the answer
        pass


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
    status = state.get("status", "answered")
    return {
        "question": question,
        "status": status,   # answered | partial | not_found
        "answer": state.get("answer", ""),
        "citations": [] if status == "not_found"
        else cited_sources(state.get("answer", ""), state.get("citation_map", {})),
        "plan": state.get("plan"),
        "cypher": state.get("cypher"),
        # for not_found, drop the free-text critique: it can restate the rejected claims (kept in MLflow)
        "assessment": {k: v for k, v in (state.get("assessment") or {}).items()
                       if status != "not_found" or k in ("grounded", "complete", "answer_found")},
        "steps": state.get("steps", []),
    }
