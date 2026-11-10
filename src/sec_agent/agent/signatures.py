"""DSPy signatures and the optimizable program used by the LangGraph agent."""
from __future__ import annotations

from pathlib import Path

import dspy

from ..config import get_settings


class AnalyzeQuestion(dspy.Signature):
    """Plan how to answer a question about companies' SEC Form 10-K annual reports.

    10-K items: 1 Business (products, segments, competition, employees), 1A Risk Factors,
    1C Cybersecurity, 2 Properties, 3 Legal Proceedings, 5 Market/stock repurchases & dividends,
    7 MD&A (results of operations, liquidity, capital resources), 7A Market risk (FX, interest rates),
    8 Financial statements and notes, 9A Controls, 10-14 Governance (often by reference to the proxy),
    15 Exhibits (some filers place financial statements here).
    """

    question: str = dspy.InputField()
    available_companies: str = dspy.InputField(desc="ingested companies: ticker, name, fiscal years, segments, products; "
                                                    "a question naming only a segment or product is about that company")
    available_metrics: str = dspy.InputField(desc="metric keys available as structured XBRL data")
    tickers: list[str] = dspy.OutputField(desc="tickers the question is about (map names to tickers); [] means all companies")
    fiscal_years: list[int] = dspy.OutputField(desc="fiscal years explicitly asked about; [] if none stated")
    sections: list[str] = dspy.OutputField(desc="10-K item numbers most likely to hold the answer, e.g. ['1A']; [] if unsure")
    search_queries: list[str] = dspy.OutputField(desc="1-3 short keyword-rich queries to search the filing text")
    metrics: list[str] = dspy.OutputField(desc="metric keys needed for numeric questions; [] if none")
    needs_graph: bool = dspy.OutputField(desc="true if the question is about relationships across entities: competitors, "
                                              "segments, products, executives, or risks shared between companies")


class GenerateCypher(dspy.Signature):
    """Write ONE read-only Neo4j Cypher query that answers the question using the graph schema.
    Match companies by ticker when known (c.ticker = 'AAPL'); match names case-insensitively with
    toLower(...) CONTAINS. Return readable properties (names, titles, fiscal_year), not whole nodes.
    Write a SINGLE statement (no semicolons) and use only labels, relationships and properties in the
    schema; numbers like revenue or R&D are NOT in the graph. Always end with LIMIT 50.
    Never use CREATE, MERGE, SET, DELETE or CALL subqueries."""

    question: str = dspy.InputField()
    graph_schema: str = dspy.InputField()
    tickers: list[str] = dspy.InputField()
    previous_error: str = dspy.InputField(desc="error from the last attempt, or empty")
    cypher: str = dspy.OutputField()


class AnswerQuestion(dspy.Signature):
    """Answer the question about 10-K filings using ONLY the provided sources.
    Cite every claim with its source id in square brackets, e.g. [S2] or [F1]. Prefer structured
    financial data [F#] for numbers; state fiscal years and units explicitly; compute changes and
    ratios step by step from the cited numbers. If the sources do not contain the answer, say what is
    missing instead of guessing. If the question asks for an exact figure (a specific date, unit or
    scope) and the sources only give an approximate or differently scoped one (e.g. "approximately
    150,000 employees" when asked for the exact headcount on a given day), first say that the sources do not state the exact figure,
    then give the closest figure with its wording and citation.
    Be concise: a direct answer first, then supporting detail."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField(desc="numbered excerpts: [S#] filing text, [F#] XBRL financial data, [G#] knowledge graph rows")
    answer: str = dspy.OutputField(desc="markdown answer with [S#]/[F#]/[G#] citations")


class AssessAnswer(dspy.Signature):
    """Check a draft answer against its sources, claim by claim.

    - 'grounded' is false if ANY factual claim is not supported by the cited sources. A claim about a
      company must be supported by a source for THAT company (a Microsoft excerpt does not support a
      claim about Apple). Numbers must match the sources. A statement that the sources do not contain
      some information is grounded.
    - 'complete' is false if part of the question is unanswered.
    - 'answer_found' is false if the sources contain no information that answers the question at all
      (e.g. the answer only says the information is not available). It is true if the answer says the
      exact figure is not stated but gives a related figure or explanation from the sources (an
      approximate number, or how the company reports it instead); 'complete' is then false.
    - If not grounded or not complete, propose search queries that would find the missing or
      unsupported information in the 10-K text, naming the company."""

    question: str = dspy.InputField()
    sources: str = dspy.InputField()
    answer: str = dspy.InputField()
    grounded: bool = dspy.OutputField()
    complete: bool = dspy.OutputField()
    answer_found: bool = dspy.OutputField()
    missing: str = dspy.OutputField(desc="what is missing or unsupported; empty if nothing")
    follow_up_queries: list[str] = dspy.OutputField(desc="1-2 search queries if not grounded or not complete; else []")


class SecQAProgram(dspy.Module):
    """Container for every LLM step, so DSPy can optimize, save and load them together."""

    def __init__(self):
        super().__init__()
        self.analyze = dspy.Predict(AnalyzeQuestion)
        self.cypher = dspy.Predict(GenerateCypher)
        # ChainOfThought adds a hidden `reasoning` output before `answer`: with qwen's thinking off, the model
        # otherwise works things out in the answer itself (e.g. states a ranking, computes, then "corrects" it)
        self.answer = dspy.ChainOfThought(AnswerQuestion)
        self.assess = dspy.Predict(AssessAnswer)


def load_program() -> SecQAProgram:
    program = SecQAProgram()
    path = Path(get_settings().compiled_program_path)
    if path.exists():
        program.load(str(path))
    return program
