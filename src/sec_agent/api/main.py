"""FastAPI service.  uvicorn sec_agent.api.main:app --port 8000"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import mlflow
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..agent.graph import ask, get_agent
from ..config import get_settings
from ..db import graph, postgres as pg
from ..llm import ensure_embed_model_loaded
from ..tracing import init_mlflow

log = logging.getLogger("sec_agent.api")
DEPLOYMENT = "ingest-companies/sec-10k-ingest"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_mlflow()
    log.info("embedding model: %s", ensure_embed_model_loaded())
    get_agent()  # compile graph + load (optionally optimized) DSPy program once
    yield


app = FastAPI(title="SEC 10-K Agent", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, examples=["What are NVIDIA's main supply chain risks?"])
    tickers: list[str] = Field(default_factory=list, description="optional filter, overrides the planner")
    fiscal_years: list[int] = Field(default_factory=list)


class IngestRequest(BaseModel):
    tickers: list[str] = Field(min_length=1, examples=[["AAPL", "MSFT"]])
    n_filings: int = Field(2, ge=1, le=10)
    force: bool = False
    with_graph: bool = True


@app.post("/ask")
def ask_endpoint(req: AskRequest) -> dict:
    # sync endpoint -> FastAPI runs it in a worker thread; LangGraph + DSPy calls are blocking
    result = ask(req.question, [t.upper() for t in req.tickers], req.fiscal_years)
    result["trace_id"] = mlflow.get_last_active_trace_id()
    return result


@app.get("/companies")
def companies() -> list[dict]:
    return pg.list_companies()


@app.post("/ingest")
async def ingest(req: IngestRequest) -> dict:
    params = req.model_dump()
    try:  # preferred: hand off to the Prefect deployment served by `python -m sec_agent.ingest.flows --serve`
        from prefect.deployments import run_deployment

        run = await run_deployment(DEPLOYMENT, parameters=params, timeout=0)
        return {"mode": "prefect-deployment", "flow_run_id": str(run.id),
                "ui": f"http://localhost:4200/runs/flow-run/{run.id}"}
    except Exception as e:
        log.info("Prefect deployment unavailable (%s); running flow in a background thread", e)
    from ..ingest.flows import ingest_companies

    threading.Thread(target=ingest_companies, kwargs=params, daemon=True).start()
    return {"mode": "background-thread", "detail": "flow started; follow it in the Prefect UI at http://localhost:4200"}


@app.get("/health")
def health() -> dict:
    s = get_settings()
    checks = {}
    try:
        with pg.conn() as c:
            checks["postgres"] = c.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"] >= 0
    except Exception as e:
        checks["postgres"] = str(e)
    try:
        graph.get_driver().verify_connectivity()
        checks["neo4j"] = True
    except Exception as e:
        checks["neo4j"] = str(e)
    try:
        if s.llm_provider == "lmstudio":
            models = httpx.get(f"{s.llm_base_url}/models", timeout=5).json()["data"]
            checks["llm"] = s.llm_model in {m["id"] for m in models}
        else:  # cloud provider: only check that a key is configured
            key = s.llm_api_key if s.llm_api_key != "lm-studio" else os.environ.get(
                "ANTHROPIC_API_KEY" if s.llm_provider == "anthropic" else "OPENAI_API_KEY")
            checks["llm"] = bool(key) or f"no API key for {s.llm_provider}"
        state = httpx.get(f"{s.embed_base_url.rsplit('/v1', 1)[0]}/api/v0/models", timeout=5).json()["data"]
        # must be loaded already: a JIT load would land on the GPU and evict the chat model
        checks["embed_model_loaded"] = any(m["id"] == s.embed_model and m.get("state") == "loaded" for m in state)
    except Exception as e:
        checks["llm"] = str(e)
    ok = all(v is True for v in checks.values())
    if not ok:
        raise HTTPException(503, checks)
    return checks


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")
