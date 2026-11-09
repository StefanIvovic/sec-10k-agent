# SEC 10-K Agent

Ask questions about companies' 10-K annual reports and get cited answers that combine three sources:

| Source | Store | Used for |
|---|---|---|
| Filing text, split by Item and chunked | PostgreSQL: **pgvector** (dense) + **pg_search** BM25 (sparse), fused with RRF in SQL | narrative questions: risks, strategy, MD&A drivers |
| XBRL financial facts from the SEC companyfacts API | PostgreSQL table `financial_facts` | exact numbers, YoY change |
| Entities extracted by the LLM (segments, products, competitors, executives, risks) | **Neo4j** | relationship and cross-company questions |

```
            ┌────────────── Prefect flow (ingest) ──────────────┐
EDGAR ─────►│ 10-K HTML → Items → LlamaIndex chunks → bge-m3     │──► Postgres (pgvector + BM25)
            │ companyfacts JSON → normalized annual metrics      │──► Postgres financial_facts
            │ DSPy extraction (Item 1, 1A)                       │──► Neo4j
            └────────────────────────────────────────────────────┘

FastAPI /ask ─► LangGraph:  analyze ─┬─ search_text ─┐
                  (DSPy)             ├─ financials ──┼─► answer ─► assess ─┬─► END
                                     └─ graph (Cypher)┘    (DSPy)  (DSPy)  └─► refine_search ─► answer
                         every node and DSPy/LM call is traced in MLflow
```

## Stack

- **LangGraph**: agent state machine with parallel retrieval branches and a self-check/refine loop (`src/sec_agent/agent/graph.py`)
- **DSPy**: every LLM step is a typed signature (`agent/signatures.py`, `ingest/extract.py`), optimizable with `eval/optimize.py`
- **LlamaIndex**: chunking (`SentenceSplitter`), embeddings (`OpenAILikeEmbedding`), and a custom `BaseRetriever` over Postgres (`agent/tools.py`)
- **PostgreSQL** (ParadeDB image): `vector` 0.8 + `pg_search` 0.26 (`sql/init.sql`, `db/postgres.py`)
- **Neo4j 5**: knowledge graph; LLM-written Cypher runs in read-only transactions (`db/graph.py`)
- **MLflow 3**: traces (LangGraph + DSPy autolog), eval runs, optimized program artifacts
- **Prefect 3**: ingestion flows with retries, a served deployment, and a weekly schedule (`ingest/flows.py`)
- **FastAPI**: `/ask`, `/ingest`, `/companies`, `/health`, plus a small UI at `/` (`api/main.py`)
- **LLM**: LM Studio (OpenAI-compatible) with `qwen3.8-27b` for chat and `text-embedding-bge-m3` (1024-d) for embeddings

## Setup

```powershell
docker compose up -d                       # postgres:5433, neo4j:7474/7687, mlflow:5000, prefect:4200
python -m venv .venv; .venv\Scripts\activate
pip install -e .
copy .env.example .env                     # then set SEC_USER_AGENT to "Your Name your@email.com"
```

In LM Studio, serve `qwen3.8-27b` and `text-embedding-bge-m3`.

Qwen's thinking is **off** by default (`LLM_REASONING_EFFORT=none`, sent as `reasoning_effort` in the request body).
Tested against LM Studio: this is the only switch it honors for qwen3.8. `chat_template_kwargs.enable_thinking=false`
and the `/no_think` suffix still produced reasoning tokens. Leave the value empty to restore the model's default thinking.

Sampling is sent explicitly on every request, so LM Studio's built-in defaults never apply: `LLM_TEMPERATURE=0.1`,
`LLM_TOP_P=0.9`, `LLM_TOP_K=20`, `LLM_MIN_P=0.0`, `LLM_REPEAT_PENALTY=1.0`. Each was tested to be honored by LM Studio.
The repeat penalty is 1.0 because JSON and tables legitimately repeat tokens.

> **Both models on one 24 GB GPU (LM Studio).** At 68k context, qwen3.8-27b fills the GPU by itself.
> At ~31k context (plenty here, since the largest prompts are ~10k tokens), bge-m3 fits beside it:
> ```
> lms load qwen3.8-27b --gpu max -c 30976 --identifier qwen3.8-27b
> lms load text-embedding-bge-m3 --gpu max -c 2048 --identifier text-embedding-bge-m3
> ```
> That leaves ~1.5 GB of VRAM free. If bge-m3 is missing, the API, CLI and ingestion flow load it at startup using
> `EMBED_GPU_OFFLOAD` (`max` = GPU, `off` = CPU) and `EMBED_CONTEXT`. `/health` reports `embed_model_loaded`.
>
> Load models **explicitly**. With *"Only keep last JIT loaded model"* on (`unloadPreviousJITModelOnLoad: true`),
> a JIT load of one model unloads the other and kills in-flight requests (`Model is unloaded`, `Channel Error`).
> Explicitly loaded models aren't JIT-evicted; concurrent chat and embedding requests were tested.
> If that isn't possible, set `LLM_SERIALIZE=true` so only one LM Studio call runs at a time.
>
> Use **`127.0.0.1`, not `localhost`**, for LM Studio on Windows. `localhost` resolves to `::1` first, LM Studio
> listens on IPv4 only, and every new connection then waits ~2.3 s.
>
> Measured: query embedding 0.09 s on GPU (2.5 s on CPU), 16-chunk batch 0.46 s (11.5 s on CPU). One full
> question went from 2m38s to 32 s.

## Use

```powershell
# 1. ingest: latest 2 10-Ks per company, plus XBRL facts and graph extraction
python -m sec_agent.ingest.flows AAPL MSFT NVDA --filings 2
#    or register a Prefect deployment (weekly Monday refresh) and keep it served:
python -m sec_agent.ingest.flows AAPL MSFT NVDA --serve

# 2. ask
python -m sec_agent.cli "What are NVIDIA's main supply chain risks?"
uvicorn sec_agent.api.main:app --port 8000          # UI at http://localhost:8000
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"Compare Apple and Microsoft R&D growth"}'
curl -X POST localhost:8000/ingest -H "Content-Type: application/json" -d '{"tickers":["AMZN"],"n_filings":1}'

# 3. evaluate and optimize (results in MLflow at http://localhost:5000)
python -m sec_agent.eval.dataset                     # numeric gold from XBRL + data/eval/curated.jsonl
python -m sec_agent.eval.run_eval --run-name baseline
python -m sec_agent.eval.optimize --train 12         # writes artifacts/compiled_program.json
python -m sec_agent.eval.run_eval --run-name optimized
python -m sec_agent.eval.retrieval_eval --agent      # retrieval only: is the answer chunk found, how high?
```

**Retrieval eval** (`data/eval/retrieval.jsonl`, 20 questions): the answer key is a ticker plus a distinctive phrase
from the filing, not a chunk ID, so it survives re-ingestion. The script refuses to run if a phrase no longer matches any chunk.
It reports hit@1/3/k and MRR for the hybrid search, BM25 alone, vector alone, no company filter, and
(`--agent`) the planner's real queries and filters, including whether the answer chunk reaches the answer prompt.
Without `--agent` it needs no LLM and runs in seconds.

`/ingest` calls the Prefect deployment when `--serve` is running; otherwise it runs the flow in a background thread.
Either way, runs appear in the Prefect UI.

## Example questions

- How did Apple's revenue and net income change in fiscal 2026, and what drove it?
- What cybersecurity governance does Microsoft describe, and who oversees it?
- Which risks do Apple and NVIDIA both report? *(graph)*
- What segments does NVIDIA report, and what was its R&D spend growth? *(graph + XBRL)*
- How much did Microsoft return to shareholders through buybacks and dividends last year?

## Design notes / limitations

- **Item splitting** (`ingest/parse.py`) handles ToC rows, small-caps headings split across spans
  (Microsoft), and running page headers. Some filers (e.g. NVIDIA) put the financial statements after
  Item 15, so those chunks are labelled Item 15. The planner prompt knows this.
- **Fiscal year** = calendar year of the period end. NVIDIA's FY2026 ends January 2026, which matches
  the company's own naming. Apple's FY ends in September.
- **XBRL metrics** are a curated list (`ingest/edgar.py: METRICS`) with fallback tags per metric.
  Only annual 10-K durations (330–380 days) are kept, and the latest filing wins for restated values.
- **Graph quality** depends on the LLM extraction. Companies merge on a normalized name key
  (`MICROSOFT CORP` and `Microsoft Corporation` both become `microsoft`). Risk nodes are per-title, so similar risks from
  different companies only connect when they get identical generic titles.
- **Cypher safety**: the query is regex-checked for write clauses and runs in a READ session with a 15 s timeout.
- Local 27B model latency: expect about 1–3 minutes per question (analyze, answer, assess), and longer when models swap (see above).
  `MAX_REFINEMENTS=0` skips the refine loop.
