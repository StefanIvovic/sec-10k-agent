# SEC 10-K Agent

Ask questions about companies' SEC 10-K annual reports and get answers with citations.
It combines the filing text (hybrid search in Postgres), XBRL financial data, and a knowledge graph (Neo4j),
and says "not found" instead of guessing.

Stack: LangGraph, DSPy, LlamaIndex, PostgreSQL (pgvector + pg_search), Neo4j, MLflow, Prefect, FastAPI.

## Requirements

- Docker
- Python 3.11+
- [LM Studio](https://lmstudio.ai) for the embedding model (bge-m3, runs on CPU), and for the chat model if you run it locally
- For a local chat model: a 24 GB GPU and 32 GB+ RAM. Otherwise an Anthropic or OpenAI API key.

## Setup

```bash
git clone https://github.com/StefanIvovic/sec-10k-agent.git
cd sec-10k-agent
docker compose up -d                  # Postgres :5433, Neo4j :7474, MLflow :5000, Prefect :4200

python -m venv .venv
.venv\Scripts\activate                # Windows
source .venv/bin/activate             # macOS / Linux
pip install -e .

cp .env.example .env                  # Windows: copy .env.example .env
```

In `.env`, set `SEC_USER_AGENT` to your name and email (the SEC requires it) and choose the chat model (below).

Using Claude Code or Codex? The agent instructions are in [AGENTS.md](AGENTS.md).

Download the models in LM Studio:

```bash
lms get https://huggingface.co/gpustack/bge-m3-GGUF                  # embeddings (Q8_0), always needed
lms get https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF   # chat (Q4_K_M), only for the local setup
```

## Chat model

**Local (tested):** LM Studio with `qwen3.8-27b` (Q4_K_M) on an RTX 3090 (24 GB) with 40 GB RAM. This is the default in `.env.example`.

```bash
lms load qwen3.8-27b --gpu max -c 30976 --identifier qwen3.8-27b
```

- In LM Studio's settings for the model, turn off **Keep model in memory** and **Try mmap()**. Otherwise LM Studio holds about 30 GB of RAM.
- bge-m3 is loaded on the CPU automatically at startup (`EMBED_GPU_OFFLOAD=off`).
- Use `127.0.0.1`, not `localhost`, in `.env` (on Windows `localhost` adds a ~2 s delay per connection).

**Anthropic or OpenAI:** comment out the LM Studio block in `.env` and uncomment one of the others:

```ini
LLM_PROVIDER=anthropic
LLM_MODEL=claude-opus-5-5
LLM_API_KEY=sk-ant-...
LLM_EFFORT=medium
```

Embeddings stay local (bge-m3 in LM Studio) with any chat model. The evals were run with qwen3.8-27b only.

## Download 10-K filings

Downloads each company's 10-K from SEC EDGAR, splits it by Item, embeds the chunks, loads the XBRL financial data,
and extracts the knowledge graph:

```bash
python -m sec_agent.ingest.flows AAPL MSFT NVDA --fiscal-year 2026   # the filings the evals use
python -m sec_agent.ingest.flows AMZN --filings 2                     # any ticker, latest 2 10-Ks
```

Add `--force` to re-ingest a filing and `--no-graph` to skip the knowledge graph.

## Ask

```bash
python -m sec_agent.cli "What are NVIDIA's main supply chain risks?"
uvicorn sec_agent.api.main:app --port 8000     # web UI at http://127.0.0.1:8000
```

API: `POST /ask` with `{"question": "..."}`; also `/ingest`, `/companies`, `/health`. Traces of every question are in MLflow at http://127.0.0.1:5000.

## Evaluate

```bash
python -m sec_agent.eval.run --split dev
python -m sec_agent.eval.run --split test
```

The questions are in `data/eval/` ([format](data/eval/README.md)). Results with qwen3.8-27b on the FY2026 filings of AAPL, MSFT and NVDA:

| Split | Passed |
|---|---|
| dev | 111 / 111 |
| test | 112 / 114 |

## License

MIT
