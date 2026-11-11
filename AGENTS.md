# Running this project (for coding agents)

Follow README.md for setup. Notes that matter when you drive it:

- Services: `docker compose up -d`. Check with `docker compose ps` (postgres must be healthy) and
  `curl http://127.0.0.1:5000` (MLflow).
- LM Studio must serve bge-m3 (embeddings) for every setup. Check: `lms ps`. The app loads it on the CPU
  at startup if missing. Never load a model with a JIT request: it can unload the other model.
- Use 127.0.0.1, not localhost, for LM Studio.
- `.env` is required: copy `.env.example` and pick one chat-model block. Never commit `.env`.
- SEC_USER_AGENT must be "Name email" (e.g. `Jane Doe jane.doe@example.com`); SEC EDGAR returns 403
  without it. Ask the user for it; don't copy it from other files or make one up.
- Ingest before asking: `python -m sec_agent.ingest.flows AAPL MSFT NVDA --fiscal-year 2026`.
  Done when `curl http://127.0.0.1:8000/companies` (API running) or the ingest log lists all three.
- Smoke test: `python -m sec_agent.cli "What offset part of the More Personal Computing revenue decrease?"`
  should answer "Search advertising" with [S#] citations.
- Health: `GET /health` reports postgres, neo4j, llm and embed_model_loaded.
- Evals take 1-1.5 h with a local 27B model (`python -m sec_agent.eval.run --split dev`). Ask before
  running them; with a cloud model they cost money.
- Do not change files in data/eval/ to make an eval pass.
