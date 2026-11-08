-- Schema for the SEC 10-K agent. Runs once on first container start.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_search;

CREATE TABLE IF NOT EXISTS companies (
    cik     TEXT PRIMARY KEY,
    ticker  TEXT UNIQUE NOT NULL,
    name    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS filings (
    accession         TEXT PRIMARY KEY,
    cik               TEXT NOT NULL REFERENCES companies(cik),
    form              TEXT NOT NULL,
    fiscal_year       INT  NOT NULL,
    period_of_report  DATE,
    filed_at          DATE,
    url               TEXT NOT NULL
);

-- Text chunks of 10-K sections: dense (pgvector) + sparse (pg_search BM25) retrieval
CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    accession    TEXT NOT NULL REFERENCES filings(accession) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    fiscal_year  INT  NOT NULL,
    item         TEXT NOT NULL,     -- e.g. '1A', '7'
    item_title   TEXT NOT NULL,     -- e.g. 'Risk Factors'
    chunk_index  INT  NOT NULL,
    content      TEXT NOT NULL,
    embedding    vector(1024) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_bm25
    ON chunks USING bm25 (id, content, ticker, item, fiscal_year);

CREATE INDEX IF NOT EXISTS chunks_filter_idx ON chunks (ticker, fiscal_year, item);

-- Structured XBRL facts from the SEC companyfacts API (annual, 10-K only)
CREATE TABLE IF NOT EXISTS financial_facts (
    ticker        TEXT NOT NULL,
    concept       TEXT NOT NULL,   -- us-gaap tag, e.g. 'NetIncomeLoss'
    metric        TEXT NOT NULL,   -- friendly key, e.g. 'net_income'
    unit          TEXT NOT NULL,   -- 'USD', 'USD/shares', ...
    value         NUMERIC NOT NULL,
    fiscal_year   INT NOT NULL,    -- year of period_end
    period_start  DATE,
    period_end    DATE NOT NULL,
    accession     TEXT NOT NULL,
    filed_at      DATE,
    PRIMARY KEY (ticker, metric, period_end)
);
