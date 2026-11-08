from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    sec_user_agent: str = "sec-10k-agent your-contact@example.com"

    llm_base_url: str = "http://127.0.0.1:1234/v1"   # not localhost: Windows tries ::1 first (+2.3 s per connection)
    llm_model: str = "qwen3.8-27b"
    llm_api_key: str = "lm-studio"
    llm_max_tokens: int = 4000          # reasoning is off; raise if you turn it back on
    llm_temperature: float = 0.1
    # sent explicitly so LM Studio's built-in defaults never apply (all four verified to be honored)
    llm_top_p: float = 0.9
    llm_top_k: int = 20
    llm_min_p: float = 0.0
    llm_repeat_penalty: float = 1.0      # >1 penalizes legitimately repeated tokens in JSON/tables
    llm_reasoning_effort: str = "none"   # "none" disables qwen thinking; "" = model default
    llm_serialize: bool = False          # one LM Studio call at a time; fallback if the embed model can't be pinned
    embed_autoload: bool = True          # explicitly load the embedding model at startup if missing
    embed_gpu_offload: str = "max"       # lms --gpu: "max" (GPU), "off" (CPU) or 0-1
    embed_context: int = 2048            # chunks are < ~900 tokens; small context = small VRAM buffer
    embed_model: str = "text-embedding-bge-m3"
    embed_dim: int = 1024

    postgres_dsn: str = "postgresql://sec:sec@localhost:5433/sec"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sec-agent-pw"
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment: str = "sec-10k-agent"

    chunk_size: int = 700                # tokens
    chunk_overlap: int = 100
    extract_max_chars: int = 24000       # text passed to the LLM for graph extraction
    retrieve_k: int = 8
    max_refinements: int = 1             # extra retrieval rounds when the grader says "insufficient"
    compiled_program_path: str = "artifacts/compiled_program.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
