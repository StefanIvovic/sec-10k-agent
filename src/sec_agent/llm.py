"""LLM / embedding clients. Chat: LM Studio, OpenAI or Anthropic (LLM_PROVIDER); embeddings: bge-m3 in LM Studio."""
import logging
import os
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from pathlib import Path

import dspy
import httpx
from llama_index.embeddings.openai_like import OpenAILikeEmbedding

from .config import get_settings

log = logging.getLogger(__name__)
_configured = False

# With LM Studio's "Only keep last JIT loaded model" on, a JIT load of one model unloads the other and
# kills whatever request is in flight ("Model is unloaded" / "Channel Error"). The fix is to keep the
# embedding model explicitly loaded (ensure_embed_model_loaded); explicitly loaded models are
# not JIT-evicted. LLM_SERIALIZE=true is a fallback for when that is not possible.
_lmstudio_lock = threading.RLock()
_SWAP_ERRORS = ("model is unloaded", "channel error", "terminated", "model unloaded")


def _call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if not any(s in str(e).lower() for s in _SWAP_ERRORS):
            raise
        log.warning("LM Studio dropped the request during a model swap; retrying once: %s", e)
        time.sleep(2)
        return fn(*args, **kwargs)


def _serialized(fn, *args, **kwargs):
    if not get_settings().llm_serialize:
        return _call(fn, *args, **kwargs)
    with _lmstudio_lock:
        return _call(fn, *args, **kwargs)


def ensure_embed_model_loaded() -> str:
    """Explicitly load the embedding model (EMBED_GPU_OFFLOAD, EMBED_CONTEXT) unless it is already loaded.

    Must run before the first embedding request: a JIT load would use LM Studio's defaults and, with
    "Only keep last JIT loaded model" on, unload qwen. Explicitly loaded models are not JIT-evicted.
    """
    s = get_settings()
    if not s.embed_autoload:
        return "skipped (EMBED_AUTOLOAD=false)"
    try:
        models = httpx.get(f"{s.embed_base_url.rsplit('/v1', 1)[0]}/api/v0/models", timeout=5).json()["data"]
    except Exception as e:
        return f"LM Studio not reachable: {e}"
    if any(m["id"] == s.embed_model and m.get("state") == "loaded" for m in models):
        return "already loaded"
    lms = shutil.which("lms") or str(Path.home() / ".lmstudio" / "bin" / ("lms.exe" if os.name == "nt" else "lms"))
    if not Path(lms).exists():
        log.warning("lms CLI not found; load %s manually", s.embed_model)
        return "lms CLI not found"
    cmd = [lms, "load", s.embed_model, "--gpu", s.embed_gpu_offload, "-c", str(s.embed_context),
           "--identifier", s.embed_model, "-y"]
    r = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=180)  # lms prints a unicode spinner
    if r.returncode != 0:
        log.warning("lms load failed: %s", r.stderr[-500:])
        return f"lms load failed (exit {r.returncode})"
    return f"loaded (gpu={s.embed_gpu_offload}, context={s.embed_context})"


class SerializedLM(dspy.LM):
    def __call__(self, prompt=None, *, messages=None, **kwargs):
        return _serialized(super().__call__, prompt, messages=messages, **kwargs)


def configure_dspy() -> dspy.LM:
    global _configured
    s = get_settings()
    if s.llm_provider == "anthropic":
        from .anthropic_engine import AnthropicEngine
        engine = AnthropicEngine(s.llm_model, s.llm_api_key if s.llm_api_key != "lm-studio" else None,
                                 s.max_tokens, s.llm_effort)
        lm = dspy.LM(f"anthropic/{s.llm_model}", engine=engine, max_tokens=s.max_tokens, cache=False)
    elif s.llm_provider == "openai":
        # model defaults for sampling (reasoning models reject temperature/top_p)
        extra = {"reasoning_effort": s.llm_effort} if s.llm_effort else {}
        if s.llm_api_key != "lm-studio":  # else LiteLLM reads OPENAI_API_KEY
            extra["api_key"] = s.llm_api_key
        lm = dspy.LM(f"openai/{s.llm_model}", max_tokens=s.max_tokens, cache=False, **extra)
    elif s.llm_provider == "lmstudio":
        # top_k / min_p / repeat_penalty are LM Studio extensions to the OpenAI API -> extra_body
        extra_body = {"top_k": s.llm_top_k, "min_p": s.llm_min_p, "repeat_penalty": s.llm_repeat_penalty}
        if s.llm_reasoning_effort:
            # LM Studio honors reasoning_effort="none" for qwen3.8 (enable_thinking / "/no_think" are ignored)
            extra_body["reasoning_effort"] = s.llm_reasoning_effort
        lm = SerializedLM(
            f"openai/{s.llm_model}",
            api_base=s.llm_base_url,
            api_key=s.llm_api_key,
            max_tokens=s.max_tokens,
            temperature=s.llm_temperature,
            top_p=s.llm_top_p,
            cache=False,
            extra_body=extra_body,
        )
    else:
        raise ValueError(f"LLM_PROVIDER must be lmstudio, openai or anthropic, not {s.llm_provider!r}")
    if not _configured:
        dspy.configure(lm=lm)
        _configured = True
    return lm


@lru_cache
def get_embedder() -> OpenAILikeEmbedding:
    s = get_settings()
    return OpenAILikeEmbedding(
        model_name=s.embed_model,
        api_base=s.embed_base_url,
        api_key="lm-studio",
        embed_batch_size=16,
    )


def embed_query(text: str) -> list[float]:
    return _serialized(get_embedder().get_query_embedding, text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _serialized(get_embedder().get_text_embedding_batch, texts)
