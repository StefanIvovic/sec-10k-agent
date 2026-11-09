"""Run the agent over eval suites and log results to MLflow.

Suites
  answers    data/eval/questions.jsonl  (numeric gold from XBRL, curated keywords, open)
  grounding  data/eval/grounding.jsonl  (unanswerable + false premise; should refuse or correct)
             + data/eval/retrieval.jsonl questions as 'answerable' (should NOT be refused)
  all        both

python -m sec_agent.eval.run_eval --suite all --repeats 3 --run-name grounding-v1
  --repeats applies to unanswerable / false_premise questions (reliability across runs)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import pandas as pd

from ..agent.graph import ask, get_agent
from ..config import get_settings
from ..llm import ensure_embed_model_loaded
from ..tracing import init_mlflow
from . import dataset
from .metrics import ANSWERABLE_KINDS, score

GROUNDING = Path("data/eval/grounding.jsonl")
RETRIEVAL = Path("data/eval/retrieval.jsonl")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_suite(suite: str, repeats: int) -> list[dict]:
    rows: list[dict] = []
    if suite in ("answers", "all"):
        rows += dataset.load()
    if suite in ("grounding", "all"):
        for ex in _jsonl(GROUNDING):
            rows += [{**ex, "repeat": r} for r in range(repeats)]
        rows += [{"question": ex["question"], "kind": "answerable", "expected": None} for ex in _jsonl(RETRIEVAL)]
    return [{"repeat": 0, **r} for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["answers", "grounding", "all"], default="answers")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--kind")
    ap.add_argument("--run-name", default="eval")
    a = ap.parse_args()

    ensure_embed_model_loaded()
    init_mlflow()
    s = get_settings()
    examples = load_suite(a.suite, a.repeats)
    if a.kind:
        examples = [e for e in examples if e["kind"] == a.kind]
    examples = examples[: a.limit] if a.limit else examples
    get_agent()

    rows = []
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params({"suite": a.suite, "repeats": a.repeats, "llm_model": s.llm_model,
                           "embed_model": s.embed_model, "retrieve_k": s.retrieve_k, "chunk_size": s.chunk_size,
                           "max_refinements": s.max_refinements, "temperature": s.llm_temperature,
                           "top_p": s.llm_top_p, "top_k": s.llm_top_k, "n_examples": len(examples),
                           "compiled_program": str(Path(s.compiled_program_path).exists())})
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            base = {"question": ex["question"], "kind": ex["kind"], "repeat": ex["repeat"],
                    "expected": json.dumps(ex.get("expected"))}
            try:
                res = ask(ex["question"])
                sc = score(ex, res["answer"], res["status"])
                asm = res.get("assessment") or {}
                rows.append({**base, "status": res["status"], "answer": res["answer"], **sc,
                             "grounded": float(bool(asm.get("grounded"))), "complete": float(bool(asm.get("complete"))),
                             "refinements": sum(st["node"] == "refine_search" for st in res["steps"]),
                             "seconds": round(time.time() - t0, 1), "error": ""})
            except Exception as e:
                rows.append({**base, "status": "error", "answer": "", "correct": 0.0, "cited": 0.0, "refused": 0.0,
                             "false_refusal": 0.0, "grounded": 0.0, "complete": 0.0, "refinements": 0,
                             "seconds": round(time.time() - t0, 1), "error": repr(e)})
            r = rows[-1]
            print(f"[{i}/{len(examples)}] {r['kind']:<13} status={r['status']:<9} correct={r['correct']} "
                  f"{r['seconds']}s  {ex['question'][:65]}", flush=True)

        df = pd.DataFrame(rows)
        mlflow.log_table(df, "eval_results.json")
        metrics = _metrics(df)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        print(pd.Series(metrics).round(3).to_string())


def _metrics(df: pd.DataFrame) -> dict:
    m = {"error_rate": (df["error"] != "").mean(), "mean_seconds": df["seconds"].mean(),
         "retry_rate": (df["refinements"] > 0).mean()}
    ans = df[df["kind"].isin(ANSWERABLE_KINDS)]
    if len(ans):
        m["false_refusal_rate"] = ans["false_refusal"].mean()          # answerable question got "not found"
        m["answerable_partial_rate"] = (ans["status"] == "partial").mean()  # answered + "part not found" note
        m["cited_rate"] = ans["cited"].mean()
        m["grounded_rate"] = ans["grounded"].mean()
    for kind in ("numeric", "keywords", "unanswerable", "false_premise"):
        sub = df[df["kind"] == kind]
        if len(sub):
            m[f"{kind}_accuracy"] = sub["correct"].astype(float).mean()
    for kind in ("unanswerable", "false_premise"):
        sub = df[df["kind"] == kind]
        if len(sub):
            m[f"{kind}_refused_rate"] = sub["refused"].mean()
            # shows up in the MLflow review list (tags.answer_status in not_found / partial)
            m[f"{kind}_review_logged_rate"] = sub["status"].isin(["not_found", "partial"]).mean()
            # reliability: share of questions whose pass/fail is identical across all repeats
            per_q = sub.groupby("question")["correct"].agg(lambda s: s.nunique() == 1)
            if sub["repeat"].nunique() > 1:
                m[f"{kind}_consistency"] = per_q.mean()
    return m


if __name__ == "__main__":
    main()
