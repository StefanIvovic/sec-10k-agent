"""Run the agent over the eval set and log results to MLflow.

python -m sec_agent.eval.run_eval [--limit 10] [--kind numeric]
"""
from __future__ import annotations

import argparse
import time

import mlflow
import pandas as pd

from ..agent.graph import ask, get_agent
from ..config import get_settings
from ..tracing import init_mlflow
from . import dataset
from .metrics import score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--kind", choices=["numeric", "keywords", "open"])
    ap.add_argument("--run-name", default="eval")
    a = ap.parse_args()

    init_mlflow()
    s = get_settings()
    examples = dataset.load()
    if a.kind:
        examples = [e for e in examples if e["kind"] == a.kind]
    examples = examples[: a.limit] if a.limit else examples
    get_agent()

    rows = []
    with mlflow.start_run(run_name=a.run_name):
        mlflow.log_params({"llm_model": s.llm_model, "embed_model": s.embed_model, "retrieve_k": s.retrieve_k,
                           "chunk_size": s.chunk_size, "max_refinements": s.max_refinements,
                           "compiled_program": str(__import__("pathlib").Path(s.compiled_program_path).exists()),
                           "n_examples": len(examples)})
        for i, ex in enumerate(examples, 1):
            t0 = time.time()
            try:
                res = ask(ex["question"])
                sc = score(ex, res["answer"])
                a_ = res.get("assessment") or {}
                rows.append({**ex, "expected": str(ex["expected"]), "answer": res["answer"], **sc,
                             "grounded": float(bool(a_.get("grounded"))), "complete": float(bool(a_.get("complete"))),
                             "seconds": round(time.time() - t0, 1), "error": ""})
            except Exception as e:
                rows.append({**ex, "expected": str(ex["expected"]), "answer": "", "correct": 0.0, "cited": 0.0,
                             "grounded": 0.0, "complete": 0.0, "seconds": round(time.time() - t0, 1), "error": repr(e)})
            r = rows[-1]
            print(f"[{i}/{len(examples)}] {r['kind']:<8} correct={r['correct']} cited={r['cited']} "
                  f"{r['seconds']}s  {ex['question'][:70]}")

        df = pd.DataFrame(rows)
        mlflow.log_table(df, "eval_results.json")
        metrics = {
            "cited_rate": df["cited"].mean(),
            "grounded_rate": df["grounded"].mean(),
            "complete_rate": df["complete"].mean(),
            "mean_seconds": df["seconds"].mean(),
            "error_rate": (df["error"] != "").mean(),
        }
        for kind in ("numeric", "keywords"):
            sub = df[df["kind"] == kind]
            if len(sub):
                metrics[f"{kind}_accuracy"] = sub["correct"].astype(float).mean()
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        print(pd.Series(metrics).round(3).to_string())


if __name__ == "__main__":
    main()
