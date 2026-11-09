"""Optimize the agent's DSPy prompts (few-shot demos) against the eval set and log to MLflow.

python -m sec_agent.eval.optimize [--train 12]

The optimizer runs a linear version of the agent (analyze -> retrieve -> answer) so DSPy can trace
every predictor call in one thread. The compiled program is saved to `compiled_program_path` and
picked up automatically by the LangGraph agent on the next start.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import dspy
import mlflow

from ..agent import tools
from ..agent.signatures import SecQAProgram
from ..config import get_settings
from ..llm import configure_dspy
from ..tracing import init_mlflow
from . import cases
from .metrics import check, has_citation


class LinearQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.program = SecQAProgram()

    def forward(self, question: str):
        known = tools.describe_companies()
        p = self.program.analyze(question=question, available_companies=known,
                                 available_metrics=tools.describe_metrics())
        plan = tools.normalize_plan(p, question, known)
        hits = tools.search_filings(plan["search_queries"], plan["tickers"], plan["fiscal_years"], plan["items"])
        facts = tools.get_financials(plan["tickers"], plan["metrics"], plan["fiscal_years"])
        sources, _ = tools.format_sources(hits[:get_settings().retrieve_k], facts, [])
        return self.program.answer(question=question, sources=sources)


def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    ans = pred.answer or ""
    row = {k: getattr(example, k) for k in ("expect", "number", "contains") if k in example}
    correct = float(check(row, ans, "answered")[0])
    score = 0.8 * correct + 0.2 * has_citation(ans)
    return score >= 0.9 if trace is not None else score  # strict when bootstrapping demos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--demos", type=int, default=3)
    a = ap.parse_args()

    init_mlflow()
    configure_dspy()
    s = get_settings()
    # dev only (never test); questions with a checkable answer
    rows = [{k: v for k, v in r.items() if k in ("question", "expect", "number", "contains")}
            for r in cases.load("dev") if r["expect"] == "answer" and ("number" in r or r.get("contains"))]
    random.Random(0).shuffle(rows)
    examples = [dspy.Example(**r).with_inputs("question") for r in rows]
    train, dev = examples[: a.train], examples[a.train:]

    with mlflow.start_run(run_name="dspy-optimize"):
        mlflow.log_params({"optimizer": "BootstrapFewShot", "train": len(train), "dev": len(dev), "demos": a.demos})
        evaluate = dspy.Evaluate(devset=dev, metric=metric, display_progress=True, num_threads=1)
        if dev:
            mlflow.log_metric("dev_score_baseline", float(evaluate(LinearQA()).score))
        opt = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=a.demos, max_labeled_demos=0)
        compiled = opt.compile(LinearQA(), trainset=train)
        if dev:
            mlflow.log_metric("dev_score_compiled", float(evaluate(compiled).score))
        path = Path(s.compiled_program_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        compiled.program.save(str(path))
        mlflow.log_artifact(str(path))
        print(f"saved {path}")


if __name__ == "__main__":
    main()
