"""List questions the agent refused (not_found) or only partly answered, from MLflow traces.

python -m sec_agent.eval.unanswered [--status not_found|partial|all] [--limit 50]

Each row shows the question, what the self-check said was missing, and the trace id. Open the trace in
the MLflow UI (http://localhost:5000, experiment sec-10k-agent, Traces tab) to see the rejected draft,
the retrieved sources and every step. Same filter in the UI: tags.answer_status = 'not_found'
"""
from __future__ import annotations

import argparse

import mlflow

from ..config import get_settings


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", choices=["not_found", "partial", "all"], default="all")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args()

    s = get_settings()
    mlflow.set_tracking_uri(s.mlflow_tracking_uri)
    exp = mlflow.get_experiment_by_name(s.mlflow_experiment)
    statuses = ["not_found", "partial"] if a.status == "all" else [a.status]
    found = 0
    for st in statuses:
        traces = mlflow.search_traces(locations=[exp.experiment_id], filter_string=f"tags.answer_status = '{st}'",
                                      max_results=a.limit, order_by=["timestamp_ms DESC"], return_type="list")
        for t in traces:
            tags = t.info.tags
            found += 1
            print(f"[{st}] {tags.get('question', '?')}\n    missing: {tags.get('missing', '')[:200]}\n"
                  f"    trace: {t.info.trace_id}  retries: {tags.get('refinements')}")
    print(f"\n{found} trace(s)")


if __name__ == "__main__":
    main()
